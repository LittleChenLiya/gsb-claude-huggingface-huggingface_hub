# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end tests for resumable download block verification.

These tests run a real local HTTP server (stdlib ``http.server`` in a background thread) that simulates the
failure modes a resumable download must recover from:

- the stream being cut mid-file (``/cut``), resumed with a ``Range`` request;
- a corrupted block served on the first request and the correct block served afterwards (``/corrupt-block``),
  repaired with a single re-download of that block only;
- the resource being updated, i.e. a different ETag + body, while a partial file exists (``/changing``), which
  must discard the stale partial file;
- an always-corrupted block (``/always-corrupt``), which must end in an error instead of a corrupt cache file;
- a truncated body (``/short``), where the temporary file must be cleaned up and must never become the
  destination.

The tests are fully local, repeatable and do not require Docker or any external service.
"""

import hashlib
import http.server
import io
import socketserver
import threading
from pathlib import Path

import pytest

from huggingface_hub.file_download import _download_to_tmp_and_move, http_get


# Small blocks keep the test fast while still exercising multiple blocks, including a shorter last block.
BLOCK_SIZE = 256
N_FULL_BLOCKS = 4
CONTENT = (bytes(range(256)) * N_FULL_BLOCKS) + b"Z" * 137  # 4 full blocks + a shorter last block (1161 bytes)


def block_checksums_of(content: bytes, block_size: int) -> tuple[str, ...]:
    return tuple(hashlib.sha256(content[i : i + block_size]).hexdigest() for i in range(0, len(content), block_size))


EXPECTED_CHECKSUMS = block_checksums_of(CONTENT, BLOCK_SIZE)
ETAG_V1 = "abcdef0123456789abcdef0123456789abcdef01"
ETAG_V2 = "0123456789abcdef0123456789abcdef01234567"
NEW_CONTENT = b"new-version-of-the-file" * 50  # 1150 bytes, served with ETAG_V2


def parse_range(range_header: str | None, length: int) -> tuple[int, int]:
    """Parse a simple ``bytes=START-`` or ``bytes=START-END`` header and return ``(start, end)`` with end exclusive."""
    if not range_header:
        return 0, length
    spec = range_header.split("=", 1)[1]
    start, end = spec.split("-", 1)
    start = int(start)
    end = int(end) + 1 if end else length
    return start, end


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler with scripted per-route behavior driven by a shared state object."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence the server
        pass

    @property
    def state(self) -> dict:
        return self.server.state  # type: ignore[attr-defined]

    def _serve(self, body: bytes, *, etag: str, status: int = 200, full_length: int | None = None) -> None:
        start = 0
        if status == 206 and full_length is not None:
            range_header = self.headers.get("Range")
            start, _ = parse_range(range_header, full_length)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if status == 206 and full_length is not None:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{full_length}")
        self.send_header("ETag", f'"{etag}"')
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _cut_after(self, body: bytes, n_bytes: int, *, etag: str, full_length: int) -> None:
        """Advertise the full body length but only send `n_bytes`, then hard-close the underlying socket.

        This simulates a real mid-transfer network outage: the client receives fewer bytes than announced and
        reads an incomplete message (``RemoteProtocolError``).
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(full_length))
        self.send_header("ETag", f'"{etag}"')
        self.end_headers()
        try:
            self.wfile.write(body[:n_bytes])
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        # Hard close the TCP connection without sending the remaining bytes.
        self.close_connection = True
        try:
            self.connection.shutdown(2)  # SHUT_RDWR
        except OSError:
            pass
        self.connection.close()

    def do_GET(self) -> None:
        route = self.path
        body = self.state["body"]
        etag = self.state["etag"]
        start, end = parse_range(self.headers.get("Range"), len(body))
        is_range = self.headers.get("Range") is not None

        if route == "/cut":
            # First non-ranged request: cut the stream after 300 bytes (mid-block-2).
            # Subsequent (resume) requests: serve the correct suffix for the requested range.
            request_no = self.state["counts"].get("/cut", 0)
            self.state["counts"]["/cut"] = request_no + 1
            if request_no == 0 and not is_range:
                self._cut_after(body, 300, etag=etag, full_length=len(body))
                return
            self._serve(body[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))
            return

        if route == "/corrupt-block":
            # First full request: serve block index 1 with flipped bytes (block 1 corrupted), rest correct.
            # All later requests (resume or ranged repair): serve the correct bytes.
            request_no = self.state["counts"].get("/corrupt-block", 0)
            self.state["counts"]["/corrupt-block"] = request_no + 1
            if request_no == 0 and not is_range:
                corrupted = bytearray(body)
                corrupted[BLOCK_SIZE : 2 * BLOCK_SIZE] = b"X" * BLOCK_SIZE
                self._serve(bytes(corrupted), etag=etag, full_length=len(body))
                return
            self._serve(body[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))
            return

        if route == "/always-corrupt":
            # Block index 2 is always corrupted, including on the repair range request.
            if is_range and start == 2 * BLOCK_SIZE:
                self._serve(b"Q" * (end - start), etag=etag, status=206, full_length=len(body))
                return
            corrupted = bytearray(body)
            corrupted[2 * BLOCK_SIZE : 3 * BLOCK_SIZE] = b"Q" * BLOCK_SIZE
            self._serve(bytes(corrupted)[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))
            return

        if route == "/changing":
            # A partial file for the old ETag exists; the server now serves the updated resource on every request.
            self._serve(body[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))
            return

        if route == "/cut-then-update":
            # First GET: cut after 300 bytes with ETAG_V1. Any resume GET afterwards: the resource has been
            # updated to ETAG_V2 + NEW_CONTENT (ranges are computed against the current body, like /changing).
            request_no = self.state["counts"].get("/cut-then-update", 0)
            self.state["counts"]["/cut-then-update"] = request_no + 1
            if request_no == 0 and not is_range:
                self._cut_after(CONTENT, 300, etag=ETAG_V1, full_length=len(CONTENT))
                return
            # Flip the server state: an in-process retry now sees the new version.
            self.state["body"] = NEW_CONTENT
            self.state["etag"] = ETAG_V2
            body = NEW_CONTENT
            start, end = parse_range(self.headers.get("Range"), len(body))
            self._serve(body[start:end], etag=ETAG_V2, status=206 if is_range else 200, full_length=len(body))
            return

        if route == "/short":
            # Advertises the full length but serves a truncated body.
            truncated = body[: len(body) // 2]
            self._serve(truncated[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))
            return

        # default ("/ok"): serve the correct content
        self._serve(body[start:end], etag=etag, status=206 if is_range else 200, full_length=len(body))


class _ThreadingHTTPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


@pytest.fixture
def fixture_server():
    server = _ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    server.state = {"body": CONTENT, "etag": ETAG_V1, "counts": {}}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Do not wait between retries in the tests."""
    monkeypatch.setattr("huggingface_hub.file_download.time.sleep", lambda *_: None)


def _url(server, route: str) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{route}"


def _assert_final_file(path: Path, expected: bytes = CONTENT) -> None:
    assert path.exists(), "destination file must exist"
    data = path.read_bytes()
    assert data == expected, "final file content does not match the expected content"


# ---------------------------------------------------------------------------
# 1. Stream cut -> resume from a Range request -> block verification passes
# ---------------------------------------------------------------------------
def test_resume_after_stream_cut_with_block_verification(tmp_path, fixture_server):
    tmp_file = tmp_path / "file.bin"
    with tmp_file.open("w+b") as f:
        http_get(
            _url(fixture_server, "/cut"),
            f,
            expected_size=len(CONTENT),
            block_checksums=EXPECTED_CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    _assert_final_file(tmp_file)
    # exactly one initial request + one resume request
    assert fixture_server.state["counts"]["/cut"] == 2


# ---------------------------------------------------------------------------
# 2. Corrupted block served on the full download -> only that block is re-pulled
# ---------------------------------------------------------------------------
def test_corrupted_block_is_redownloaded_only(tmp_path, fixture_server):
    tmp_file = tmp_path / "file.bin"
    with tmp_file.open("w+b") as f:
        http_get(
            _url(fixture_server, "/corrupt-block"),
            f,
            expected_size=len(CONTENT),
            block_checksums=EXPECTED_CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    _assert_final_file(tmp_file)
    # 1 full GET + 1 ranged repair GET for block index 1
    assert fixture_server.state["counts"]["/corrupt-block"] == 2


# ---------------------------------------------------------------------------
# 3. Pre-existing partial file with a corrupted block -> resume then repair only that block
# ---------------------------------------------------------------------------
def test_resume_with_corrupt_existing_block_repaired(tmp_path, fixture_server):
    """A partial file already on disk (300 bytes) contains a corrupted block 0. Resuming the stream from byte 300
    must not trust the on-disk prefix: after completion block verification re-pulls only the corrupted block."""
    partial = bytearray(CONTENT[:300])
    partial[:BLOCK_SIZE] = b"X" * BLOCK_SIZE
    tmp_file = tmp_path / "file.bin"
    tmp_file.write_bytes(bytes(partial))

    # /cut counts the first non-ranged GET as the cut request: emulate that it already happened.
    fixture_server.state["counts"]["/cut"] = 1

    with tmp_file.open("rb+") as f:
        http_get(
            _url(fixture_server, "/cut"),
            f,
            resume_size=300,
            expected_size=len(CONTENT),
            block_checksums=EXPECTED_CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    _assert_final_file(tmp_file)


# ---------------------------------------------------------------------------
# 4. ETag changed while a partial file exists -> stale fragments are discarded
# ---------------------------------------------------------------------------
def test_etag_change_discards_old_fragments(tmp_path, fixture_server):
    # Partial file downloaded for the old ETag (old full size = len(CONTENT)).
    tmp_file = tmp_path / "file.bin"
    partial = b"old-prefix-of-v1" + b"\x00" * 50
    tmp_file.write_bytes(partial)

    # Server now serves the new version.
    fixture_server.state["body"] = NEW_CONTENT
    fixture_server.state["etag"] = ETAG_V2

    with tmp_file.open("rb+") as f:
        http_get(
            _url(fixture_server, "/changing"),
            f,
            resume_size=len(partial),
            # Caller still believes in the old metadata (etag v1, full size len(CONTENT)).
            expected_size=len(CONTENT),
            expected_etag=ETAG_V1,
        )

    # The file is exactly the new version, no stale bytes left.
    _assert_final_file(tmp_file, expected=NEW_CONTENT)


def test_etag_change_guard_during_full_download(tmp_path, fixture_server):
    """Even on a fresh GET, a response ETag different from the metadata ETag must not be accepted as-is: the
    download transparently restarts against the new version rather than silently storing mismatched content."""
    fixture_server.state["body"] = NEW_CONTENT
    fixture_server.state["etag"] = ETAG_V2
    tmp_file = tmp_path / "file.bin"
    with tmp_file.open("w+b") as f:
        http_get(
            _url(fixture_server, "/changing"),
            f,
            expected_size=len(NEW_CONTENT),
            expected_etag=ETAG_V1,  # metadata says v1, server already serves v2
        )
    _assert_final_file(tmp_file, expected=NEW_CONTENT)


def test_etag_change_between_retry_gets_restarts(tmp_path, fixture_server):
    """In-process resume: the first GET starts downloading v1 and gets cut; the retry GET observes v2. The
    download must detect the ETag change, discard the v1 prefix and end with exactly the v2 content."""
    tmp_file = tmp_path / "file.bin"
    with tmp_file.open("w+b") as f:
        http_get(
            _url(fixture_server, "/cut-then-update"),
            f,
            expected_size=len(CONTENT),
            expected_etag=ETAG_V1,
        )
    _assert_final_file(tmp_file, expected=NEW_CONTENT)


# ---------------------------------------------------------------------------
# 5. A block that stays corrupted must raise, never produce a "completed" file
# ---------------------------------------------------------------------------
def test_permanently_corrupt_block_raises(tmp_path, fixture_server):
    tmp_file = tmp_path / "file.bin"
    with tmp_file.open("w+b") as f, pytest.raises(OSError, match="Consistency check failed"):
        http_get(
            _url(fixture_server, "/always-corrupt"),
            f,
            expected_size=len(CONTENT),
            block_checksums=EXPECTED_CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    assert tmp_file.read_bytes()[2 * BLOCK_SIZE : 3 * BLOCK_SIZE] == b"Q" * BLOCK_SIZE


# ---------------------------------------------------------------------------
# 6. Truncated body via _download_to_tmp_and_move: no destination, no leftover temp file
# ---------------------------------------------------------------------------
def test_tmp_file_error_is_not_published_as_cache(tmp_path, fixture_server):
    incomplete_path = tmp_path / "file.incomplete"
    destination_path = tmp_path / "file.bin"
    with pytest.raises(OSError):
        _download_to_tmp_and_move(
            incomplete_path=incomplete_path,
            destination_path=destination_path,
            url_to_download=_url(fixture_server, "/short"),
            headers={},
            expected_size=len(CONTENT),
            filename="file.bin",
            force_download=False,
            etag=ETAG_V1,
            xet_file_data=None,
        )
    assert not destination_path.exists(), "a failed download must never become the cached blob"
    leftover = list(tmp_path.glob("*.incomplete"))
    assert leftover == [], f"temporary files must be cleaned up, found {leftover}"


def test_successful_download_publishes_destination(tmp_path, fixture_server):
    incomplete_path = tmp_path / "file.bin.incomplete"
    destination_path = tmp_path / "file.bin"
    _download_to_tmp_and_move(
        incomplete_path=incomplete_path,
        destination_path=destination_path,
        url_to_download=_url(fixture_server, "/ok"),
        headers={},
        expected_size=len(CONTENT),
        filename="file.bin",
        force_download=False,
        etag=ETAG_V1,
        xet_file_data=None,
    )
    _assert_final_file(destination_path)
    assert not incomplete_path.exists()
    assert list(tmp_path.glob("*.incomplete")) == []


# ---------------------------------------------------------------------------
# 7. Argument validation
# ---------------------------------------------------------------------------
def test_invalid_block_arguments_raise():
    with pytest.raises(ValueError, match="`block_size` must be a positive integer"):
        http_get("http://x", io.BytesIO(), expected_size=10, block_checksums=("a",))
    with pytest.raises(ValueError, match="must be a 64-character hex-encoded SHA-256 digest"):
        http_get(
            "http://x",
            io.BytesIO(),
            expected_size=10,
            block_checksums=("not-a-hash",),
            block_size=10,
        )
    with pytest.raises(ValueError, match="Invalid block checksums"):
        http_get(
            "http://x",
            io.BytesIO(),
            expected_size=30,
            block_checksums=(hashlib.sha256(b"a").hexdigest(),),  # 3 blocks needed for 30 bytes, 1 given
            block_size=10,
        )
