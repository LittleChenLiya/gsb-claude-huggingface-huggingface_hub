# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Local HTTP fixture tests for resumable downloads with per-block checksum verification.

These tests use a stdlib HTTP server (no Docker) to simulate, against a real
``hf_hub_download`` call:

- stream interruption (connection dropped mid-body) -> resume with HTTP Range;
- a corrupted block in the partial download -> only that block is re-fetched;
- an ETag change between attempts -> the partial file is discarded and the
  download restarts from scratch;
- a persistent temp-file error -> the failed download is never promoted to the
  completed cache.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from huggingface_hub import constants, hf_hub_download
from huggingface_hub._block_checksum import BlockChecksumManifest, manifest_path_for
from huggingface_hub.file_download import repo_folder_name

REPO_ID = "user/repo"
FILENAME = "model.bin"
REVISION = "main"
COMMIT_HASH = "a" * 40
BLOCK_SIZE = 16
CONTENT = bytes(range(100))  # 100 bytes -> 7 blocks (6 full + 1 short)


def _etag_for(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class _DownloadState:
    """Shared, thread-safe state for the HTTP fixture."""

    def __init__(self, content: bytes) -> None:
        self.lock = threading.Lock()
        self.content = content
        self.etag = _etag_for(content)
        self.drop_after: int | None = None  # one-shot: next full GET drops after N bytes
        self.fail_gets: int = 0  # return 503 for this many GET requests
        self.request_log: list[tuple[str, str, str | None, int]] = []

    def update(self, content: bytes) -> None:
        with self.lock:
            self.content = content
            self.etag = _etag_for(content)

    def consume_drop(self) -> int | None:
        with self.lock:
            value = self.drop_after
            self.drop_after = None
            return value

    def consume_fail(self) -> bool:
        with self.lock:
            if self.fail_gets > 0:
                self.fail_gets -= 1
                return True
            return False

    def log(self, method: str, path: str, range_header: str | None, status: int) -> None:
        with self.lock:
            self.request_log.append((method, path, range_header, status))


class _FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # silence stderr
        pass

    def _parse_target(self) -> tuple[str, str, str]:
        # /user/repo/resolve/main/model.bin -> ("user/repo", "main", "model.bin")
        path = self.path.split("?", 1)[0]
        left, right = path.split("/resolve/", 1)
        repo_id = left.lstrip("/")
        revision, filename = right.split("/", 1)
        return repo_id, revision, filename

    def _send_etag_headers(self, content_length: int) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        self.send_header("ETag", f'"{state.etag}"')
        self.send_header("x-linked-etag", f'"{state.etag}"')
        self.send_header("x-repo-commit", COMMIT_HASH)
        self.send_header("x-linked-size", str(content_length))

    def do_HEAD(self) -> None:
        self._parse_target()
        with self.server.state.lock:  # type: ignore[attr-defined]
            content = self.server.state.content  # type: ignore[attr-defined]
        self.send_response(200)
        self._send_etag_headers(len(content))
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.server.state.log("HEAD", self.path, None, 200)  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        self._parse_target()
        state = self.server.state  # type: ignore[attr-defined]
        range_header = self.headers.get("Range")

        if state.consume_fail():
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            state.log("GET", self.path, range_header, 503)
            return

        with state.lock:
            content = state.content

        drop_after = state.consume_drop()
        if drop_after is not None and range_header is None:
            # Simulate a dropped connection: send a partial body then close the socket.
            self.send_response(200)
            self._send_etag_headers(len(content))
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content[:drop_after])
            self.wfile.flush()
            self.close_connection = True
            state.log("GET", self.path, range_header, 200)
            return

        # Parse the Range header (supports "bytes=start-end" and "bytes=start-").
        start, end = 0, len(content) - 1
        status = 200
        if range_header is not None and range_header.startswith("bytes="):
            spec = range_header[len("bytes=") :]
            if spec.startswith("-"):
                length = int(spec[1:])
                start = max(0, len(content) - length)
            elif "-" in spec:
                s, e = spec.split("-", 1)
                start = int(s)
                end = int(e) if e else len(content) - 1
            status = 206

        body = content[start : end + 1]
        self.send_response(status)
        self._send_etag_headers(len(content))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        state.log("GET", self.path, range_header, status)


@pytest.fixture
def fixture_server():
    state = _DownloadState(CONTENT)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureHandler)
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _enable_block_checksum(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(constants, "HF_HUB_BLOCK_CHECKSUM_ENABLED", True)
    monkeypatch.setattr(constants, "HF_HUB_BLOCK_CHECKSUM_BLOCK_SIZE", BLOCK_SIZE)
    # Retries are instant in tests.
    monkeypatch.setattr("huggingface_hub.file_download.time.sleep", lambda _: None)


def _endpoint(server: ThreadingHTTPServer) -> str:
    host, port = server.server_address  # type: ignore[misc]
    return f"http://{host}:{port}"


def _blob_path(cache_dir: Path, etag: str) -> Path:
    storage = cache_dir / repo_folder_name(repo_id=REPO_ID, repo_type="model")
    return storage / "blobs" / etag


def _build_partial(
    cache_dir: Path,
    etag: str,
    content: bytes,
    partial_size: int,
    *,
    etag_for_manifest: str | None = None,
) -> tuple[Path, Path]:
    """Create a partial download temp file + block-checksum sidecar in the cache."""
    blob_path = _blob_path(cache_dir, etag)
    tmp_path = Path(str(blob_path) + ".incomplete")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_bytes(content[:partial_size])
    manifest = BlockChecksumManifest(etag=etag_for_manifest or etag, block_size=BLOCK_SIZE)
    for i in range(0, partial_size, BLOCK_SIZE):
        manifest.observe(content[i : i + BLOCK_SIZE])
    manifest.save(manifest_path_for(tmp_path))
    return tmp_path, blob_path


def _get_request_log(server: ThreadingHTTPServer) -> list[tuple[str, str, str | None, int]]:
    return list(server.state.request_log)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resume_after_stream_interruption(fixture_server: ThreadingHTTPServer, tmp_path: Path) -> None:
    """A dropped connection mid-body is resumed with an HTTP Range request; the final file is correct."""
    fixture_server.state.drop_after = 64  # type: ignore[attr-defined]
    cache_dir = tmp_path / "cache"

    result = hf_hub_download(
        REPO_ID,
        FILENAME,
        endpoint=_endpoint(fixture_server),
        cache_dir=cache_dir,
        token=False,
    )

    assert Path(result).read_bytes() == CONTENT
    # The retry must have asked for the remaining bytes only.
    ranges = [r[2] for r in _get_request_log(fixture_server) if r[0] == "GET"]
    assert "bytes=64-" in ranges


def test_corrupted_block_is_refetched_individually(
    fixture_server: ThreadingHTTPServer, tmp_path: Path
) -> None:
    """A corrupted block in the partial download is re-fetched on its own, not the whole file."""
    cache_dir = tmp_path / "cache"
    etag = _etag_for(CONTENT)
    tmp_path_file, blob_path = _build_partial(cache_dir, etag, CONTENT, partial_size=64)

    # Corrupt block 2 (bytes 32..47) on disk, leaving the manifest untouched.
    data = bytearray(tmp_path_file.read_bytes())
    data[32:48] = b"\x00" * 16
    tmp_path_file.write_bytes(bytes(data))

    result = hf_hub_download(
        REPO_ID,
        FILENAME,
        endpoint=_endpoint(fixture_server),
        cache_dir=cache_dir,
        token=False,
    )

    assert Path(result).read_bytes() == CONTENT
    # Only the corrupted block (bytes 32..47) must have been re-fetched individually;
    # the tail (bytes 64..) is requested separately.
    ranges = [r[2] for r in _get_request_log(fixture_server) if r[0] == "GET"]
    assert "bytes=32-47" in ranges, f"corrupted block was not re-fetched individually: {ranges}"
    assert "bytes=0-" not in ranges  # no full re-download


def test_etag_change_discards_partial_download(
    fixture_server: ThreadingHTTPServer, tmp_path: Path
) -> None:
    """When the remote ETag changes, the partial file is discarded and the new version is downloaded fully."""
    cache_dir = tmp_path / "cache"
    old_content = CONTENT
    old_etag = _etag_for(old_content)
    new_content = bytes(reversed(CONTENT))
    new_etag = _etag_for(new_content)

    # Partial file (old content, old manifest) sitting at the NEW etag's temp path.
    _build_partial(cache_dir, new_etag, old_content, partial_size=64, etag_for_manifest=old_etag)

    # The remote file changes before the download resumes.
    fixture_server.state.update(new_content)  # type: ignore[attr-defined]

    result = hf_hub_download(
        REPO_ID,
        FILENAME,
        endpoint=_endpoint(fixture_server),
        cache_dir=cache_dir,
        token=False,
    )

    assert Path(result).read_bytes() == new_content
    # The stale fragments must have been discarded: a full GET (no Range) for the new version.
    gets = [r for r in _get_request_log(fixture_server) if r[0] == "GET"]
    assert any(r[2] is None for r in gets), f"no full re-download after ETag change: {gets}"


def test_failed_download_is_not_promoted_to_cache(
    fixture_server: ThreadingHTTPServer, tmp_path: Path
) -> None:
    """A download that keeps failing never moves the temp file into the completed cache."""
    cache_dir = tmp_path / "cache"
    etag = _etag_for(CONTENT)
    fixture_server.state.fail_gets = 100  # type: ignore[attr-defined]  # every GET fails

    with pytest.raises(Exception):
        hf_hub_download(
            REPO_ID,
            FILENAME,
            endpoint=_endpoint(fixture_server),
            cache_dir=cache_dir,
            token=False,
        )

    # The completed blob must not exist (only the partial .incomplete file may linger).
    blob_path = _blob_path(cache_dir, etag)
    assert not blob_path.exists(), "failed download was promoted to the cache"
