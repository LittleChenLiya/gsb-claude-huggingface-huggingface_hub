#!/usr/bin/env python3
"""Minimal, repeatable end-to-end example for resumable download block verification.

Run with:

    python examples/resumable_block_download_example.py

What it demonstrates, all against a local HTTP server started by this script (no Docker, no network):

1. `/cut`           - the stream is cut mid-file; the download resumes with a Range request.
2. `/corrupt-block` - one block is served corrupted on the first GET; only that block is re-pulled.
3. `/changing`      - the resource is updated (new ETag + new body) while a partial file exists; the
                      stale fragments are discarded and the new version is downloaded from scratch.

After each scenario the final file is compared byte-for-byte with the expected content and a
verdict is printed. Exit code is 0 only if every scenario produced the exact expected file.
"""

from __future__ import annotations

import hashlib
import http.server
import socketserver
import sys
import tempfile
import threading
from pathlib import Path


# Make `src/` importable when the example is run straight from a repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from huggingface_hub.file_download import http_get  # noqa: E402


BLOCK_SIZE = 256
CONTENT = (bytes(range(256)) * 4) + b"Z" * 137  # 4 full blocks + a shorter last block (1161 bytes)
ETAG_V1 = "abcdef0123456789abcdef0123456789abcdef01"
ETAG_V2 = "0123456789abcdef0123456789abcdef01234567"
NEW_CONTENT = b"new-version-of-the-file" * 50  # served with ETAG_V2

CHECKSUMS = tuple(hashlib.sha256(CONTENT[i : i + BLOCK_SIZE]).hexdigest() for i in range(0, len(CONTENT), BLOCK_SIZE))


def _range_bounds(range_header: str | None, length: int) -> tuple[int, int]:
    if not range_header:
        return 0, length
    start, end = range_header.split("=", 1)[1].split("-", 1)
    return int(start), int(end) + 1 if end else length


class ExampleHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        pass

    @property
    def state(self) -> dict:
        return self.server.state  # type: ignore[attr-defined]

    def _send(self, body: bytes, *, status: int = 200, full_length: int | None = None) -> None:
        start = 0
        if status == 206 and full_length is not None:
            start, _ = _range_bounds(self.headers.get("Range"), full_length)
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if status == 206 and full_length is not None:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{full_length}")
        self.send_header("ETag", f'"{self.state["etag"]}"')
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _cut(self, n_bytes: int) -> None:
        body = self.state["body"]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", f'"{self.state["etag"]}"')
        self.end_headers()
        self.wfile.write(body[:n_bytes])
        self.wfile.flush()
        self.close_connection = True
        try:
            self.connection.shutdown(2)  # hard-close => client reads an incomplete message
        except OSError:
            pass
        self.connection.close()

    def do_GET(self) -> None:  # noqa: N802
        body = self.state["body"]
        start, end = _range_bounds(self.headers.get("Range"), len(body))
        ranged = self.headers.get("Range") is not None

        if self.path == "/cut":
            n = self.state["count"].get("/cut", 0)
            self.state["count"]["/cut"] = n + 1
            if n == 0 and not ranged:
                self._cut(300)  # cut mid-block-2 on the first full request
                return
            self._send(body[start:end], status=206 if ranged else 200, full_length=len(body))
            return

        if self.path == "/corrupt-block":
            n = self.state["count"].get("/corrupt-block", 0)
            self.state["count"]["/corrupt-block"] = n + 1
            if n == 0 and not ranged:
                corrupted = bytearray(body)
                corrupted[BLOCK_SIZE : 2 * BLOCK_SIZE] = b"X" * BLOCK_SIZE  # block 1 corrupted
                self._send(bytes(corrupted), full_length=len(body))
                return
            self._send(body[start:end], status=206 if ranged else 200, full_length=len(body))
            return

        # /changing (and anything else): serve the current body/etag verbatim
        self._send(body[start:end], status=206 if ranged else 200, full_length=len(body))


class ExampleServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_server(body: bytes, etag: str) -> ExampleServer:
    server = ExampleServer(("127.0.0.1", 0), ExampleHandler)
    server.state = {"body": body, "etag": etag, "count": {}}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _url(server: ExampleServer, route: str) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}{route}"


def scenario_stream_cut(workdir: Path) -> bool:
    server = _start_server(CONTENT, ETAG_V1)
    target = workdir / "cut.bin"
    with target.open("w+b") as f:
        http_get(
            _url(server, "/cut"),
            f,
            expected_size=len(CONTENT),
            block_checksums=CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    ok = target.read_bytes() == CONTENT
    print(f"[1] stream cut + resume + block verification: {'OK' if ok else 'FAIL'}")
    server.shutdown()
    return ok


def scenario_corrupt_block(workdir: Path) -> bool:
    server = _start_server(CONTENT, ETAG_V1)
    target = workdir / "corrupt.bin"
    with target.open("w+b") as f:
        http_get(
            _url(server, "/corrupt-block"),
            f,
            expected_size=len(CONTENT),
            block_checksums=CHECKSUMS,
            block_size=BLOCK_SIZE,
            expected_etag=ETAG_V1,
        )
    ok = target.read_bytes() == CONTENT
    print(
        f"[2] corrupted block re-pulled individually: {'OK' if ok else 'FAIL'} "
        f"(1 full GET + 1 ranged repair = {server.state['count']['/corrupt-block']} requests)"
    )
    server.shutdown()
    return ok


def scenario_etag_update(workdir: Path) -> bool:
    server = _start_server(NEW_CONTENT, ETAG_V2)  # server already serves the new version
    target = workdir / "changing.bin"
    partial = b"old-prefix-of-v1" + b"\x00" * 50
    target.write_bytes(partial)  # stale partial file from the v1 download
    with target.open("rb+") as f:
        http_get(
            _url(server, "/changing"),
            f,
            resume_size=len(partial),
            expected_size=len(CONTENT),  # metadata still says v1...
            expected_etag=ETAG_V1,
        )
    ok = target.read_bytes() == NEW_CONTENT
    print(f"[3] ETag update discards stale fragments: {'OK' if ok else 'FAIL'}")
    server.shutdown()
    return ok


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        results = [
            scenario_stream_cut(workdir),
            scenario_corrupt_block(workdir),
            scenario_etag_update(workdir),
        ]
    if all(results):
        print("\nAll scenarios verified the expected final file.")
        return 0
    print("\nAt least one scenario produced an unexpected file.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
