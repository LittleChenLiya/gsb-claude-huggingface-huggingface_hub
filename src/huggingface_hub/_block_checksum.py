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
"""Per-block SHA-256 digests for resumable HTTP downloads.

When a download is interrupted and resumed, the bytes already on disk are only
trustworthy if they can be verified. This module implements a small sidecar
manifest (JSON) that records one SHA-256 digest per fixed-size block of the file
being downloaded. On resume, each complete block of the partial file is
re-hashed and compared to the manifest:

- a full block whose digest mismatches is corrupted on disk => re-fetched
  individually with an HTTP Range request (only that block is re-pulled);
- a block that is short (file truncated mid-block) is left to the tail of the
  resume request, which re-downloads it and everything after it.

The manifest is tied to the file's ETag. If the ETag changes between attempts,
the remote file has changed: the partial file and its manifest are discarded
and the download restarts from scratch. A failed download is never promoted to
the completed cache (the caller only moves the temp file into place after the
whole file has been verified).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from .utils import logging

logger = logging.get_logger(__name__)

MANIFEST_VERSION = 1
MANIFEST_SUFFIX = ".blocks.json"


def manifest_path_for(tmp_path: Path) -> Path:
    """Return the path of the block-checksum sidecar for a partial download file."""
    return Path(str(tmp_path) + MANIFEST_SUFFIX)


@dataclass
class BlockChecksumManifest:
    """In-memory record of per-block SHA-256 digests for a download.

    The manifest is persisted next to the partial download file so that a later
    attempt (e.g. a new process after a crash) can verify the bytes already on
    disk.

    Args:
        etag: ETag of the file being downloaded. A manifest recorded for another
            ETag is discarded: the remote file changed and partial bytes are
            obsolete.
        block_size: Block size in bytes. The last block may be shorter.
        digests: Hex-encoded SHA-256 digests, one per complete block already
            observed. The final partial block is only recorded once the stream
            finishes (see [`~BlockChecksumManifest.finish`]).
    """

    etag: str
    block_size: int
    digests: list[str] = field(default_factory=list)
    _buffer: bytearray = field(default_factory=bytearray, repr=False)
    _observed_size: int = field(default=0, repr=False)
    _dirty: bool = field(default=False, repr=False)

    def observe(self, data: bytes) -> None:
        """Feed bytes just downloaded; record a digest every time a block fills up."""
        self._observed_size += len(data)
        self._buffer.extend(data)
        while len(self._buffer) >= self.block_size:
            block = bytes(self._buffer[: self.block_size])
            del self._buffer[: self.block_size]
            self.digests.append(hashlib.sha256(block).hexdigest())
            self._dirty = True

    def finish(self) -> None:
        """Record the final partial block (if any) once the stream is exhausted."""
        if self._buffer:
            self.digests.append(hashlib.sha256(bytes(self._buffer)).hexdigest())
            self._buffer.clear()
            self._dirty = True

    def reset(self) -> None:
        """Drop all recorded blocks. Used when the download has to restart from scratch."""
        self.digests.clear()
        self._buffer.clear()
        self._observed_size = 0
        self._dirty = True

    @property
    def verified_size(self) -> int:
        """Number of bytes covered by recorded digests.

        Equals the number of bytes observed minus the bytes still buffered (the partial block
        being filled). Once the stream finishes, the buffer is empty and this equals the file size.
        """
        return self._observed_size - len(self._buffer)

    def verify_file(self, f: BinaryIO, expected_size: int | None) -> tuple[list[int], int | None]:
        """Re-hash the blocks of a partial file against the manifest.

        Args:
            f: Open file-like object; it is seeked to 0 and read sequentially.
            expected_size: Expected total file size, if known. Used to tell a
                truncated block apart from the final (short) block.

        Returns:
            A tuple ``(corrupted, short_start)`` where:
            - ``corrupted`` is the list of block indices whose on-disk digest
              does not match the manifest (a full block with wrong bytes);
            - ``short_start`` is the first block index that could not be read in
              full because the file is truncated, or ``None`` if every recorded
              block is present in full.

            Bytes from ``short_start`` onwards are re-downloaded as the tail of
            the resume request; corrupted blocks before that are re-fetched
            individually.
        """
        corrupted: list[int] = []
        short_start: int | None = None
        f.seek(0)
        for index, expected_digest in enumerate(self.digests):
            block = f.read(self.block_size)
            if len(block) < self.block_size:
                # Short read: either the file is truncated mid-block, or this is the
                # final block of the file (which may legitimately be shorter).
                is_final_block = expected_size is not None and (index + 1) * self.block_size >= expected_size
                if not is_final_block:
                    short_start = index
                    break
                # Final short block: still verify its digest.
                if hashlib.sha256(block).hexdigest() != expected_digest:
                    corrupted.append(index)
                break
            if hashlib.sha256(block).hexdigest() != expected_digest:
                corrupted.append(index)
        return corrupted, short_start

    def save(self, path: Path) -> None:
        """Persist the manifest atomically (temp file + rename)."""
        if not self._dirty:
            return
        payload = {
            "version": MANIFEST_VERSION,
            "etag": self.etag,
            "block_size": self.block_size,
            "verified_size": self.verified_size,
            "digests": self.digests,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp_name, path)
        except OSError:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._dirty = False

    @classmethod
    def load(cls, path: Path, *, etag: str, block_size: int) -> "BlockChecksumManifest | None":
        """Load the manifest recorded for ``etag``/``block_size``.

        Returns ``None`` (and logs the reason) if the manifest is missing,
        unreadable, from an unsupported format, or tied to a different ETag or
        block size. Callers must then discard the partial file and restart the
        download from scratch: the remote file changed (ETag), the block layout
        changed, or the manifest itself is corrupt.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (FileNotFoundError, OSError, ValueError) as e:
            logger.debug(f"Could not read block checksum manifest '{path}': {e}")
            return None
        if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
            logger.info(f"Ignoring block checksum manifest '{path}': unsupported format.")
            return None
        if payload.get("etag") != etag:
            logger.info(
                f"Discarding partial download '{path}': ETag changed "
                f"({payload.get('etag')} -> {etag}); restarting from scratch."
            )
            return None
        if payload.get("block_size") != block_size:
            logger.info(f"Discarding block checksum manifest '{path}': block size changed.")
            return None
        digests = payload.get("digests")
        if not isinstance(digests, list) or not all(isinstance(d, str) for d in digests):
            logger.info(f"Ignoring block checksum manifest '{path}': invalid digests.")
            return None
        manifest = cls(etag=etag, block_size=block_size, digests=list(digests))
        # Restore the verified byte count (the on-disk manifest only records complete blocks).
        verified_size = payload.get("verified_size")
        if isinstance(verified_size, int) and verified_size >= 0:
            manifest._observed_size = verified_size
        else:
            # Older/newer layout: derive from complete blocks.
            manifest._observed_size = len(digests) * block_size
        return manifest
