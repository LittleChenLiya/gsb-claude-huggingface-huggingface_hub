# Copyright 2026-present, the HuggingFace Inc. team.
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
"""On-disk failure/success manifests for snapshot downloads.

A snapshot manifest records, for a given commit hash, which files of a snapshot were downloaded successfully and
which failed. It is written when a snapshot download completes with one or more failures, so that a later run can
resume safely: files already downloaded are skipped, failed files are retried, and the manifest is deleted once the
snapshot is complete.

The manifest also acts as a completeness marker: as long as a manifest with failed entries exists, the snapshot must
not be considered complete (e.g. when the Hub cannot be reached).

```json
{
  "format_version": 1,
  "commit_hash": "<commit hash>",
  "files": {
    "config.json": {"status": "success"},
    "model.safetensors": {"status": "failed", "error_type": "HfHubHTTPError", "error": "500 Server Error: ..."}
  }
}
```
"""

import json
import os
import tempfile
from dataclasses import dataclass

from .utils import logging
from .utils._paths import as_extended_path


logger = logging.get_logger(__name__)

SNAPSHOT_MANIFEST_FORMAT_VERSION = 1

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


@dataclass
class SnapshotFileStatus:
    """Download status of a single file in a snapshot manifest."""

    status: str  # `STATUS_SUCCESS` or `STATUS_FAILED`
    error_type: str | None = None
    error: str | None = None

    def to_json(self) -> dict:
        info: dict = {"status": self.status}
        if self.error_type is not None:
            info["error_type"] = self.error_type
        if self.error is not None:
            info["error"] = self.error
        return info

    @classmethod
    def from_json(cls, info: dict) -> "SnapshotFileStatus":
        return cls(
            status=info["status"],
            error_type=info.get("error_type"),
            error=info.get("error"),
        )


@dataclass
class SnapshotManifest:
    """Per-file download statuses of a snapshot at a given commit."""

    commit_hash: str
    files: dict[str, SnapshotFileStatus]

    def to_json(self) -> dict:
        return {
            "format_version": SNAPSHOT_MANIFEST_FORMAT_VERSION,
            "commit_hash": self.commit_hash,
            "files": {path: status.to_json() for path, status in sorted(self.files.items())},
        }

    @classmethod
    def from_json(cls, data: dict) -> "SnapshotManifest":
        return cls(
            commit_hash=data["commit_hash"],
            files={path: SnapshotFileStatus.from_json(info) for path, info in data["files"].items()},
        )

    @property
    def failed_files(self) -> dict[str, SnapshotFileStatus]:
        """Files whose download failed (path -> status)."""
        return {path: status for path, status in self.files.items() if status.status == STATUS_FAILED}

    @property
    def successful_files(self) -> dict[str, SnapshotFileStatus]:
        """Files downloaded successfully (path -> status)."""
        return {path: status for path, status in self.files.items() if status.status == STATUS_SUCCESS}


def _snapshot_manifest_path(tree_cache_folder: str, commit_hash: str) -> str:
    # Both readers and writers go through here, so they agree on the path, including on Windows where a deep
    # `local_dir`/`cache_dir` requires the extended-length form.
    return as_extended_path(os.path.join(tree_cache_folder, "manifests", f"{commit_hash}.json"))


def read_snapshot_manifest(tree_cache_folder: str, commit_hash: str) -> SnapshotManifest | None:
    """Return the snapshot manifest for a commit hash, or `None` if absent, invalid, or unreadable."""
    path = _snapshot_manifest_path(tree_cache_folder, commit_hash)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format_version") != SNAPSHOT_MANIFEST_FORMAT_VERSION:
            # Unknown format (e.g. written by a newer version) => ignore and re-download.
            return None
        manifest = SnapshotManifest.from_json(data)
        if manifest.commit_hash != commit_hash:
            return None
        return manifest
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Ignoring corrupted snapshot manifest file {path}: {e}")
        return None


def write_snapshot_manifest(tree_cache_folder: str, commit_hash: str, manifest: SnapshotManifest) -> None:
    """Atomically write a snapshot manifest to disk, ignoring write failures."""
    path = _snapshot_manifest_path(tree_cache_folder, commit_hash)
    data = manifest.to_json()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(f"Ignored error while writing snapshot manifest file {path}: {e}")


def clear_snapshot_manifest(tree_cache_folder: str, commit_hash: str) -> None:
    """Delete the snapshot manifest of a commit hash, marking the snapshot as complete."""
    path = _snapshot_manifest_path(tree_cache_folder, commit_hash)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"Ignored error while deleting snapshot manifest file {path}: {e}")
