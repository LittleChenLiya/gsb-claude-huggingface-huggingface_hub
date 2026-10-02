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
"""On-disk record of the outcome of a [`snapshot_download`] run.

After (partially or fully) downloading a snapshot, its outcome is recorded as a human-readable JSON file under
`<tree_cache_folder>/snapshots/<commit_hash>.json`. The folder depends on the download target: the per-repo
`storage_folder` for `cache_dir` downloads, or `local_dir/.cache/huggingface/` for `local_dir` downloads (same as for
the tree listing cache, see [`huggingface_hub._tree_cache`]).

```json
{
  "format_version": 1,
  "completed": false,
  "succeeded_files": ["config.json", "model.safetensors"],
  "failed_files": [
    {"path": "tokenizer.json", "error_type": "EntryNotFoundError", "error_message": "Entry Not Found ..."}
  ]
}
```

The manifest serves two purposes:

1. It prevents a partial snapshot directory from being reported as complete (e.g. when offline on the next run).
2. It lists which files were successfully downloaded and which ones failed, making a re-run a safe resume: cached
   files are skipped and only the failed files are downloaded again.
"""

import json
import os
import tempfile
from dataclasses import dataclass, field

from .errors import FailedFileDownload
from .utils import logging
from .utils._paths import as_extended_path


logger = logging.get_logger(__name__)

SNAPSHOT_MANIFEST_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SnapshotManifest:
    """Outcome of a snapshot download for a given commit.

    Attributes:
        completed (`bool`):
            Whether every requested file was downloaded successfully on the last run.
        succeeded_files (`list[str]`):
            Repo-relative paths ("/"-separated) downloaded successfully, sorted.
        failed_files (`list[FailedFileDownload]`):
            One entry per file that could not be downloaded, sorted by path. For manifests read back from disk, the
            entries carry only `path`, `error_type` and `error_message` (the original exception is not persisted).
    """

    completed: bool
    succeeded_files: list[str] = field(default_factory=list)
    failed_files: list[FailedFileDownload] = field(default_factory=list)

    @property
    def failed_paths(self) -> list[str]:
        return [failure.path for failure in self.failed_files]

    def to_json(self) -> dict:
        return {
            "format_version": SNAPSHOT_MANIFEST_FORMAT_VERSION,
            "completed": self.completed,
            "succeeded_files": list(self.succeeded_files),
            "failed_files": [
                {
                    "path": failure.path,
                    "error_type": failure.error_type,
                    "error_message": failure.error_message,
                }
                for failure in self.failed_files
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> "SnapshotManifest":
        return cls(
            completed=bool(data["completed"]),
            succeeded_files=list(data.get("succeeded_files", [])),
            failed_files=[
                FailedFileDownload(
                    path=info["path"],
                    error=None,
                    error_type=info.get("error_type"),
                    error_message=info.get("error_message"),
                )
                for info in data.get("failed_files", [])
            ],
        )


def _snapshot_manifest_path(tree_cache_folder: str, commit_hash: str) -> str:
    # Both readers and writers go through here, so they agree on the path, including on Windows where a deep
    # `local_dir`/`cache_dir` requires the extended-length form.
    return as_extended_path(os.path.join(tree_cache_folder, "snapshots", f"{commit_hash}.json"))


def write_snapshot_manifest(
    tree_cache_folder: str,
    commit_hash: str,
    *,
    succeeded_files: list[str],
    failed_files: list[FailedFileDownload],
) -> SnapshotManifest | None:
    """Atomically record the outcome of a snapshot download. Returns the written manifest, or `None` on failure."""
    manifest = SnapshotManifest(
        completed=not failed_files,
        succeeded_files=sorted(succeeded_files),
        failed_files=sorted(failed_files, key=lambda failure: failure.path),
    )
    path = _snapshot_manifest_path(tree_cache_folder, commit_hash)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(manifest.to_json(), f, indent=1)
        os.replace(tmp_path, path)
    except OSError as e:
        logger.warning(f"Ignored error while writing snapshot manifest {path}: {e}")
        return None
    return manifest


def read_snapshot_manifest(tree_cache_folder: str, commit_hash: str) -> SnapshotManifest | None:
    """Return the recorded outcome of a snapshot download, or `None` if not recorded, invalid, or unreadable."""
    path = _snapshot_manifest_path(tree_cache_folder, commit_hash)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("format_version") != SNAPSHOT_MANIFEST_FORMAT_VERSION:
            # Unknown format (e.g. written by a newer version) => ignore it.
            return None
        return SnapshotManifest.from_json(data)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Ignoring corrupted snapshot manifest {path}: {e}")
        return None
