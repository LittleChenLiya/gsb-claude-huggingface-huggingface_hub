"""Acceptance demo for the snapshot download failure list / safe-resume feature.

Runs fully locally without any network access or Docker: a fake 3-file repository is emulated by patching the
network-facing parts of `snapshot_download` (`repo_info`, `list_repo_tree`, `hf_hub_download`). The fake
`hf_hub_download` mimics the real cache layout (`snapshots/<commit>/<file>`), skips files already on disk (resume)
and can be configured to make a given file fail.

Scenario:
  1. First `snapshot_download`: one of the three files ("c.bin") fails.
  2. The raised `SnapshotDownloadError` lists succeeded/failed files, keeps the partial files on disk and is also an
     instance of the original per-file error (`EntryNotFoundError`) for backward compatibility.
  3. An offline re-pull refuses to return the partial folder (`IncompleteSnapshotError`).
  4. After the failure is fixed, re-running `snapshot_download` resumes: only the missing file is fetched; the
     on-disk manifest is rewritten as completed.

Run with the in-repo sources:
    PYTHONPATH=src HF_HUB_DISABLE_PROGRESS_BARS=1 python examples/snapshot_failure_manifest_demo.py
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from huggingface_hub import constants, get_snapshot_download_manifest, snapshot_download
from huggingface_hub._snapshot_manifest import read_snapshot_manifest
from huggingface_hub.errors import (
    EntryNotFoundError,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    SnapshotDownloadError,
)
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import OfflineModeIsEnabled


REPO_ID = "fake-user/fake-repo"
COMMIT = "fedcba9876543210fedcba9876543210fedcba98"
FILES = {"a.txt": b"content-a", "subdir/b.txt": b"content-b", "c.bin": b"\x00\x01\x02\x03"}

fail_paths: set[str] = {"c.bin"}
calls: list[str] = []


def fake_repo_info(*args, **kwargs):
    if constants.HF_HUB_OFFLINE:
        raise OfflineModeIsEnabled("offline (demo)")
    return MagicMock(sha=COMMIT)


def fake_hf_hub_download(repo_id, filename, *, repo_type=None, revision=None, cache_dir=None, local_dir=None, **kw):
    base = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type="model"), "snapshots", revision)
    path = os.path.join(base, *filename.split("/"))
    if os.path.isfile(path):
        return path  # already on disk => skipped, like the real hf_hub_download
    calls.append(filename)
    if filename in fail_paths:
        raise EntryNotFoundError(f"simulated failure for {filename}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Path(path).write_bytes(FILES[filename])
    return path


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="snapshot_demo_")
    storage = os.path.join(tmp, repo_folder_name(repo_id=REPO_ID, repo_type="model"))

    with (
        patch("huggingface_hub._snapshot_download.HfApi.repo_info", side_effect=fake_repo_info),
        patch(
            "huggingface_hub._snapshot_download.HfApi.list_repo_tree",
            return_value=[RepoFile(path=p, size=len(c), oid=f"blob-{p}") for p, c in sorted(FILES.items())],
        ),
        patch("huggingface_hub._snapshot_download.hf_hub_download", side_effect=fake_hf_hub_download),
    ):
        print("=" * 70)
        print("STEP 1: first download, c.bin fails")
        print("=" * 70)
        try:
            snapshot_download(REPO_ID, cache_dir=tmp, max_workers=1)
        except SnapshotDownloadError as e:
            print(f"Caught: {type(e).__name__}")
            print(f"  isinstance(EntryNotFoundError) = {isinstance(e, EntryNotFoundError)}  <- old pattern compat")
            print(f"  __cause__                      = {type(e.__cause__).__name__}")
            print(f"  succeeded_files                = {e.succeeded_files}")
            print(f"  failed_files                   = {[f.path for f in e.failed_files]}")
            print(f"  failed[0].error_type           = {e.failed_files[0].error_type}")
            snapshot_folder = e.snapshot_path
        assert snapshot_folder == os.path.join(storage, "snapshots", COMMIT)

        print("\nCache layout after failed run:")
        for _root, _dirs, files in os.walk(storage):
            for f in files:
                print(f"  {os.path.relpath(os.path.join(_root, f), storage)}")

        print("\nManifest (<storage>/snapshots/<commit>.json):")
        print(Path(os.path.join(storage, "snapshots", f"{COMMIT}.json")).read_text())

        print("=" * 70)
        print("STEP 2: offline pull must NOT return the partial folder")
        print("=" * 70)
        constants.HF_HUB_OFFLINE = True
        try:
            snapshot_download(REPO_ID, cache_dir=tmp)
        except IncompleteSnapshotError as e:
            print(f"Caught: {type(e).__name__}")
            print(f"  isinstance(LocalEntryNotFoundError) = {isinstance(e, LocalEntryNotFoundError)}")
            print(f"  snapshot_path ends with snapshots/{COMMIT}")
            print(f"  succeeded_files = {e.succeeded_files}")
            print(f"  failed_files    = {e.failed_files}")
        constants.HF_HUB_OFFLINE = False

        print("\n" + "=" * 70)
        print("STEP 3: resume - failure fixed, re-run snapshot_download")
        print("=" * 70)
        fail_paths.clear()
        calls.clear()
        folder = snapshot_download(REPO_ID, cache_dir=tmp, max_workers=1)
        print(f"Files actually fetched on resume: {calls}  (only the missing one)")
        for p, c in sorted(FILES.items()):
            assert Path(folder, *p.split("/")).read_bytes() == c, p
        print("All 3 files present with expected content.")

        manifest = read_snapshot_manifest(storage, COMMIT)
        print(
            f"\nManifest completed={manifest.completed}, succeeded={manifest.succeeded_files}, "
            f"failed={manifest.failed_paths}"
        )
        assert get_snapshot_download_manifest(REPO_ID, revision=COMMIT, cache_dir=tmp) == manifest
        print("get_snapshot_download_manifest() returns the same manifest.")

        print("\n" + "=" * 70)
        print("STEP 4: offline pull now returns the complete snapshot")
        print("=" * 70)
        constants.HF_HUB_OFFLINE = True
        returned = snapshot_download(REPO_ID, cache_dir=tmp)
        constants.HF_HUB_OFFLINE = False
        assert returned == folder
        print(f"Offline snapshot_download returned: ...{os.sep}snapshots{os.sep}{COMMIT}")
        print("\nALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
