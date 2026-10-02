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
"""Tests for the snapshot download failure manifest and safe-resume behavior.

These tests use a fully local, controllable "fake repo": the network-facing parts of `snapshot_download`
(`repo_info`, `list_repo_tree` and `hf_hub_download`) are patched. The fake `hf_hub_download` mimics the real cache
layout (`snapshots/<commit>/<filename>`, or `local_dir/<filename>`), skips files already on disk (resume) and can be
configured to make a given file fail.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huggingface_hub import get_snapshot_download_manifest, snapshot_download
from huggingface_hub._snapshot_manifest import read_snapshot_manifest, write_snapshot_manifest
from huggingface_hub.errors import (
    EntryNotFoundError,
    FailedFileDownload,
    IncompleteSnapshotError,
    LocalEntryNotFoundError,
    SnapshotDownloadError,
)
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.hf_api import RepoFile

from .testing_utils import OfflineSimulationMode, offline


FAKE_REPO_ID = "fake-user/fake-repo"
FAKE_COMMIT_HASH = "fedcba9876543210fedcba9876543210fedcba98"
FAKE_FILES = {"a.txt": b"content-a", "subdir/b.txt": b"content-b", "c.bin": b"\x00\x01\x02\x03"}
FAILING_FILE = "c.bin"


def _fake_repo_files():
    return [RepoFile(path=path, size=len(content), oid=f"blob-{path}") for path, content in sorted(FAKE_FILES.items())]


class _FakeHfHubDownloader:
    """Stand-in for `hf_hub_download` writing files into the real cache layout.

    Files already present on disk are skipped (resume behavior). Files in `fail_paths` raise `EntryNotFoundError`
    until removed from that set. `calls` records every filename attempted.
    """

    def __init__(self):
        self.calls: list[str] = []
        self.fail_paths: set[str] = set()

    def __call__(self, repo_id, filename, *, repo_type=None, revision=None, cache_dir=None, local_dir=None, **kwargs):
        base_dir = (
            local_dir
            if local_dir is not None
            else os.path.join(
                cache_dir,
                repo_folder_name(repo_id=repo_id, repo_type=repo_type or "model"),
                "snapshots",
                revision,
            )
        )
        file_path = os.path.join(base_dir, *filename.split("/"))
        if os.path.isfile(file_path):
            return file_path  # already on disk => skipped, like the real `hf_hub_download`
        self.calls.append(filename)
        if filename in self.fail_paths:
            raise EntryNotFoundError(f"Simulated failure while downloading {filename!r}.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(FAKE_FILES[filename])
        return file_path


@pytest.fixture
def fake_snapshot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch the network-facing bits of `snapshot_download` with the local fake repo.

    The fake API respects `HF_HUB_OFFLINE`: when offline mode is enabled, `repo_info` raises (mimicking a failed
    connection) so `snapshot_download` takes its offline/cache-only code path.
    """
    downloader = _FakeHfHubDownloader()

    def _fake_repo_info(*args, **kwargs):
        from huggingface_hub import constants

        if constants.HF_HUB_OFFLINE:
            from huggingface_hub.utils import OfflineModeIsEnabled

            raise OfflineModeIsEnabled("Offline mode is enabled (fake repo).")
        return MagicMock(sha=FAKE_COMMIT_HASH)

    with (
        patch("huggingface_hub._snapshot_download.HfApi.repo_info", side_effect=_fake_repo_info),
        patch(
            "huggingface_hub._snapshot_download.HfApi.list_repo_tree",
            return_value=_fake_repo_files(),
        ),
        patch("huggingface_hub._snapshot_download.hf_hub_download", side_effect=downloader),
    ):
        yield downloader, tmp_path


class TestSnapshotDownloadFailureManifest:
    def test_failed_run_records_manifest_and_keeps_partial_download(self, fake_snapshot_env):
        """One file fails: the other two are kept on disk, a compatible error is raised and a manifest is written."""
        downloader, cache_dir = fake_snapshot_env
        downloader.fail_paths = {FAILING_FILE}

        with pytest.raises(SnapshotDownloadError) as exc_info:
            snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)

        error = exc_info.value
        assert error.succeeded_files == ["a.txt", "subdir/b.txt"]
        assert [failure.path for failure in error.failed_files] == [FAILING_FILE]
        assert error.failed_files[0].error_type == "EntryNotFoundError"
        assert "Simulated failure" in error.failed_files[0].error_message
        # Backward compatibility: the raised error is ALSO an instance of the per-file error class.
        assert isinstance(error, EntryNotFoundError)
        assert isinstance(error.__cause__, EntryNotFoundError)

        storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=FAKE_REPO_ID, repo_type="model"))
        snapshot_folder = os.path.join(storage_folder, "snapshots", FAKE_COMMIT_HASH)
        assert error.snapshot_path == snapshot_folder
        # Successful files are kept; the failed one is absent.
        assert os.path.isfile(os.path.join(snapshot_folder, "a.txt"))
        assert os.path.isfile(os.path.join(snapshot_folder, "subdir", "b.txt"))
        assert not os.path.exists(os.path.join(snapshot_folder, FAILING_FILE))

        # The partial snapshot is recorded as incomplete (not marked complete).
        manifest = read_snapshot_manifest(storage_folder, FAKE_COMMIT_HASH)
        assert manifest is not None
        assert manifest.completed is False
        assert manifest.succeeded_files == ["a.txt", "subdir/b.txt"]
        assert manifest.failed_paths == [FAILING_FILE]
        assert manifest.failed_files[0].error_type == "EntryNotFoundError"

        # Public helper exposes the same information.
        assert get_snapshot_download_manifest(FAKE_REPO_ID, revision=FAKE_COMMIT_HASH, cache_dir=cache_dir) == manifest

    def test_partial_folder_not_returned_while_offline(self, fake_snapshot_env):
        """After a failed run, an offline re-pull raises instead of silently returning the partial folder."""
        downloader, cache_dir = fake_snapshot_env
        downloader.fail_paths = {FAILING_FILE}
        with pytest.raises(SnapshotDownloadError):
            snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)

        # The fake repo reads the offline flag straight from `constants.HF_HUB_OFFLINE`.
        with offline(mode=OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
            with pytest.raises(IncompleteSnapshotError) as exc_info:
                snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir)
        error = exc_info.value
        # Old exception pattern preserved.
        assert isinstance(error, LocalEntryNotFoundError)
        assert error.succeeded_files == ["a.txt", "subdir/b.txt"]
        assert error.failed_files == [FAILING_FILE]

    def test_resume_only_downloads_missing_file_and_marks_complete(self, fake_snapshot_env):
        """Second run resumes: cached files skipped, missing file fetched, manifest completed."""
        downloader, cache_dir = fake_snapshot_env
        downloader.fail_paths = {FAILING_FILE}
        with pytest.raises(SnapshotDownloadError):
            snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)
        assert sorted(downloader.calls) == ["a.txt", "c.bin", "subdir/b.txt"]

        # Fix the failure and re-run: safe resume.
        downloader.fail_paths = set()
        downloader.calls.clear()
        snapshot_folder = snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)

        # Only the missing file was attempted again.
        assert downloader.calls == [FAILING_FILE]
        # All 3 files present with expected content.
        for path, content in FAKE_FILES.items():
            with open(os.path.join(snapshot_folder, *path.split("/")), "rb") as f:
                assert f.read() == content

        # Manifest now reports a completed download.
        storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=FAKE_REPO_ID, repo_type="model"))
        manifest = read_snapshot_manifest(storage_folder, FAKE_COMMIT_HASH)
        assert manifest.completed is True
        assert manifest.succeeded_files == ["a.txt", "c.bin", "subdir/b.txt"]
        assert manifest.failed_files == []

        # A complete snapshot is returned offline.
        with offline(mode=OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
            assert snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir) == snapshot_folder

    def test_all_files_fail_manifest_still_written(self, fake_snapshot_env):
        """If every file fails, the run is still recorded as incomplete (empty success list)."""
        downloader, cache_dir = fake_snapshot_env
        downloader.fail_paths = set(FAKE_FILES)
        with pytest.raises(SnapshotDownloadError) as exc_info:
            snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)
        assert exc_info.value.succeeded_files == []
        assert sorted(failure.path for failure in exc_info.value.failed_files) == sorted(FAKE_FILES)

        storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=FAKE_REPO_ID, repo_type="model"))
        manifest = read_snapshot_manifest(storage_folder, FAKE_COMMIT_HASH)
        assert manifest.completed is False
        assert manifest.succeeded_files == []
        assert manifest.failed_paths == sorted(FAKE_FILES)

    def test_successful_run_records_completed_manifest(self, fake_snapshot_env):
        """A plain successful run writes a completed manifest."""
        downloader, cache_dir = fake_snapshot_env
        snapshot_download(FAKE_REPO_ID, cache_dir=cache_dir, max_workers=1)
        storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=FAKE_REPO_ID, repo_type="model"))
        manifest = read_snapshot_manifest(storage_folder, FAKE_COMMIT_HASH)
        assert manifest.completed is True
        assert manifest.succeeded_files == sorted(FAKE_FILES)
        assert manifest.failed_files == []

    def test_failure_and_resume_with_local_dir(self, fake_snapshot_env, tmp_path: Path):
        """The same failure/resume behavior applies to local_dir downloads."""
        downloader, _ = fake_snapshot_env
        local_dir = tmp_path / "local"
        downloader.fail_paths = {FAILING_FILE}
        with pytest.raises(SnapshotDownloadError) as exc_info:
            snapshot_download(FAKE_REPO_ID, local_dir=local_dir, max_workers=1)
        assert Path(exc_info.value.snapshot_path).resolve() == local_dir.resolve()
        assert exc_info.value.succeeded_files == ["a.txt", "subdir/b.txt"]

        # Manifest lives under local_dir/.cache/huggingface.
        manifest_cache = os.path.join(local_dir, ".cache", "huggingface")
        manifest = read_snapshot_manifest(manifest_cache, FAKE_COMMIT_HASH)
        assert manifest is not None and manifest.completed is False
        assert manifest.failed_paths == [FAILING_FILE]

        # Partial local_dir is not returned while offline.
        with offline(mode=OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
            with pytest.raises(IncompleteSnapshotError):
                snapshot_download(FAKE_REPO_ID, local_dir=local_dir)

        # Resume.
        downloader.fail_paths = set()
        returned = snapshot_download(FAKE_REPO_ID, local_dir=local_dir, max_workers=1)
        assert Path(returned).resolve() == local_dir.resolve()
        for path, content in FAKE_FILES.items():
            assert local_dir.joinpath(*path.split("/")).read_bytes() == content
        manifest = read_snapshot_manifest(manifest_cache, FAKE_COMMIT_HASH)
        assert manifest.completed is True
        # The completed folder is returned offline.
        with offline(mode=OfflineSimulationMode.HF_HUB_OFFLINE_SET_TO_1):
            assert Path(snapshot_download(FAKE_REPO_ID, local_dir=local_dir)).resolve() == local_dir.resolve()

    def test_manifest_round_trip_from_disk_drops_exception(self, tmp_path: Path):
        """A manifest read back from disk carries the failure type/message but no live exception object."""
        write_snapshot_manifest(
            str(tmp_path / "store"),
            FAKE_COMMIT_HASH,
            succeeded_files=["a.txt"],
            failed_files=[FailedFileDownload.from_error("c.bin", EntryNotFoundError("boom"))],
        )
        manifest = read_snapshot_manifest(str(tmp_path / "store"), FAKE_COMMIT_HASH)
        assert manifest.completed is False
        failure = manifest.failed_files[0]
        assert failure.path == "c.bin"
        assert failure.error is None
        assert failure.error_type == "EntryNotFoundError"
        assert failure.error_message == "boom"
