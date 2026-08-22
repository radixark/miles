import hashlib
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from miles.utils.audit_utils.event_logger.models import EnvReportGitRepoInfo

logger = logging.getLogger(__name__)
_HEAD_MOVE_ATTEMPTS = 2
_UNTRACKED_MAX_FILES = 1000
_UNTRACKED_MAX_FILE_BYTES = 10 * 1024 * 1024


def collect_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    if not location or not os.path.isdir(location):
        return None
    for _ in range(_HEAD_MOVE_ATTEMPTS):
        try:
            return _collect_stable_git_info(package_name=package_name, location=location)
        except _HeadMoved:
            continue
    logger.warning("Gave up collecting the git state of %s at %s, its HEAD keeps moving", package_name, location)
    return None


class _HeadMoved(Exception):
    pass


def _collect_stable_git_info(*, package_name: str, location: str) -> EnvReportGitRepoInfo | None:
    try:
        commit_result = _run_git(args=["rev-parse", "HEAD"], location=location)
        if commit_result.returncode != 0:
            return None

        diff_stat_result = _run_git(args=["diff", "--stat", "HEAD"], location=location)
        diff_stat = _decode(diff_stat_result.stdout).strip()

        patch_result = _run_git(args=["diff", "HEAD"], location=location)
        diff_patch = patch_result.stdout if patch_result.returncode == 0 else None

        untracked = _collect_untracked(location=location)

        if _run_git(args=["rev-parse", "HEAD"], location=location).stdout != commit_result.stdout:
            raise _HeadMoved

        return EnvReportGitRepoInfo(
            package_name=package_name,
            location=location,
            commit=_decode(commit_result.stdout).strip(),
            dirty=bool(diff_stat) or bool(untracked.paths),
            diff_stat=diff_stat,
            uncommitted_hash=_hash_uncommitted(diff_patch=diff_patch, untracked=untracked),
            untracked_paths=untracked.paths,
            untracked_paths_truncated=untracked.paths_truncated,
            untracked_unhashed_paths=untracked.unhashed_paths,
        )
    except _HeadMoved:
        raise
    except Exception:
        logger.warning("Failed to collect git info for %s at %s", package_name, location, exc_info=True)
        return None


@dataclass(frozen=True)
class _UntrackedFiles:
    paths: list[str]
    hash_entries: list[bytes]
    paths_truncated: bool
    total_count: int
    unhashed_paths: list[str]


def _hash_uncommitted(*, diff_patch: bytes | None, untracked: _UntrackedFiles) -> str | None:
    if diff_patch is None:
        return None

    digest = hashlib.sha256()
    digest.update(b"diff\0%d\0" % len(diff_patch))
    digest.update(diff_patch)
    for entry in untracked.hash_entries:
        digest.update(b"untracked\0" + entry + b"\0")
    if untracked.paths_truncated:
        digest.update(b"untracked-truncated\0%d\0" % untracked.total_count)

    return digest.hexdigest()


def _collect_untracked(*, location: str) -> _UntrackedFiles:
    result = _run_git(args=["ls-files", "--others", "--exclude-standard", "-z"], location=location)
    if result.returncode != 0:
        return _UntrackedFiles(paths=[], hash_entries=[], paths_truncated=False, total_count=0, unhashed_paths=[])

    all_paths = sorted(path for path in result.stdout.split(b"\0") if path)
    selected = all_paths[:_UNTRACKED_MAX_FILES]

    root = Path(location)
    paths: list[str] = []
    hash_entries: list[bytes] = []
    unhashed_paths: list[str] = []
    for raw_path in selected:
        entry, hashed = _untracked_hash_entry(root=root, raw_path=raw_path)
        paths.append(_decode(raw_path))
        hash_entries.append(entry)
        if not hashed:
            unhashed_paths.append(_decode(raw_path))

    return _UntrackedFiles(
        paths=paths,
        hash_entries=hash_entries,
        paths_truncated=len(all_paths) > len(selected),
        total_count=len(all_paths),
        unhashed_paths=unhashed_paths,
    )


def _untracked_hash_entry(*, root: Path, raw_path: bytes) -> tuple[bytes, bool]:
    full_path = root / os.fsdecode(raw_path)
    try:
        if (size := full_path.stat().st_size) > _UNTRACKED_MAX_FILE_BYTES:
            return b"%s\0size:%d" % (raw_path, size), False
        content_digest = hashlib.sha256(full_path.read_bytes()).hexdigest()
        return b"%s\0sha256:%s" % (raw_path, content_digest.encode()), True
    except OSError:
        logger.warning("Failed to hash the untracked file %s", full_path, exc_info=True)
        return b"%s\0unreadable" % raw_path, False


def _run_git(*, args: list[str], location: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        timeout=10,
        cwd=location,
    )


def _decode(raw: bytes) -> str:
    return raw.decode(errors="replace")
