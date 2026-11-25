"""Facts that tie a report to the code and labels that produced it."""

import hashlib
import json
import subprocess

from app.core.config import settings

from .cases import Case


def git(*args: str) -> str | None:
    """Run a read-only git command; None if git or the repository is unavailable."""
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def cases_hash(cases: list[Case]) -> str:
    """Short fingerprint of the labels, so a compare can tell the exam did not change."""
    payload = json.dumps([case.model_dump() for case in cases], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def run_info(cases: list[Case], judge_model: str | None) -> dict:
    """Describe this run: commit, whether the tree was dirty, label hash, judge model."""
    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_dirty": None if status is None else bool(status),
        "cases_hash": cases_hash(cases),
        "judge_model": judge_model,
        "min_chunks_per_document": settings.RAG_MIN_CHUNKS_PER_DOCUMENT,
    }
