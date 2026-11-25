"""Show what changed between two benchmark runs.

Each side can be a saved snapshot (evals/history/NAME.json) or a full report:

    uv run python -m evals.chat.compare evals/history/baseline-v1.json evals/history/after.json
"""

import argparse
import json
import sys
from pathlib import Path

from .summary import load_rows, ratio, snapshot

METRICS = [
    ("retrieval", "retrieval"), ("in_prompt", "in prompt"), ("citations", "citations"),
    ("correct", "correct"), ("supported", "supported"),
]
CASE_FIELDS = ["status", "retrieval", "in_prompt", "citations_valid", "correct", "supported"]


def load(path: Path) -> dict:
    """Read a snapshot, or build one from a full report."""
    if path.suffix == ".jsonl":
        return snapshot(load_rows(path), path.stem)
    return json.loads(path.read_text(encoding="utf-8"))


def rate(pair: list[int]) -> float | None:
    """found/total as a fraction, or None when nothing was measured."""
    return pair[0] / pair[1] if pair[1] else None


def change(before: list[int], after: list[int]) -> str:
    """Describe the move in percentage points."""
    old, new = rate(before), rate(after)
    if old is None or new is None:
        return "n/a"
    points = round((new - old) * 100)
    return "same" if points == 0 else f"{points:+d} pts"


def metric_rows(before: dict, after: dict) -> list[str]:
    """One line per case type and metric that was measured on either side."""
    lines = []
    for group in after["summary"]:
        if group not in before["summary"]:
            continue
        for key, label in METRICS:
            old, new = before["summary"][group][key], after["summary"][group][key]
            if not old[1] and not new[1]:
                continue
            lines.append(
                f"{group:<13}{label:<11}{ratio(*old):>12}{ratio(*new):>12}  {change(old, new)}"
            )
    return lines


def case_changes(before: dict, after: dict) -> list[str]:
    """List the cases whose scores differ, field by field."""
    lines = []
    for case_id, new in after["cases"].items():
        old = before["cases"].get(case_id)
        if old is None:
            continue
        diffs = [
            f"{field} {old.get(field)} -> {new.get(field)}"
            for field in CASE_FIELDS
            if old.get(field) != new.get(field)
        ]
        if diffs:
            lines.append(f"{case_id} ({new['type']}): " + "; ".join(diffs))
    return lines


def warnings(before: dict, after: dict) -> list[str]:
    """Flag comparisons that are not like for like."""
    notes = []
    old, new = before["run"].get("cases_hash"), after["run"].get("cases_hash")
    if old and new and old != new:
        notes.append("The case labels changed between runs, so scores are not comparable.")
    if set(before["cases"]) != set(after["cases"]):
        notes.append("The two runs cover different case IDs; only shared cases are compared.")
    if after["run"].get("git_dirty"):
        notes.append("The 'after' run came from a dirty working tree; commit before recording.")
    return notes


def label(side: dict) -> str:
    """Name a side with its commit when known."""
    commit = side["run"].get("git_commit")
    return f"{side['name']} ({commit})" if commit else side["name"]


def main() -> int:
    """Print the metric deltas and the cases that flipped."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()
    before, after = load(args.before), load(args.after)
    print(f"before: {label(before)}\nafter:  {label(after)}")
    for note in warnings(before, after):
        print(f"WARNING: {note}")
    print(f"\n{'type':<13}{'metric':<11}{'before':>12}{'after':>12}  change")
    print("\n".join(metric_rows(before, after)))
    changed = case_changes(before, after)
    print("\nCases that changed:" if changed else "\nNo case changed.")
    print("\n".join(f"  {line}" for line in changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
