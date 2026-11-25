"""Roll a chat benchmark report up into one table.

Reads only the JSONL report, so it works on old reports and needs no services:

    uv run python -m evals.chat.summary [report.jsonl] [--save NAME] [--note TEXT]

--save also writes a compact snapshot to evals/history/NAME.json, which is meant
to be committed (reports stay local because they contain book text).
"""

import argparse
import json
import sys
from pathlib import Path

TYPES = ["direct", "follow_up", "comparison", "unanswerable"]
HISTORY_DIR = Path("evals/history")


def load_rows(path: Path) -> list[dict]:
    """Read one report row per case."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ratio(found: int, total: int) -> str:
    """Format a count as 'found/total pct', or N/A when nothing was measured."""
    return f"{found}/{total} {found / total:.0%}" if total else "N/A"


def tally(rows: list[dict]) -> dict:
    """Count one group of case rows. Scores that were not measured are skipped."""
    scored = [row["scores"] for row in rows if row["scores"]]
    out = {"cases": len(rows), "failed": len(rows) - len(scored)}
    for name, key in [("retrieval", "retrieval"), ("in_prompt", "prompt_evidence")]:
        parts = [score[key] for score in scored if score[key]]
        out[name] = (sum(p["found"] for p in parts), sum(p["total"] for p in parts))
    for name, key in [
        ("citations", "citation_labels_valid"),
        ("correct", "answer_correct"),
        ("supported", "citations_support_claims"),
    ]:
        values = [score[key] for score in scored if score.get(key) is not None]
        out[name] = (sum(values), len(values))
    return out


def summarize(rows: list[dict]) -> dict[str, dict]:
    """Tally every case type that appears, plus an 'all' row."""
    groups = {kind: [r for r in rows if r["case"]["type"] == kind] for kind in TYPES}
    summary = {kind: tally(group) for kind, group in groups.items() if group}
    summary["all"] = tally(rows)
    return summary


def format_summary(summary: dict[str, dict]) -> str:
    """Render the tally as an aligned text table."""
    columns = [
        ("retrieval", "retrieval"), ("in_prompt", "in prompt"), ("citations", "citations"),
        ("correct", "correct"), ("supported", "supported"),
    ]
    header = f"{'type':<13}{'cases':>6}{'failed':>7}" + "".join(f"{label:>14}" for _, label in columns)
    lines = [header]
    for kind, row in summary.items():
        cells = "".join(f"{ratio(*row[key]):>14}" for key, _ in columns)
        lines.append(f"{kind:<13}{row['cases']:>6}{row['failed']:>7}{cells}")
    lines.append(
        "retrieval/in prompt: labeled quotes found.  citations: labels point at real passages.\n"
        "correct: judge verdict (for unanswerable cases: declined without inventing).  "
        "N/A: not measured or not judged."
    )
    return "\n".join(lines)


def fraction(part: dict | None) -> str | None:
    """Write a {found, total} score as 'found/total'."""
    return f"{part['found']}/{part['total']}" if part else None


def snapshot(rows: list[dict], name: str, note: str | None = None) -> dict:
    """Condense a report into the small, commit-friendly record that compare reads.

    It keeps scores and run facts but none of the book text the full report holds.
    """
    run = dict(rows[0].get("run") or {}) if rows else {}
    if note:
        run["note"] = note
    run["chunks_per_question"] = max((len(row["last_turn_sources"]) for row in rows), default=0)
    cases = {}
    for row in rows:
        scores = row["scores"]
        entry = {"type": row["case"]["type"], "status": "completed" if scores else "failed"}
        if scores:
            entry |= {
                "retrieval": fraction(scores["retrieval"]),
                "in_prompt": fraction(scores["prompt_evidence"]),
                "citations_valid": scores["citation_labels_valid"],
                "correct": scores.get("answer_correct"),
                "supported": scores.get("citations_support_claims"),
            }
        cases[row["case"]["id"]] = entry
    return {"name": name, "run": run, "summary": summarize(rows), "cases": cases}


def main() -> int:
    """Summarize a report (default: the newest); optionally save a snapshot of it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, help="Report JSONL (default: newest)")
    parser.add_argument("--save", metavar="NAME", help="Write evals/history/NAME.json")
    parser.add_argument("--note", help="Free text stored with the snapshot")
    args = parser.parse_args()
    path = args.report
    if path is None:
        reports = sorted(Path("evals/results").glob("chat-*.jsonl"))
        if not reports:
            print("No reports found in evals/results.", file=sys.stderr)
            return 2
        path = reports[-1]
    rows = load_rows(path)
    print(f"Report: {path}")
    print(format_summary(summarize(rows)))
    if args.save:
        target = HISTORY_DIR / f"{args.save}.json"
        if target.exists():
            print(f"Snapshot already exists: {target}", file=sys.stderr)
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(snapshot(rows, args.save, args.note), indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved snapshot: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
