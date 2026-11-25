"""Run the real project chat pipeline against a labeled PDF benchmark."""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from .cases import fixture_paths, read_cases

DEFAULT_CASES = Path(__file__).with_name("hg_wells_chat_benchmark.jsonl")
DEFAULT_FIXTURES = DEFAULT_CASES.parent.parent / "fixtures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_FIXTURES,
        help="Folder with the labeled PDFs (default: evals/fixtures)",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Override the bundled JSONL")
    parser.add_argument("--output", type=Path, help="Write to a new JSONL report at this path")
    parser.add_argument(
        "--judge-model",
        help="Model that grades answers (default: the backend's default chat model)",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip answer grading and leave those scores null"
    )
    args = parser.parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or Path("evals/results") / f"chat-{stamp}.jsonl"

    try:
        cases = read_cases(args.cases)
        paths = fixture_paths(cases, args.pdf_dir)
        # The shell variable wins; otherwise read it from the backend's .env.
        database_url = os.getenv("CHAT_EVAL_DATABASE_URL") or dotenv_values(".env").get(
            "CHAT_EVAL_DATABASE_URL"
        )
        if not database_url:
            raise ValueError(
                "Set CHAT_EVAL_DATABASE_URL (shell or .env) to your migrated local eval database."
            )
        # Set the URL before importing the app. Its normal DATABASE_URL may point
        # at a different environment.
        os.environ["DATABASE_URL"] = database_url
        from .judge import Judge
        from .runner import run_benchmark

        judge = None if args.no_judge else Judge(args.judge_model)
        return asyncio.run(run_benchmark(cases, paths, output, judge))
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, error["loc"])) for error in exc.errors())
        print(f"Chat eval setup failed. Check app configuration fields: {fields}", file=sys.stderr)
    except SQLAlchemyError as exc:
        print(
            f"Chat eval database error ({type(exc).__name__}). Check the database URL and migrations.",
            file=sys.stderr,
        )
    except (ValueError, OSError) as exc:
        print(f"Chat eval setup failed: {exc}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
