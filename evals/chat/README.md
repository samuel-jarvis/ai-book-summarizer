# Chat benchmark

Run this command from the backend directory.

```powershell
uv run python -m evals.chat
```

The command uses the 12 cases in `hg_wells_chat_benchmark.jsonl`, runs each
conversation through the app's chat pipeline, and writes one JSONL row per case
under `evals/results/`. It uses the backend's normal model settings and API keys.
The runner calls the pipeline in-process, so it only needs PostgreSQL with
pgvector and access to the model providers.

## One-time setup

1. Place the labeled PDF editions in `evals/fixtures/`, or pass `--pdf-dir`.
   The runner verifies all labeled quotes on their 1-based PDF pages before
   indexing. Fixture bytes and index settings determine whether an existing index
   can be reused.
   - `the-time-machine.pdf` from [Fourmilab](https://www.fourmilab.ch/etexts/www/wells/timemach/timemach.pdf)
   - `the-invisible-man.pdf` from [CBSE Academic](https://cbseacademic.nic.in/web_material/doc/novels/2_The%20Invisible%20Man%2C%20by%20H.%20G%20-%20Class%20-%20XII.pdf)
2. Set `CHAT_EVAL_DATABASE_URL` to a dedicated local PostgreSQL database. Use the
   normal Alembic migrations to prepare it. This example uses a database named
   `lumen_chat_eval` and the Compose dev PostgreSQL port.

   ```powershell
   $env:CHAT_EVAL_DATABASE_URL = "postgresql+asyncpg://USER:PASSWORD@127.0.0.1:5433/lumen_chat_eval"
   $env:DATABASE_URL = $env:CHAT_EVAL_DATABASE_URL
   uv run alembic upgrade head
   ```

   Run the migration in a separate shell if you already use `DATABASE_URL` for
   the application. For subsequent eval runs, only `CHAT_EVAL_DATABASE_URL` is needed
   alongside the usual backend configuration.

   With the Compose dev stack running (`compose.dev.yaml` publishes Postgres on
   port 5433), create the database inside the existing container, then migrate it
   as above. It is a separate database on the same server, so the app's data is
   untouched. `CHAT_EVAL_DATABASE_URL` can live in the backend's `.env`; a shell
   variable of the same name takes precedence.

   ```powershell
   docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d postgres -c "CREATE DATABASE lumen_chat_eval"'
   ```

## Reading the report

- **Retrieval** and **prompt evidence** count labeled quotes found in retrieved
  and selected chunks respectively. A match needs the same filename, page, and
  normalized quote. These counts are null for abstention cases.
- **Citation labels** checks labels in citations such as `[S1]` and `(S1, p. 32)`
  and verifies they refer to passages selected for the current answer.
- **Answer correctness** and **citation support** come from an LLM judge
  (`judge.py`), one temperature-0 call per completed case. It sees the
  conversation, the labeled facts, the answer and only the passages that were in
  the prompt. For unanswerable cases, "correct" means the answer declined
  without inventing a value, and citation support is left null. The judge's
  one-line reason is stored under `scores.judge`. If a judge call fails, that
  case stays ungraded and records the error type; it does not fail the run.
  The judge defaults to the same model that writes the answers, which tends to
  favor its own style, so pass `--judge-model` to use another one, and spot-check
  a few verdicts against the reasons. `--no-judge` skips grading entirely.
- A **summary table** by case type prints at the end of every run. To print it
  for an older report, with no services or API calls:

  ```powershell
  uv run python -m evals.chat.summary                          # newest report
  uv run python -m evals.chat.summary evals/results/chat-<stamp>.jsonl
  ```
- For pipeline failures, the report records the failed turn and application
  error code. The command exits nonzero if any pipeline turn fails. Evidence
  misses and citation-label failures are reported as scores.

Use `--cases` for a different JSONL file or `--output` to choose a new report path.

## Tracking changes over time

Each report row carries a `run` block: the git commit and branch, whether the working
tree was dirty, a hash of the case labels and the judge model. Reports stay local, so
save a compact snapshot of any run worth keeping and commit that instead:

```powershell
uv run python -m evals.chat.summary --save my-change --note "what changed"
uv run python -m evals.chat.compare evals/history/baseline-v1.json evals/history/my-change.json
```

`compare` accepts snapshots or full reports, prints the change per case type and metric
in percentage points, lists the cases whose scores flipped, and warns when the labels
changed or the newer run came from a dirty tree. [../EXPERIMENTS.md](../EXPERIMENTS.md)
holds the method and the log of every experiment.
