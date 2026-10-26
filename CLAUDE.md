# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI service ("Lumen AI" backend) that turns uploaded PDF books into LLM output. Two
pipelines live side by side:

- **Legacy summaries** (`/summary`, used by the React client): `POST /summary` uploads a PDF,
  extracts its text in the request, creates a `PENDING` `Summary` row, and dispatches the
  taskiq task `process_pdf_task`, which map-reduces the text through DeepSeek and writes the
  result back (`PROCESSING` → `COMPLETED`/`FAILED`, text in `Summary.content`).
- **Projects / RAG / conversations** (no client yet): `POST /project` stores a `Document`
  linked to a `Project` and enqueues a document summary and an embedding/indexing job. A
  service-only conversation pipeline answers questions over the indexed chunks.

The worker (`uv run taskiq worker app.taskiq_broker:broker app.tasks`) is required for
either pipeline to make progress.

## Commands

This project uses **uv** (see `uv.lock`, `.python-version` = 3.13). Requires Python >= 3.13.

```powershell
uv sync                                   # install/lock dependencies (incl. the dev group: pytest)
uv run fastapi dev app/main.py            # dev server with reload on :8000
uv run taskiq worker app.taskiq_broker:broker app.tasks
uv run pytest                             # whole suite
uv run pytest tests/test_history_service.py -k budget   # one file / matching tests
uv run ruff check app tests               # ruff is a dependency but has no config; the tree is not clean yet
```

Docs at `/docs`, root index at `/`, health at `/api/v1/health`.

Alembic **owns the schema**; nothing is created at startup. Run migrations from this directory
(`prepend_sys_path = .` in `alembic.ini` is what makes `import app.models` resolve):

```powershell
uv run alembic upgrade head                        # apply migrations
uv run alembic revision --autogenerate -m "msg"    # generate after model changes
uv run alembic check                               # fail if models drifted from migrations
uv run alembic current                             # show applied revision
```

`alembic/env.py` reads the URL from `settings.DATABASE_URL` and swaps `+asyncpg` for `+psycopg`,
since the sync `engine_from_config` path can't drive asyncpg. `sqlalchemy.url` stays commented out
in `alembic.ini` so credentials live only in `.env`. New ORM models must be imported in
`app/models/__init__.py` or autogenerate won't see them. Always read the generated revision before
applying it. The database needs the **pgvector** extension (a migration enables it).

Docker Compose (`compose.yaml`) runs API, worker, Postgres 17 + pgvector, Redis, and a one-shot
`migrate` service: `docker compose up --build -d`. Add `-f compose.yaml -f compose.dev.yaml` for
bind-mounted source with auto-reload (and to run `alembic revision` inside the container so the
file lands in the checkout). Compose overrides `DATABASE_URL`/`REDIS_URL` with the `db`/`redis`
service hosts; the localhost URLs in `.env` are for running Python directly. The README covers
the rest.

## Tests

`tests/test_<feature>.py`, written as `unittest.TestCase` / `IsolatedAsyncioTestCase` classes and
run through pytest (no asyncio plugin; `pythonpath = ["."]` in `pyproject.toml` makes `app`
importable). Unit tests inject fakes — e.g. `LLMService(chat_model=...)`, `AsyncMock` embedders —
so nothing calls a provider.

The PostgreSQL tests (`test_conversation_service.py`, `test_retrieval_service.py`,
`test_conversation_migration.py`) **skip unless `CHAT_TEST_DATABASE_URL` is set** to an asyncpg
URL for a database with pgvector. They create and drop disposable schemas, so they don't touch
existing tables; the migration test runs the real Alembic revision upgrade/downgrade.

## Configuration

`app/core/config.py` (pydantic-settings, loads `.env`, `case_sensitive=True`, `extra="ignore"`).
Required with no default: `PROJECT_NAME`, `DEEPSEEK_API_KEY`, `DATABASE_URL`, `REDIS_URL`,
`SECRET_KEY`, `OPENAI_API_KEY` (embeddings; may be empty if you never index). Auth defaults:
`ALGORITHM` (HS256), `ACCESS_TOKEN_EXPIRE_MINUTES` (15), `REFRESH_TOKEN_EXPIRE_DAYS` (30).
`ENVIRONMENT` drives the refresh cookie's `Secure` flag (`cookie_secure` is false only in
`development`). `BACKEND_CORS_ORIGINS` is parsed as JSON.

`SECRET_KEY` signs the JWTs and **must be at least 32 bytes** for HS256 — PyJWT warns below
that. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
Rotating it invalidates every issued access token (sessions survive; clients recover on their
next `/auth/refresh`).

`DATABASE_URL` **must use the asyncpg driver**; a validator rewrites `postgres://` and
`postgresql://` to `postgresql+asyncpg://`. On Windows the psycopg async driver cannot run on
`ProactorEventLoop`, and the taskiq worker creates its loop before importing app modules — asyncpg
works on either loop. `main.py` and `taskiq_broker.py` still set `WindowsSelectorEventLoopPolicy`
on win32. psycopg is installed only as Alembic's sync driver.

## Architecture

Request flow is layered: **routes → services → models/schema**.

- `app/main.py` — app, CORS + GZip middleware, and the `AppError` handler that renders
  `{message, data: None, success: False}`. The `lifespan` starts the taskiq broker as a producer
  (skipped in the worker process) and **tolerates Redis being down**: the API boots, and uploads
  return 503 until Redis is reachable. It does not create tables.
- `app/api/router.py` — aggregates v1 routers: `health`, `auth`, `summary`, `project`.
  `app/api/v1/conversations.py` and `documents.py` are **empty placeholders** and not routed.
- `app/api/deps.py` — `DbSession = Annotated[AsyncSession, Depends(get_db)]`,
  `CurrentAuth` (user + session), `CurrentUser`. Every `/summary` and `/project` route requires
  `CurrentUser`.
- `app/api/v1/health.py` — runs `SELECT 1` and pings Redis; returns `healthy`, `degraded`
  (DB up, queue down) or `unhealthy`, with **503 only when the DB is unreachable**. It does not
  check whether a worker is running.
- `app/core/exception.py` — `AppError` subclasses (`NotFoundError`, `ValidationError`,
  `ConflictError`, `RateLimitError`, …). Services raise these; routes don't translate them.

Data & persistence:
- `app/core/database.py` — SQLAlchemy 2.0 **async** engine + `AsyncSessionLocal`. Declarative
  `Base` (with `NAMING_CONVENTION` so Alembic emits stable constraint names), the `utcnow()`
  column-default helper, and `TimestampMixin` (`created_at`/`updated_at`, both `timestamptz`).
  `init_db()` exists for ad-hoc use only and is never called at startup.
- `app/models/summary.py` — `Summary` (`id` UUID pk, nullable `user_id` FK → `users.id`,
  `title`, `file_path`, `content`, `source_text`, `status`). `SummaryStatus`:
  `pending` / `processing` / `completed` / `failed`. `user_id` is nullable while auth beds in.
- `app/models/{user,auth_session,refresh_token}.py` — auth schema. `User` (email unique,
  `hashed_password`, `status: UserStatus`) ← `AuthSession` (device/IP, `expires_at`, `revoked`)
  ← `RefreshToken` (`token_hash` = SHA-256 hex, unique). All FKs are `ON DELETE CASCADE` with
  matching ORM `cascade="all, delete-orphan"`. Cross-model imports are `TYPE_CHECKING`-only;
  relationships resolve through SQLAlchemy's registry.
- RAG models: `Project` ↔ `Document` through `ProjectDocument` (`project_documents`);
  `DocumentSummary`; `EmbeddingGeneration` → `DocumentEmbedding` (pgvector
  `Vector(EMBEDDING_DIMENSION)`); `Conversation` → `ConversationMessage` / `ConversationRun`.
  Status enums are stored as strings (`native_enum=False`). `ConversationRun.snapshot` and
  `.metrics` are Pydantic models (`RunSnapshot`, `RunMetrics`) persisted as JSON.
- `app/schema/` — Pydantic layer. All endpoints return the generic envelope
  `ApiResponse[T]` from `response.py` (`{message, data, success}`), aliased per resource
  (`SummarizeApiResponse`, `CreateProjectResponse`, …).
- **Everything user-facing is owner-scoped.** `SummaryService`, `ProjectService`,
  `ConversationService`, `HistoryService`, `RetrievalService` and `VectorStoreService` all take a
  `user_id` and filter on it; another user's id yields **404, not 403**, so ids aren't confirmed
  to non-owners.

Authentication (`/api/v1/auth/{register,login,refresh,logout,me,sessions}`):
- **Access token** — short-lived JWT (`ACCESS_TOKEN_EXPIRE_MINUTES`, default 15) carrying
  `sub` (user id) and `sid` (session id). Sent as `Authorization: Bearer <token>`.
- **Refresh token** — opaque `secrets.token_urlsafe(48)`; only its SHA-256 digest is stored
  in `refresh_tokens.token_hash`. **Single-use**: `/auth/refresh` revokes the presented token
  and issues a new pair, so a replayed token always 401s.
- **Two delivery modes.** Browsers get the refresh token as an httpOnly cookie (`app/core/
  cookies.py`, `SameSite=Lax`, scoped to `/api/v1/auth`) and it is **omitted from the response
  body**. Native clients send `X-Auth-Mode: bearer` and get it in the body instead, with no
  cookie set. Both `/auth/refresh` and `/auth/logout` read the cookie first, then the body.
  `SameSite=Lax` keeps the cookie off cross-site POSTs, which is what covers CSRF on those two
  POST-only routes — there is no separate CSRF token.
- **Sessions** — `GET /auth/sessions` lists live sessions (the caller's own is flagged
  `current: true`); `DELETE /auth/sessions/{id}` revokes one. Ownership is enforced, and
  someone else's session id returns 404.
- **Cleanup is opportunistic**: `AuthService.prune_expired` runs on login and deletes that
  user's expired/revoked sessions and tokens. No scheduler process — dormant accounts just
  keep harmless dead rows until they return.
- A **new login always creates a new session**, deliberately. `user_agent`/`ip_address` are
  spoofable and non-unique, so matching on them to reuse a session would let anyone with the
  password merge into the victim's session and corrupt both revocation and the audit trail.
  Same-device continuity is `/auth/refresh`'s job — it keeps the session and rotates the token.
- `get_current_user` re-reads the `AuthSession` on **every** request. That's a deliberate
  extra query — it's what makes `/auth/logout` kill an unexpired access token immediately.
- `app/utils/auth.py` — hashing (`pwdlib` Argon2id) and token encode/decode. `verify_password`
  runs against a dummy hash when the user or hash is missing, so login timing can't be used to
  enumerate accounts. `AuthService.authenticate` returns the same message for unknown email and
  wrong password for the same reason.
- `app/services/auth_service.py` — registration (relies on the unique index, not a prior
  SELECT, to settle signup races), session lifecycle, rotation, revocation.

Legacy summary pipeline (`/summary`):
- `app/api/v1/summary.py` — upload checks Redis first, then `ensure_daily_quota`
  (`DAILY_SUMMARY_LIMIT` = 3 non-failed summaries per user per UTC day → `RateLimitError` 429)
  **before** streaming anything. It validates PDF-only, streams to `temp_uploads/<uuid>.pdf`
  (50 MB cap, 1 MB chunks), and calls `SummaryService.start_summary`, which extracts the text
  **inside the request** (PyMuPDF via `asyncio.to_thread`) into `Summary.source_text`. If dispatch fails, the row and file
  are removed and the route returns 503. The worker reads only `source_text`, never the PDF.
- `SummaryService.delete_book` refuses only while the summary is `processing`.
- `app/services/processor_service.py` — `process_summary(summary_id)` opens its own
  `AsyncSessionLocal` session, flips status, and runs `summarize_pdf()`: `chunk_text` (character
  based) → summarize each chunk → join → final synthesis pass. `clean_text()` and
  `extract_and_clean_text_from_pdf()` are **not in the live path**; stored text is raw page text.
- **The map stage runs concurrently.** `summarize_chunks` fans chunks out through
  `asyncio.gather` behind an `asyncio.Semaphore`; `gather` preserves chunk order, which the
  reduce step depends on. Knobs in `config.py`: `SUMMARY_CHUNK_SIZE` (20000 chars),
  `SUMMARY_CHUNK_OVERLAP` (400), `SUMMARY_MAX_CONCURRENCY` (15).
- **`summarize_pdf` is the only place that times anything** — it prints `[timing]` lines per
  stage (chunking / map / reduce / total). Deeper layers stay timing-free on purpose:
  `summarize_chunks` prints a bare `[chunk] n/total returned` progress line and
  `DeepSeekAIService` prints no duration at all. Don't reintroduce per-call timers.
- `app/services/ai_service.py` — `DeepSeekAIService` wraps the **OpenAI SDK's** `AsyncOpenAI`
  client pointed at `DEEPSEEK_BASE_URL` (DeepSeek is OpenAI-compatible). Model
  `deepseek-v4-flash` for both map and reduce; the reduce pass only gets a longer timeout. Each
  call logs one `[deepseek]` line — model, finish reason, token counts, and a
  `LOG_PREVIEW_CHARS`-truncated preview.
- **Every call is tagged with `user_id`.** DeepSeek uses it for content-safety, KVCache and
  scheduling isolation between end users sharing one API key. It is a DeepSeek extension,
  *not* an OpenAI parameter, so it goes through `extra_body={"user_id": ...}` — passing it
  as `user=` would silently tag nothing. Values are checked against `[a-zA-Z0-9\-_]{1,512}` and
  dropped with a log line if malformed — a bad value would 400 the whole summary, and losing
  isolation beats losing the run. `LLMService.generate_response` also sends `user_id` through
  `extra_body`, but does not validate it.
- **Provider errors never leave the worker log.** DeepSeek's error text describes *our*
  account (402 insufficient balance, 401 bad key, quota), so none of it is persisted or
  returned; a failed run reaches the client purely as `SummaryStatus.FAILED`, and there is
  deliberately no `error_message` column. `ai_service` switches on `APIStatusError.status_code`
  rather than exception class — 402 has no dedicated SDK class — attaching an `OPERATOR_HINTS`
  line for whoever reads the log. If a user-facing reason is ever wanted, it needs a separate
  hand-written message, never provider text.
- **Retries are the SDK's, not hand-rolled.** Clients are built with
  `max_retries=DEEPSEEK_MAX_RETRIES` (3) and an explicit timeout, since the 600s default would
  pin a semaphore slot for ten minutes. `DEEPSEEK_TIMEOUT_SECONDS` (120) covers map calls;
  `DEEPSEEK_REDUCE_TIMEOUT_SECONDS` (300) the reduce pass.
- `app/utils/prompts.py` — legacy `CHUNK_PROMPT` / `SUMMARY_PROMPT` (`.format(text=...)`).

Projects, indexing and RAG:
- `POST /api/v1/project` (multipart `title` + `file`) → `ProjectService.create_project`:
  `DocumentService.create_document` streams and SHA-256-hashes the upload to `temp_uploads/`
  (flush only — **the caller owns the transaction**), then the project and `ProjectDocument`
  link are added and committed once; the stored file is deleted on failure. The route then
  enqueues `summarize_document_task` and `embed_document_task`. `GET /project` lists the
  caller's projects; `GET /project/{id}` adds the first document's latest summary.
- `app/workers/` — `DocumentSummaryWorker` and `DocumentEmbeddingWorker`. Both **read the PDF
  from disk** (`DocumentService.storage_path`, LangChain/pypdf loaders in
  `app/services/indexing/loaders.py`), so unlike the legacy worker they need the upload volume.
  Each locks the `Document` row `FOR UPDATE` and exits if its status is already
  `STARTED`/`COMPLETED`; the summary job also exits if `summary_job_id` changed. They commit the
  `STARTED` flip before doing provider work and mark `FAILED` (then re-raise) on error.
- `app/tasks.py` is the worker's import target; it re-exports the `app/workers` tasks. **A new
  task module must be imported there** or the worker never registers it.
- `app/services/indexing/` — `splitters.py` (token-based via `tiktoken`/`CHUNK_ENCODING`;
  `SUMMARY_CHUNK_TOKENS` for summaries, `RAG_CHUNK_TOKENS`/`RAG_CHUNK_OVERLAP_TOKENS` for
  embeddings), `embedder.py` (LangChain `OpenAIEmbeddings`, `EMBEDDING_MODEL`), and
  `vector_store.py`. `VectorStoreService.add_document_chunks` replaces the document's **entire
  active `EmbeddingGeneration`** in one transaction (savepoint on failure keeps the old one);
  never index part of a document. `similarity_search` filters on ownership, project membership,
  active completed generation, model and dimension. `settings.rag_chunking_version` is stamped
  on each generation — bump its `-v1` suffix when the splitting strategy changes. Changing
  `EMBEDDING_DIMENSION` needs a migration and re-indexing.
- `app/services/llm/llm_service.py` — `LLMService`, the LangChain `ChatOpenAI` client pointed at
  DeepSeek (lazy model creation; pass `chat_model=` in tests). Provides `generate`,
  `generate_response` (optional `max_tokens`/`user_id` via `extra_body`) and `stream`. Prompts are
  `ChatPromptTemplate`s in `app/services/llm/prompts.py`. Use this, not `DeepSeekAIService`, for
  new code.

Conversations (services only; the design notes in `docs/` are authoritative — read the matching
one before changing a stage):
- A turn is a user + assistant `ConversationMessage` pair and a `ConversationRun` (retries add
  attempts). `RunStatus`: `pending` / `running` / `completed` / `failed` / `cancelled`; `run.stage`
  advances `reserved` → `starting` → `rewritten` → `retrieved` → `context_ready` → `generating` →
  terminal. Each stage's output is saved with `ConversationService.checkpoint` into
  `run.snapshot`.
- `ConversationService` **never calls a provider and never holds a transaction across
  inference**: the caller commits each reservation/checkpoint before external work, and every
  turn mutation locks the conversation row first (one active run per conversation).
- Stages: `HistoryService` (completed exchanges within a token budget, default 2000) →
  `QueryRewriteService` (standalone search query; the original question is never replaced in
  the answer prompt or the stored message) → `RetrievalService` (top-k chunks via
  `VectorStoreService`) → `AnswerContextService` (builds and saves the exact prompt; may drop
  whole passages to fit) → `AnswerGenerationService` (`claim_generation`, one model call from the
  saved prompt, publish only on a clean `stop`). Generation failures surface as
  `AnswerGenerationError` (502) with an internal code; provider text is never exposed.

## Conventions

- Everything DB-touching is `async`; use `AsyncSession` and `await`. Never use the sync
  SQLAlchemy API here.
- Services that build up rows (`DocumentService`, `ConversationService`, `VectorStoreService`)
  flush but leave commit/rollback to the caller; workers open their own `AsyncSessionLocal`.
- Endpoints return `ApiResponse[...]` envelopes; validate ORM objects into response schemas with
  `Model.model_validate(obj)` (schemas use `from_attributes=True`).
- Task arguments are strings (UUIDs via `str(...)`) so they serialize across the broker.
- Newer modules give every function a docstring stating its purpose; match that there.
