# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

FastAPI service that summarizes uploaded PDF books via an LLM. `POST /summary` uploads a
PDF, creates a `PENDING` `Summary` row, and dispatches a taskiq background task
(`process_pdf_task`) that runs the map-reduce summarization pipeline through DeepSeek and
writes the result back to the row (`PROCESSING` → `COMPLETED`/`FAILED`, generated text in
`Summary.content`). Running the worker (`uv run taskiq worker app.taskiq_broker:broker
app.tasks`) is required for uploads to actually get summarized.

## Commands

This project uses **uv** (see `uv.lock`, `.python-version` = 3.13). Requires Python >= 3.13.

```powershell
uv sync                                   # install/lock dependencies
uv run fastapi dev app/main.py            # dev server with reload (fastapi[standard] CLI)
uv run uvicorn app.main:app --reload      # equivalent
uv run python -m app.main                 # runs via the __main__ block (port 8000, reload)
```

Docs at `/docs`, root index at `/`, health at `/api/v1/health`.

There is **no test suite** (no pytest/test config in the repo) and no linter configured.

Alembic **owns the schema**. Run migrations from this directory (`prepend_sys_path = .` in
`alembic.ini` is what makes `import app.models` resolve):

```powershell
uv run alembic upgrade head                        # apply migrations
uv run alembic revision --autogenerate -m "msg"    # generate after model changes
uv run alembic current                             # show applied revision
```

`alembic/env.py` reads the URL from `settings.DATABASE_URL` and swaps `+asyncpg` for `+psycopg`,
since the sync `engine_from_config` path can't drive asyncpg. `sqlalchemy.url` stays commented out
in `alembic.ini` so credentials live only in `.env`. Always read the generated revision before
applying it.

## Configuration

`app/core/config.py` (pydantic-settings, loads `.env`, `case_sensitive=True`). Required
env vars with no default: `PROJECT_NAME`, `DEEPSEEK_API_KEY`, `DATABASE_URL`, `REDIS_URL`,
`SECRET_KEY`. Auth tuning has defaults: `ALGORITHM` (HS256), `ACCESS_TOKEN_EXPIRE_MINUTES`
(15), `REFRESH_TOKEN_EXPIRE_DAYS` (30).

`SECRET_KEY` signs the JWTs and **must be at least 32 bytes** for HS256 — PyJWT warns below
that. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
Rotating it invalidates every issued access token (sessions survive; clients recover on their
next `/auth/refresh`).

`DATABASE_URL` **must use the asyncpg driver** (`postgresql+asyncpg://...`), not psycopg.
On Windows the psycopg async driver cannot run on `ProactorEventLoop`, and the taskiq
worker creates its loop before importing app modules (so an event-loop-policy override
can't fix it) — asyncpg works on either loop. psycopg[binary] stays installed only as the
natural sync driver should Alembic migrations ever be added.

## Architecture

Request flow is layered: **routes → services → models/schema**.

- `app/main.py` — app factory, CORS + GZip middleware, `lifespan` calls `init_db()` on
  startup. Mounts `api_router` under `settings.API_V1_PREFIX` (`/api/v1`).
- `app/api/router.py` — aggregates v1 routers (`health`, `auth`, `summary`).
- `app/api/deps.py` — `DbSession = Annotated[AsyncSession, Depends(get_db)]`; inject this
  into every route needing DB access. `CurrentUser = Annotated[User, Depends(get_current_user)]`
  for authenticated routes — **every `/summary` route requires it**.
- `app/api/v1/summary.py` — upload endpoint validates PDF-only, streams to
  `temp_uploads/<uuid>.pdf` (50 MB cap, 1 MB chunks) to avoid path traversal, then calls
  `SummaryService`.

Data & persistence:
- `app/core/database.py` — SQLAlchemy 2.0 **async** engine + `AsyncSessionLocal`. Declarative
  `Base` (with `NAMING_CONVENTION` so Alembic emits stable constraint names), the `utcnow()`
  column-default helper, and `TimestampMixin` (`created_at`/`updated_at`, both `timestamptz`).
  `init_db()` still exists for ad-hoc/test use but is **no longer called at startup** — the
  lifespan does not create tables. New ORM models must be imported via `app/models/__init__.py`
  or Alembic autogenerate won't see them.
- `app/models/summary.py` — `Summary` (`id` UUID pk, nullable `user_id` FK → `users.id`,
  `title`, `file_path`, `content`, `source_text`, `status` enum, timestamps). `SummaryStatus`:
  `pending` / `processing` / `completed` / `failed`. `user_id` is nullable while auth is being
  built out; tighten to `NOT NULL` once rows are backfilled.
- `app/models/{user,auth_session,refresh_token}.py` — auth schema. `User` (email unique,
  `hashed_password`, `status: UserStatus`) ← `AuthSession` (device/IP, `expires_at`, `revoked`)
  ← `RefreshToken` (`token_hash` = SHA-256 hex, unique). All FKs are `ON DELETE CASCADE` with
  matching ORM `cascade="all, delete-orphan"`. Cross-model imports are `TYPE_CHECKING`-only;
  relationships resolve through SQLAlchemy's registry.
- `app/schema/` — Pydantic layer. All endpoints return the generic envelope
  `ApiResponse[T]` from `response.py` (`{message, data, success}`), aliased per-resource
  in `summary.py` (`SummarizeApiResponse`, `SummarizeListApiResponse`).
- `app/services/summary_service.py` — DB CRUD for `Summary`; instantiated per-request with
  the injected session (`SummaryService(db)`).

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
- **Summaries are owner-scoped.** Every `SummaryService` method takes `user_id` and filters on
  it; another user's summary returns **404, not 403**, so ids aren't confirmed to non-owners.

Summarization pipeline (map-reduce):
- `app/services/processor_service.py` — `summarize_pdf()` orchestrates: extract text with
  **pymupdf** (`import fitz`) → `clean_text` (regex de-hyphenation/whitespace) →
  `chunk_text` → summarize each chunk → combine → final synthesis pass.
  `process_summary(summary_id)` wraps it for the worker: opens its own
  `AsyncSessionLocal` session, transitions the record's status, and persists the result to
  `Summary.content`.
- **The map stage runs concurrently.** `summarize_chunks` fans every chunk out through
  `asyncio.gather` behind an `asyncio.Semaphore`; `asyncio.gather` is what preserves chunk
  order, which the reduce step depends on. Tuning lives in `config.py`:
  `SUMMARY_CHUNK_SIZE` (24000 chars ≈ 6k tokens), `SUMMARY_CHUNK_OVERLAP` (400), and
  `SUMMARY_MAX_CONCURRENCY` (6). Chunk size was 4000 with a serial loop, which meant
  ~167 sequential calls for a 300-page book.
- **`summarize_pdf` is the only place that times anything** — it prints `[timing]` lines per
  stage (chunking / map / reduce / total). Deeper layers stay timing-free on purpose:
  `summarize_chunks` prints a bare `[chunk] n/total returned` progress line and
  `DeepSeekAIService` prints no duration at all. Don't reintroduce per-call timers.
- `app/services/ai_service.py` — `DeepSeekAIService` wraps the **OpenAI SDK's**
  `AsyncOpenAI` client pointed at `DEEPSEEK_BASE_URL` (DeepSeek is OpenAI-compatible).
  Default model `deepseek-v4-flash`; the reduce pass overrides to `deepseek-v4-pro`.
  Each call logs one `[deepseek]` line — model, finish reason, token counts, and a
  `LOG_PREVIEW_CHARS`-truncated preview of the text (not the whole response object).
- **Every call is tagged with `user_id`.** DeepSeek uses it for content-safety, KVCache and
  scheduling isolation between end users sharing one API key. It is a DeepSeek extension,
  *not* an OpenAI parameter, so it goes through `extra_body={"user_id": ...}` — passing it
  as `user=` would silently tag nothing. The value is the owning `User.id` UUID, threaded
  `process_summary → summarize_pdf → summarize_chunks/summarize_combined_summary →
  summarize_text`. `Summary.user_id` is still nullable, so `None` is normal and simply
  omits the field. Values are checked against `[a-zA-Z0-9\-_]{1,512}` and dropped with a
  log line if malformed — a bad value would 400 the whole summary, and losing isolation
  beats losing the run.
- **Provider errors never leave the worker log.** DeepSeek's error text describes *our*
  account (402 insufficient balance, 401 bad key, quota), so none of it is persisted or
  returned; a failed run reaches the client purely as `SummaryStatus.FAILED`, and there is
  deliberately no `error_message` column or field. `ai_service` catches `APIStatusError`
  and switches on `exc.status_code` rather than on exception class — 402 has no dedicated
  class in the SDK and would otherwise go unclassified — attaching an `OPERATOR_HINTS`
  line for whoever reads the log. If a user-facing failure reason is ever wanted, it needs
  a separate hand-written message, never provider text.
- **Retries are the SDK's, not hand-rolled.** `AsyncOpenAI` is constructed with
  `max_retries=DEEPSEEK_MAX_RETRIES` (3) and an explicit `timeout`; it already backs off
  exponentially and honours `Retry-After` for connection errors, timeouts, 408/409/429 and
  5xx. The timeout must be set — the 600s default would pin a semaphore slot for ten
  minutes. `DEEPSEEK_TIMEOUT_SECONDS` (120) covers map calls; the reduce pass is
  constructed with `DEEPSEEK_REDUCE_TIMEOUT_SECONDS` (300) since it has the largest input
  and longest generation. Exhausted timeouts and 429s are re-raised as `RuntimeError` with
  the knob to turn named in the message.
- `app/utils/prompts.py` — `CHUNK_PROMPT` (per-chunk) and `SUMMARY_PROMPT` (final
  synthesis) templates; both use `.format(text=...)`.

Background tasks:
- `app/taskiq_broker.py` — taskiq `RedisStreamBroker` (`broker`). The API process starts
  it as a producer in `main.py`'s `lifespan` (guarded by `broker.is_worker_process`).
- `app/tasks.py` — `process_pdf_task(summary_id: str)` calls `process_summary`. The
  `summary_id` is passed as a string so it serializes across the broker.
- Run the worker with `uv run taskiq worker app.taskiq_broker:broker app.tasks`.

## Conventions

- Everything DB-touching is `async`; use `AsyncSession` and `await`. Never use the sync
  SQLAlchemy API here.
- Endpoints return `ApiResponse[...]` envelopes; validate ORM objects into response
  schemas with `SummarizeResponse.model_validate(obj)` (schemas use
  `from_attributes=True`).
- The `/api/v1/health` endpoint currently returns a hard-coded `"status": "unhealthy"`
  regardless of the DB check — a known quirk, not a live outage signal.
