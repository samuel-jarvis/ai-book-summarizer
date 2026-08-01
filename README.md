# AI Book Summarizer API

An asynchronous FastAPI service that turns long-form PDF books into structured AI summaries. The API extracts text with PyMuPDF, queues expensive work in Redis, processes it with Taskiq, and persists summary state in PostgreSQL while clients poll for completion. Summaries belong to authenticated users, with session-based token auth and per-device revocation.

This repository contains the backend for **Lumen AI**. The React client lives in the [Lumen AI frontend repository](https://github.com/samuel-jarvis/lumen-ai).

> **Project status:** Active portfolio project. Authentication, per-user libraries, and database migrations are implemented; the remaining production-readiness work is documented in [Roadmap and known limitations](#roadmap-and-known-limitations).

## Highlights

- **Non-blocking processing** — uploads return quickly while PDF summarization runs in a Taskiq worker.
- **Hierarchical summarization** — long documents are split into overlapping chunks, summarized individually, then synthesized into one cohesive result.
- **Session-based authentication** — Argon2id password hashing, short-lived JWT access tokens, and single-use rotating refresh tokens bound to revocable sessions.
- **Per-user libraries** — every summary query is scoped to its owner; another user's summary is indistinguishable from one that does not exist.
- **Device management** — users can list where they are signed in and revoke any session, which takes effect on the next request rather than at token expiry.
- **Dual-client token delivery** — browsers receive the refresh token as an httpOnly cookie; native clients opt into bearer mode and manage it themselves.
- **Versioned schema** — Alembic owns the database schema, with a naming convention that keeps generated constraint names stable.
- **Persistent job lifecycle** — PostgreSQL tracks `pending`, `processing`, `completed`, and `failed` states.
- **Resilient queue integration** — Redis availability is exposed through health checks, unavailable queues produce an explicit `503`, and workers retry Redis connections.
- **Async data layer** — SQLAlchemy 2.0, asyncpg, and async sessions are used throughout the API.
- **Defensive uploads** — the API accepts PDFs only, streams uploads to disk, and enforces a 50 MB server-side limit.
- **Consistent API contracts** — typed Pydantic schemas and a shared response envelope keep client integration predictable.

## Architecture

```mermaid
flowchart LR
    UI["React client"] -->|"Sign in"| AUTH["Auth routes"]
    AUTH -->|"Session + rotating tokens"| DB[(PostgreSQL)]
    UI -->|"Upload PDF (Bearer token)"| API["FastAPI API"]
    API -->|"Extract text + create record"| DB
    API -->|"Enqueue summary ID"| REDIS[(Redis Streams)]
    REDIS --> WORKER["Taskiq worker"]
    WORKER -->|"Chunk + summarize"| AI["DeepSeek API"]
    WORKER -->|"Update status + content"| DB
    UI -->|"Poll summary status"| API
```

The request path deliberately stops before model inference. This keeps the HTTP API responsive and allows the worker tier to scale independently from the web tier.

## Tech stack

| Area | Technology |
| --- | --- |
| API | Python 3.13, FastAPI, Pydantic |
| Database | PostgreSQL, SQLAlchemy 2.0, asyncpg |
| Migrations | Alembic (psycopg driver) |
| Authentication | PyJWT, pwdlib with Argon2id |
| Background jobs | Taskiq, Redis Streams |
| PDF processing | PyMuPDF |
| AI integration | DeepSeek through the OpenAI-compatible async client |
| Package management | uv |

## Authentication

Two credentials with different lifetimes and different jobs:

- **Access token** — a short-lived JWT (15 minutes by default) carrying the user id (`sub`) and session id (`sid`). Sent as `Authorization: Bearer <token>`.
- **Refresh token** — an opaque 384-bit random string. Only its SHA-256 digest is stored, so a database dump cannot be replayed against the API. It is **single-use**: refreshing revokes the presented token and issues a new pair, so a replayed token always fails.

Both belong to an `AuthSession`, which represents one credential grant — one successful sign-in. Sessions are re-read on every request, which costs a query but makes revocation and sign-out take effect immediately instead of waiting out the access token's lifetime.

**Delivery modes.** Browsers receive the refresh token as an httpOnly cookie (`SameSite=Lax`, scoped to `/api/v1/auth`) and it is omitted from the response body entirely. Native clients send `X-Auth-Mode: bearer` and receive it in the body with no cookie set. `/auth/refresh` and `/auth/logout` read the cookie first, then fall back to the request body.

`SameSite=Lax` is what protects the cookie-bearing routes from CSRF, and it works because those routes are POST-only. See [Deployment](#deployment) for the same-site constraint this places on production domains.

**Housekeeping.** Expired and revoked sessions and tokens are deleted opportunistically during sign-in, so no scheduler process is required.

## Core workflow

1. The client registers or signs in and receives an access token plus a refresh credential.
2. The client submits a title and PDF using multipart form data with a bearer token.
3. The API validates and streams the file to `temp_uploads/`.
4. PyMuPDF extracts the source text and the API stores a `pending` record owned by the caller.
5. The summary ID is dispatched to the Redis-backed Taskiq queue.
6. A worker chunks the text with overlap, summarizes each chunk, and creates a final synthesis.
7. The worker saves the result and moves the record to `completed`; errors move it to `failed`.
8. The client polls the detail endpoint and renders the finished Markdown-style response.

## Getting started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL
- Redis
- A DeepSeek API key

### 1. Install dependencies

```powershell
cd ai-book-summarizer
uv sync
```

### 2. Configure the environment

Create `ai-book-summarizer/.env` with values for your local services. The file is gitignored and must never be committed.

```dotenv
PROJECT_NAME="AI Book Summarizer"
DEEPSEEK_API_KEY="your-deepseek-api-key"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/book_summarizer"
REDIS_URL="redis://localhost:6379/0"
SECRET_KEY="generate-a-real-one-see-below"
ENVIRONMENT="development"
DEBUG=false
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

`SECRET_KEY` signs the JWTs and **must be at least 32 bytes** for HS256 — PyJWT warns below that. Generate one with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Optional auth settings, shown with their defaults: `ALGORITHM="HS256"`, `ACCESS_TOKEN_EXPIRE_MINUTES=15`, `REFRESH_TOKEN_EXPIRE_DAYS=30`.

`DATABASE_URL` must target PostgreSQL. Standard `postgres://` and `postgresql://` URLs are automatically converted to the asyncpg driver format.

### 3. Apply migrations

The schema is owned by Alembic and is **not** created at startup. Run this against a new database, and after pulling any change that touches a model:

```powershell
uv run alembic upgrade head
```

### 4. Start the API

```powershell
uv run fastapi dev app/main.py
```

The API starts at `http://localhost:8000`.

Useful endpoints:

- OpenAPI documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`
- Root service metadata: `http://localhost:8000/`

### 5. Start the worker

Run this in a second terminal from `ai-book-summarizer/`:

```powershell
uv run taskiq worker app.taskiq_broker:broker app.tasks
```

PostgreSQL, Redis, the API, and the worker must all be running for end-to-end summarization.

### 6. Start the frontend (optional)

```powershell
cd ../ai-book-frontend
npm install
npm run dev
```

The Vite client runs at `http://localhost:3000` and proxies `/api` requests to the backend. Use that origin rather than `http://localhost:8000` directly — the proxy keeps the browser same-origin, which is what allows the refresh cookie to work in development.

## Working with migrations

```powershell
uv run alembic upgrade head                        # apply
uv run alembic revision --autogenerate -m "msg"    # generate after model changes
uv run alembic current                             # show applied revision
uv run alembic check                               # fail if models have drifted from the schema
```

Run Alembic from `ai-book-summarizer/`; `prepend_sys_path = .` in `alembic.ini` is what makes `import app.models` resolve. New models must be imported in `app/models/__init__.py` or autogenerate will silently omit them. Always read a generated revision before applying it.

`alembic/env.py` reads the URL from `settings.DATABASE_URL` and swaps `+asyncpg` for `+psycopg`, since the synchronous migration engine cannot drive the async driver. `sqlalchemy.url` is intentionally left commented out in `alembic.ini` so credentials stay in `.env`.

## API reference

Successful typed endpoints use the following envelope:

```json
{
  "success": true,
  "message": "Summary retrieved successfully",
  "data": {}
}
```

### Authentication

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/auth/register` | Create an account |
| `POST` | `/api/v1/auth/login` | Sign in and open a session |
| `POST` | `/api/v1/auth/refresh` | Exchange a refresh token for a new pair |
| `POST` | `/api/v1/auth/logout` | Revoke the current session |
| `GET` | `/api/v1/auth/me` | Retrieve the signed-in user |
| `GET` | `/api/v1/auth/sessions` | List live sessions; the caller's own is flagged `current` |
| `DELETE` | `/api/v1/auth/sessions/{id}` | Revoke one of the caller's sessions |

### Summaries

All summary endpoints require a bearer token and operate only on the caller's own records.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Check database and task-queue availability (public) |
| `GET` | `/api/v1/summary` | List the caller's summaries and their current status |
| `POST` | `/api/v1/summary` | Upload a PDF using `title` and `file` form fields |
| `GET` | `/api/v1/summary/{id}` | Retrieve status and completed summary content |
| `PUT` | `/api/v1/summary/{id}` | Rename a summary using a `title` form field |
| `DELETE` | `/api/v1/summary/{id}` | Delete an eligible summary |

A summary belonging to another user returns `404`, not `403`, so the API never confirms that an inaccessible id exists.

## Project structure

```text
alembic/                 # Migration environment and versions
app/
├── api/                 # Versioned routes and dependencies
│   ├── deps.py          # DbSession, CurrentUser, CurrentAuth
│   └── v1/              # health, auth, summary routers
├── core/                # Settings, cookies, database, application errors
├── models/              # User, AuthSession, RefreshToken, Summary
├── schema/              # Pydantic request/response contracts
├── services/            # Auth, PDF, AI, and persistence workflows
├── utils/               # Password hashing, tokens, prompt templates
├── main.py              # FastAPI application and lifespan
├── taskiq_broker.py     # Redis broker and queue health checks
└── tasks.py             # Background task entry points
```

The backend follows a routes → services → models/schemas structure so transport, business logic, and persistence concerns remain separate. Ownership filtering lives in the service layer, not in route handlers.

## Deployment

The schema is no longer created at startup, so **migrations must run as part of deployment**. On Railway, set the pre-deploy command on the API service only:

```bash
uv run alembic upgrade head
```

The worker service must not run migrations — two services migrating concurrently is a race worth avoiding.

Production environment notes:

- Set `ENVIRONMENT=production`. The `Secure` flag on the refresh cookie is derived from it, and an unset value silently defaults to development.
- Set a fresh `SECRET_KEY`, distinct from the local one.
- `BACKEND_CORS_ORIGINS` is parsed as JSON, so it must be a bracketed array: `["https://app.example.com"]`.
- **Same-site constraint:** the refresh cookie is `SameSite=Lax`, so it is not sent on cross-site requests. The frontend and API must share a registrable domain (`app.example.com` and `api.example.com`). Sibling subdomains on a PaaS domain listed in the Public Suffix List count as cross-site and will break token refresh. Relaxing to `SameSite=None` requires adding CSRF protection to the cookie-bearing routes.

## Roadmap and known limitations

The core summarization flow and authentication work. The following items remain before this should be considered production-ready.

### Highest priority

- [ ] Add backend unit/integration tests and end-to-end worker tests; no automated test suite exists yet.
- [ ] Add rate limiting to `/auth/login`. Argon2 makes each attempt expensive, which is both the brute-force defence and a denial-of-service vector.
- [ ] Add linting, static type checks, and CI for every pull request.
- [ ] Remove uploaded files after processing/deletion and move production uploads to durable object storage.
- [ ] Replace development `print` statements with structured, redacted logging; model responses must not be logged in production.

### Reliability and security

- [ ] Add email verification. `UserStatus.PENDING` exists but registration currently activates accounts immediately.
- [ ] Add password reset and change-password flows.
- [ ] Add refresh-token replay detection (a `replaced_by_id` self-reference), so a reused token revokes the whole session rather than simply failing.
- [ ] Cap concurrent sessions per user to bound the blast radius of a stolen credential.
- [ ] Backfill `summaries.user_id` and tighten it to `NOT NULL`; it is nullable while auth beds in.
- [ ] Add job retries with backoff, idempotency, timeouts, cancellation, and dead-letter handling.
- [ ] Add quota controls and stronger file-content validation beyond the filename extension.
- [ ] Add error monitoring, metrics, tracing, and worker/job dashboards.
- [ ] Containerize and document repeatable deployment for the API, worker, PostgreSQL, and Redis.

### Product capabilities

- [ ] Add OCR for scanned or image-only PDFs.
- [ ] Add EPUB support; the current implementation accepts PDFs only.
- [ ] Report page/chunk progress instead of status-only polling.
- [ ] Support configurable summary length, audience, tone, and output format.
- [ ] Add pagination and sorting for large summary libraries.

### Completed

- [x] Alembic migrations and a deployment migration workflow.
- [x] Authentication, user ownership, authorization, and per-user libraries.

## Engineering notes

- Uploaded files, `.env`, virtual environments, and build artifacts are intentionally ignored.
- The default AI model is configured in `DeepSeekAIService`; validate model availability before deployment.
- Summary generation is sequential per chunk today. Concurrency should be introduced only alongside provider rate-limit and cost controls.
- A new sign-in always opens a new session. `user_agent` and `ip_address` are spoofable and non-unique, so matching on them to reuse a session would let anyone holding the password merge into the victim's session and corrupt both revocation and the audit trail. Same-device continuity is `/auth/refresh`'s job.
- Password verification runs against a dummy hash when no user exists, so response timing cannot be used to enumerate accounts. Unknown-email and wrong-password failures return the same message for the same reason.
- Passwords use Argon2id (deliberately slow, since humans choose guessable secrets); refresh tokens use plain SHA-256 (already high-entropy, so there is nothing to brute-force and lookups stay a single indexed query).
- Because refresh tokens are single-use, clients must deduplicate concurrent refreshes. Two parallel requests that both refresh will invalidate each other and sign the user out.

## Author

Built by [Samuel Jarvis](https://github.com/samuel-jarvis) as a full-stack portfolio project focused on asynchronous Python systems, AI integration, and practical product engineering.
