# AI Book Summarizer API

An asynchronous FastAPI service that turns long-form PDF books into structured AI summaries. The API extracts text with PyMuPDF, queues expensive work in Redis, processes it with Taskiq, and persists summary state in PostgreSQL while clients poll for completion.

This repository contains the backend for **Lumen AI**. The React client lives in the [Lumen AI frontend repository](https://github.com/samuel-jarvis/lumen-ai).

> **Project status:** Active portfolio project. The end-to-end local workflow is implemented; the remaining production-readiness work is documented in [Roadmap and known limitations](#roadmap-and-known-limitations).

## Highlights

- **Non-blocking processing** — uploads return quickly while PDF summarization runs in a Taskiq worker.
- **Hierarchical summarization** — long documents are split into overlapping chunks, summarized individually, then synthesized into one cohesive result.
- **Persistent job lifecycle** — PostgreSQL tracks `pending`, `processing`, `completed`, and `failed` states.
- **Resilient queue integration** — Redis availability is exposed through health checks, unavailable queues produce an explicit `503`, and workers retry Redis connections.
- **Async data layer** — SQLAlchemy 2.0, asyncpg, and async sessions are used throughout the API.
- **Defensive uploads** — the API accepts PDFs only, streams uploads to disk, and enforces a 35 MB server-side limit.
- **Consistent API contracts** — typed Pydantic schemas and a shared response envelope keep client integration predictable.
- **Operational basics** — CORS configuration, GZip responses, centralized application errors, and dependency health reporting are included.

## Architecture

```mermaid
flowchart LR
    UI["React client"] -->|"Upload PDF"| API["FastAPI API"]
    API -->|"Extract text + create record"| DB[(PostgreSQL)]
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
| Background jobs | Taskiq, Redis Streams |
| PDF processing | PyMuPDF |
| AI integration | DeepSeek through the OpenAI-compatible async client |
| Package management | uv |

## Core workflow

1. The client submits a title and PDF using multipart form data.
2. The API validates and streams the file to `temp_uploads/`.
3. PyMuPDF extracts the source text and the API stores a `pending` record.
4. The summary ID is dispatched to the Redis-backed Taskiq queue.
5. A worker chunks the text with overlap, summarizes each chunk, and creates a final synthesis.
6. The worker saves the result and moves the record to `completed`; errors move it to `failed`.
7. The client polls the detail endpoint and renders the finished Markdown-style response.

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
ENVIRONMENT="development"
DEBUG=false
BACKEND_CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
```

`DATABASE_URL` must target PostgreSQL. Standard `postgres://` and `postgresql://` URLs are automatically converted to the asyncpg driver format.

### 3. Start the API

```powershell
uv run fastapi dev app/main.py
```

The API starts at `http://localhost:8000`. On startup, the current SQLAlchemy schema is created if it does not exist.

Useful endpoints:

- OpenAPI documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`
- Root service metadata: `http://localhost:8000/`

### 4. Start the worker

Run this in a second terminal from `ai-book-summarizer/`:

```powershell
uv run taskiq worker app.taskiq_broker:broker app.tasks
```

PostgreSQL, Redis, the API, and the worker must all be running for end-to-end summarization.

### 5. Start the frontend (optional)

```powershell
cd ../ai-book-frontend
npm install
npm run dev
```

The Vite client runs at `http://localhost:3000` and proxies `/api` requests to the backend.

## API reference

Successful typed endpoints use the following envelope:

```json
{
  "success": true,
  "message": "Summary retrieved successfully",
  "data": {}
}
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Check database and task-queue availability |
| `GET` | `/api/v1/summary` | List summaries and their current status |
| `POST` | `/api/v1/summary` | Upload a PDF using `title` and `file` form fields |
| `GET` | `/api/v1/summary/{id}` | Retrieve status and completed summary content |
| `PUT` | `/api/v1/summary/{id}` | Rename a summary using a `title` form field |
| `DELETE` | `/api/v1/summary/{id}` | Delete an eligible summary |

## Project structure

```text
app/
├── api/                 # Versioned routes and dependencies
├── core/                # Settings, database, and application errors
├── models/              # SQLAlchemy entities and status enum
├── schema/              # Pydantic request/response contracts
├── services/            # PDF, AI, and persistence workflows
├── utils/               # Prompt templates
├── main.py              # FastAPI application and lifespan
├── taskiq_broker.py     # Redis broker and queue health checks
└── tasks.py             # Background task entry points
```

The backend follows a routes → services → models/schemas structure so transport, business logic, and persistence concerns remain separate.

## Roadmap and known limitations

The core summarization flow works. The following items remain before this should be considered production-ready.

### Highest priority

- [ ] Add backend unit/integration tests and end-to-end worker tests; no automated test suite exists yet.
- [ ] Add linting, static type checks, and CI for every pull request.
- [ ] Add Alembic migration files and a deployment migration workflow; tables are currently created from SQLAlchemy metadata at startup.
- [ ] Remove uploaded files after processing/deletion and move production uploads to durable object storage.
- [ ] Replace development `print` statements with structured, redacted logging; model responses must not be logged in production.

### Reliability and security

- [ ] Add authentication, user ownership, authorization, and per-user libraries.
- [ ] Add job retries with backoff, idempotency, timeouts, cancellation, and dead-letter handling.
- [ ] Add request throttling, quota controls, and stronger file-content validation beyond the filename extension.
- [ ] Add error monitoring, metrics, tracing, and worker/job dashboards.
- [ ] Containerize and document repeatable deployment for the API, worker, PostgreSQL, and Redis.

### Product capabilities

- [ ] Add OCR for scanned or image-only PDFs.
- [ ] Add EPUB support; the current implementation accepts PDFs only.
- [ ] Report page/chunk progress instead of status-only polling.
- [ ] Support configurable summary length, audience, tone, and output format.
- [ ] Add pagination and sorting for large summary libraries.

## Engineering notes

- Uploaded files, `.env`, virtual environments, and build artifacts are intentionally ignored.
- The default AI model is configured in `DeepSeekAIService`; validate model availability before deployment.
- Summary generation is sequential per chunk today. Concurrency should be introduced only alongside provider rate-limit and cost controls.

## Author

Built by [Samuel Jarvis](https://github.com/samuel-jarvis) as a full-stack portfolio project focused on asynchronous Python systems, AI integration, and practical product engineering.
