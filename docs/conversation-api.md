# Conversation API and frontend handoff

The conversation backend is connected through authenticated HTTP routes and a
Taskiq background task. The frontend can create conversations, submit messages,
poll answers, reload history, display sources, cancel attempts, and retry failures.
New functions have docstrings. No additional database migration is required.

## Running it

The API, PostgreSQL with pgvector, Redis, and a worker are required. Retrieval
uses eligible indexed documents in the selected project. Without matching
passages, the answer prompt tells the model to acknowledge missing evidence.

```powershell
uv run alembic upgrade head
uv run fastapi dev app/main.py
# In a second terminal:
uv run taskiq worker app.taskiq_broker:broker app.tasks
```

The existing Compose setup also works. `app.tasks` now registers
`answer_conversation_task`. Open `http://localhost:8000/docs` and select
**Conversations** to inspect requests and responses.

`CONVERSATION_MODEL` selects the answer model, defaulting to
`deepseek-v4-flash`. Existing provider keys and timeouts still apply. The pipeline
uses the existing defaults: 2000 history tokens, 6 retrieved chunks, a 16000-token
context allowance, and 2000 reserved answer tokens. Rewriting records its model
separately.

All paths below have the `/api/v1` prefix. Use the existing access-cookie or
`Authorization: Bearer <access_token>` authentication. Ownership checks cover
projects, conversations, messages, and runs; foreign IDs return 404. Never send
`user_id` in a request body.

## Endpoints

| Method | Path | Request / result |
| --- | --- | --- |
| POST | `/conversations` | `{project_id, title?}` → conversation, 201 |
| GET | `/conversations?project_id=UUID` | Project conversations, newest updated first |
| GET | `/conversations/{id}` | One conversation |
| PATCH | `/conversations/{id}` | `{title}` → renamed conversation |
| DELETE | `/conversations/{id}` | Deletes the conversation, messages, and attempts |
| GET | `/conversations/{id}/messages` | Chronological messages with latest assistant attempts |
| POST | `/conversations/{id}/messages` | `{content, client_request_id}` → reserved turn, 202 |
| GET | `/conversations/{id}/runs` | All attempts, newest first |
| GET | `/conversation-runs/{id}` | Lightweight polling state, answer, and sources |
| GET | `/conversation-runs/{id}/sources` | Captured source metadata and excerpts |
| GET | `/conversation-runs/{id}/debug` | Owner-only prompts, budgets, and metrics |
| POST | `/conversation-runs/{id}/cancel` | Cancel an active attempt |
| POST | `/conversation-runs/{id}/retry` | Queue a new attempt for the latest failed/cancelled turn, 202 |

Conversation and run lists support `limit` (1–200, default 50) and `offset`
(nonnegative, default 0). Message history uses `limit` and `after_position`
(nonnegative, default 0). IDs are UUIDs; timestamps use ISO 8601.

Titles must have 1–255 non-whitespace characters and are trimmed by the service.
Messages must have 1–20000 characters and cannot be whitespace-only. Their original
text is preserved. Unknown request fields are rejected. A question can still
exceed the complete token budget; that attempt fails with `context_failed`
rather than truncating the question.

Successful responses use the existing envelope:

```json
{"message": "Conversation retrieved.", "data": {}, "success": true}
```

`data` contains the endpoint's resource, list, or page; deletion returns null.
Application errors use the same envelope with `success=false` and null data.
FastAPI request-validation errors retain its standard 422 `detail` list.

## Frontend flow

1. Create a conversation for the selected project, or load an existing one.
2. Load message pages. Empty assistant messages can represent pending, failed,
   or cancelled attempts; inspect their `latest_run`.
3. Generate one `crypto.randomUUID()` for the intended user message. Keep it
   across network retries. POST the original text and UUID.
4. Add the returned `user_message` and `assistant_message` using server IDs.
   When `reused=true`, reconcile these IDs instead of appending duplicate rows.
5. Poll `data.run.id`, for example once per second while pending or running.
6. On completion, update the assistant identified by `assistant_message_id`
   using `response_text`, and display `sources`. On failure or cancellation,
   leave the placeholder empty and show the state with an eligible retry action.

Illustrative calls using an API helper that prefixes `/api/v1`, sends JSON and
existing auth credentials, and parses envelopes:

```javascript
const created = await api("/conversations", {
  method: "POST",
  body: JSON.stringify({ project_id: projectId, title: "Book discussion" }),
});

const requestId = crypto.randomUUID(); // Reuse if this POST must be retried.
const submitted = await api(`/conversations/${created.data.id}/messages`, {
  method: "POST",
  body: JSON.stringify({ content: originalText, client_request_id: requestId }),
});

const runId = submitted.data.run.id;
const polled = await api(`/conversation-runs/${runId}`);
// Inspect polled.data.status before polling again.
```

These are examples, not frontend changes. A submission returns
`{user_message, assistant_message, run, reused}`. It is a reservation, not a
generated answer. A fast worker may already have finished when polling starts.

## Reloading and pagination

Message pages contain `{messages, next_after_position}`. Continue from the
cursor until it is null. A full final page can produce one last empty page.
Assistant messages include their latest run, preserving retry/error states after
reload; user messages have null `latest_run`.

Positions do not change when a placeholder is filled. Reading only new positions
will not refresh an existing assistant message: update it from the run response
or reload that message page.

Run responses include `id`, `conversation_id`, `user_message_id`,
`assistant_message_id`, `attempt_number`, `status`, `stage`, `model`,
`rewritten_query`, `rewrite_model`, `response_text`, `error_code`, timestamps,
and `sources`. Full prompts are available through `/debug`, keeping ordinary
polling responses small.

## Status, cancellation, and retry

| Status | Frontend behavior |
| --- | --- |
| `pending` | Queued; keep polling. |
| `running` | Processing; keep polling. |
| `completed` | Render the answer and sources; stop polling. |
| `failed` | Show failure and eligible retry; stop polling. |
| `cancelled` | Show cancelled state; stop polling. |

Stages normally progress through `reserved`, `starting`, `rewritten`,
`retrieving`, `retrieved`, `context_ready`, `generating`, then a terminal stage.
Use status for behavior; stage labels are optional progress text.

Only one active attempt is allowed per conversation. Sending a new turn while
one is active returns 409. Cancel first or wait. Cancellation is idempotent for
an already-cancelled attempt; cancelling a completed/failed attempt returns 409.
Reload its final state if completion wins the race.

Cancellation prevents late publication. It does not forcibly abort an in-flight
provider HTTP request. Deleting the conversation also prevents late publication.

Retry has no body. It returns a new run ID with an incremented attempt number
and reuses the original user message and assistant slot. Only the latest attempt
of the latest turn can be retried, and it must be failed or cancelled. Submitting
a new user turn makes older turns ineligible. Poll the new ID and replace that
assistant's `latest_run` in local state.

## Idempotency, queue errors, and safe failures

Resending the same request UUID and exact text returns the latest existing run
without queuing another answer. The same UUID with changed text returns 409.
A new UUID means a new turn, not a retry of the original one.

Reservations and retries commit before Taskiq dispatch. If dispatch fails, the
API returns 503. A still-pending attempt is failed with `queue_unavailable`;
an attempt already claimed by a worker is not overwritten. If a message POST
gets 503 or loses its response, repeat it with the same UUID and text to recover
the existing run. Use its retry endpoint if it failed. If a retry POST loses its
response, reload history to find the latest run before trying again.

Safe failed-run codes include `queue_unavailable`, `rewrite_failed`,
`retrieval_failed`, `context_failed`, `answer_generation_failed`, `answer_empty`,
`answer_truncated`, `answer_filtered`, and `answer_unusable`. Map these to helpful
UI messages. Raw provider/account exceptions are not persisted on the run.

## Sources and diagnostics

Source labels such as `S1` and `S3` are retained; skipped passages are not
renumbered. Sources contain book title, document/chunk/generation IDs, chunk
index, optional page indices, distance, and an excerpt of at most 600 characters.
PDF page indices are zero-based: add 1 for display and omit unknown pages.
Use the answer run's captured sources rather than a fresh search.

The debug endpoint adds `snapshot` and `metrics`, including exact rewrite/answer
prompts, selection flags, estimated and reported token counts, answer allowance,
and fallback status. It has the same ownership checks as polling. Keep full debug
prompts out of the normal sources panel.

## Implementation and verification

`ConversationPipeline` claims a pending attempt once and calls the history/query
rewriter, retrieval, context builder, and generator. Stage checkpoints commit
separately. Duplicate task delivery cannot restart a claimed attempt. Failures
use safe codes; cancellation and deletion take precedence over late writes.
Model work never runs in an HTTP route.

Tests cover routes and validation, auth/ownership, durable messages and sources,
idempotency, queue failure/recovery, follow-ups, cancellation, retries, and the
complete HTTP-to-pipeline-to-database flow. Model, embedding, and dispatch calls
are controlled fakes, with no paid inference. PostgreSQL/pgvector integration
tests run in disposable schemas when `CHAT_TEST_DATABASE_URL` is supplied.

```powershell
uv run pytest -q
```

This completes the first conversation backend through the API. Streaming
transport, automatic memory summaries, retrieval tuning, and automatic crash
recovery remain separate improvements. After a worker crash, the current API
lets the user cancel an active attempt and retry it.
