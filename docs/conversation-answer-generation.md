# Answer generation

`AnswerGenerationService` generates a complete answer from the exact prompt saved
by [answer context assembly](conversation-answer-context.md). It publishes the
answer to the assistant message only after generation succeeds. New functions
have docstrings; no migration or new dependency is needed.

## Usage

```python
from app.services.answer_generation_service import AnswerGenerationService

# run_id must identify a running, context_ready attempt owned by user_id.
# AnswerContextService.assemble_run has already committed the answer context.
answer = await AnswerGenerationService().generate_run(run_id, user_id)
```

The method returns the generated text after committing it to the database.
By default it uses the model recorded on the run by `start_run`. For tests or
an explicit model override, inject an `LLMService`; the actual selected model
is recorded on the run before inference.

```python
from app.services.llm.llm_service import LLMService

generator = AnswerGenerationService(LLMService(model=answer_model))
answer = await generator.generate_run(run_id, user_id)
```

Tests can supply a different session factory through the `sessions` argument,
as in the history, rewriting, and context services.

## Lifecycle and persistence

Generation has three steps:

1. Claim the run under the existing conversation lock. Ownership is checked,
   status must be `RUNNING`, and stage must be `context_ready`. A saved prompt,
   a saved answer-token allowance, and an answer model are required. The stage
   changes to `generating`, and this transaction commits before inference.
2. Call the existing LLM wrapper without an open database transaction. It uses
   the saved ordered messages as literal text, preserving braces, evidence,
   citations, history, and the original question. It does not rerun retrieval or
   rebuild context from current documents.
3. On success, the existing locked `complete` operation atomically sets
   `run.response_text` and the visible assistant message, records metrics, marks
   the attempt `COMPLETED`, and updates its finish time and conversation timestamp.

A second generation call cannot claim a run that is already generating or
finished. Once claimed, checkpoints cannot replace its prompt/rewrite inputs or
reset it to an earlier stage. Metrics and partial-response checkpoints at the
`generating` stage remain supported for future streaming work.

The user message, saved prompt and source selection, rewrite query/model, fallback
flag, and earlier stage metrics are preserved. This implementation waits for a
complete response; it does not stream text into the assistant placeholder.

## Output allowance and metadata

`LLMService.generate_response` is a small addition that returns the complete
LangChain message, including provider metadata. Existing `generate` callers
still receive a string, and the existing streaming wrapper is unchanged.

Generation passes `snapshot.answer_token_budget` as DeepSeek's `max_tokens`
request field and tags the request with the owner's UUID as `user_id`. These
fields are documented in the
[DeepSeek Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion/).
They go through `extra_body` so the installed ChatOpenAI client does not rename
`max_tokens` to `max_completion_tokens`. A local mock-HTTP test verifies the
actual serialized request body.

Available provider-reported metadata is stored in the existing run metrics:

- `input_tokens` and `output_tokens` from the returned usage metadata.
- `finish_reason` from the response metadata.
- `provider_request_id` only when an explicit `request_id` is exposed.

Missing values stay null. The response ID and LangChain-generated message ID are
not substituted for a provider request ID. Context token estimates remain in
the snapshot, separate from actual provider usage.

## Failures and cancellation

The visible assistant message stays empty for a failed attempt. The service
records an application-defined error code and raises `AnswerGenerationError`,
an `AppError` with status 502 and a generic retry message. Provider exception
text, raw response objects, and reasoning content are not stored or exposed.

| Error code | Meaning |
| --- | --- |
| `answer_generation_failed` | Provider call or response processing failed. |
| `answer_empty` | The response contains no non-whitespace answer text. |
| `answer_truncated` | The provider reports `finish_reason="length"`. |
| `answer_filtered` | The provider reports content filtering. |
| `answer_unusable` | Unexpected finish reason, refusal metadata, or tool calls. |

A normal `stop` response is accepted. A nonempty response without a finish reason
is also accepted for adapters that omit this metadata. The answer's Markdown,
citations, and whitespace are preserved. Citation grounding is instructed by the
assembled prompt; this step does not add a semantic evidence checker or a second
model call.

When response metadata is available, rejected output still records its reported
usage and finish reason. Truncated text is not saved as a completed response.
Retries use the existing failed-run retry operation, which creates a new attempt
and reuses the same assistant slot. Assemble context for that new attempt before
calling generation again. The generation service adds no automatic retry loop;
the existing provider SDK retry settings still apply.

Cancellation of the generation coroutine marks the attempt cancelled and
re-raises `CancelledError`. A concurrent user cancellation, completion, or deletion
takes precedence: late results cannot overwrite it. Claims and final writes use
separate transactions, so no database lock is held while waiting for the model.
Database errors during persistence propagate; a process crash or unavailable
database can still leave a claimed attempt for later recovery.

## Verification and remaining work

Run `uv run pytest -q`. Unit tests cover exact prompt reuse, output-token limits,
serialized provider parameters, usage capture, unchanged user text and snapshots,
successful publication, safe provider errors, empty/truncated output, duplicate
claims, immutable claimed input, and cancellation. Model and HTTP calls are local
fakes; no paid inference is used.

Additional PostgreSQL tests verify durable answer/usage storage, ownership,
failure and retry, concurrent claims, and cancellation during inference. Set
`CHAT_TEST_DATABASE_URL` to an asyncpg PostgreSQL URL to run them in disposable
schemas. They are skipped when that variable is absent.

`ConversationPipeline` now connects these stages through a Taskiq worker and
authenticated [conversation API](conversation-api.md). The frontend can use
polling, cancellation, retries, and captured sources. Streaming transport,
frontend integration, and automatic recovery of crashed attempts remain separate
work.
