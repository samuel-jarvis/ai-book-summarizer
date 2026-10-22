# Conversation history and query rewriting

This step gives retrieval a standalone search query while preserving the exact
user message. It adds `HistoryService`, `QueryRewriteService`, a rewrite prompt,
and tests. Every new function has a docstring explaining its purpose.

## History

`HistoryService.load` checks conversation ownership through `ConversationService`
and loads at most 20 recent completed exchanges by default. Each exchange is the
user message and its published assistant answer. Failed, cancelled, pending, and
running attempts are excluded, as are empty or whitespace-only messages.
A successful retry contributes its completed pair once. The current turn and
later turns are excluded using `before_position`.

The default history budget is 2000 tokens. `tiktoken` uses the existing
`CHUNK_ENCODING` setting and counts four extra tokens per message for framing.
This is an estimate for the provider's chat format, not a provider-reported count.
The budget covers history and memory only; rewrite instructions, the current
question, retrieved evidence, and answer output need their own context allowance.
The tokenizer may download its vocabulary on first use; prewarm its cache when
deploying without network access.

History retains whole pairs, newest first when choosing what fits, then returns
them in chronological order. If the next pair cannot fit, selection stops.
An oversized latest pair therefore does not cause stale older turns to be used.
A zero budget returns no history. `max_exchanges` can be set from 1 to 200.

Saved `Conversation.memory_summary` is included as labelled context only when
`memory_through_position` identifies an earlier point than the current turn and
the whole memory fits within half the token budget. This leaves room for recent
exchanges and avoids cutting a remembered fact into a misleading fragment.
Included memory replaces the turns through that position. If memory is absent,
too large, or has an invalid/future boundary, those earlier turns remain eligible
within the exchange limit. This step reads saved memory; it does not generate it.

## Rewriting and fallback

The existing `LLMService` handles generation with temperature zero. The prompt
asks the model to resolve references from history, retain the user's intent and
language, and keep clear topic changes independent of earlier books or topics.
For example, given a discussion of habit formation with Book A first and Book B
second, “What about the second book?” can become “How does Book B explain habit
formation?”

The model must return a JSON object with `status` and `query`. A `ready` query
must contain text, fit within 1000 characters, and contain no line breaks or
control characters. Unknown fields, malformed JSON, empty queries, and
`ambiguous` status are rejected. Ambiguous references must be marked ambiguous
rather than resolved by guessing. Semantic resolution depends on the model;
the validator checks its declared status and response format.

Unusable output or a provider exception falls back to the exact original question
and sets `rewrite_fallback=True`. This allows retrieval to proceed without an
invented interpretation. The later answer stage can use this flag when deciding
whether clarification is needed. Provider exception text is not stored.
Cancellation propagates instead of producing a fallback.

When history is empty, including a first turn or a zero history budget, rewriting
is skipped. The original question is used, the rewrite model is null, the rewrite
prompt is empty, and fallback is false. Existing saved memory can still make a
first visible follow-up eligible for rewriting.

## Run persistence and retrieval handoff

Reserve and start the run, committing those operations before calling
`rewrite_run`. It opens a short session to read the original question and history,
closes that session before inference, then commits a checkpoint under the
existing conversation lock. A run cancelled or failed during inference rejects
the late checkpoint with `ConflictError`.

The checkpoint writes:

- `run.rewritten_query`: the search query, including the raw question on fallback.
- `run.rewrite_model`: the attempted rewrite model, or null when skipped.
- `run.snapshot.rewrite_prompt_messages`: the exact rendered model input.
- `run.snapshot.history_messages`: the budgeted memory and completed exchanges.
- `run.metrics.rewrite_fallback`: whether a rewrite was attempted but unusable.
- `run.stage`: `rewritten`.

Snapshot and metrics objects are copied and replaced so JSONB changes persist.
Other snapshot fields and metrics survive the checkpoint. The user message is
never assigned the rewritten query. Existing JSONB rows read the added
`history_messages` field as an empty list; no database migration is needed.

```python
from app.core.database import AsyncSessionLocal
from app.services.conversation_service import ConversationService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.retrieval_service import RetrievalService

# run_id refers to a reserved, started, committed attempt owned by user_id.
rewrite = await QueryRewriteService().rewrite_run(
    run_id, user_id, token_budget=2000, max_exchanges=20
)

async with AsyncSessionLocal() as db:
    evidence = await RetrievalService(db, user_id).retrieve(
        rewrite.query, project_id=project_id
    )

async with AsyncSessionLocal.begin() as db:
    service = ConversationService(db, user_id)
    run = await service.get_run(run_id)
    await service.checkpoint(
        run_id, stage="retrieved", snapshot=evidence.to_run_snapshot(run.snapshot)
    )
```

These are service-level building blocks; the conversation HTTP routes and worker
orchestration are not yet wired. The next step can assemble answer context from
`rewrite.history.messages`, the original message, and `evidence.documents`.
This change does not assemble or generate the final answer.

## Verification

```powershell
uv run pytest -q
```

Unit tests cover token limits, whole pairs, memory size and boundaries, empty
messages, rendered follow-up and topic-change prompts, ambiguous references,
output validation, provider errors, skipped rewrites, and cancellation. The LLM
tests use controlled fake responses; they do not measure a live model's rewrite
quality or make paid provider calls.

PostgreSQL integration tests in `test_conversation_service.py` also cover history
filtering, successful retries, owner checks, memory boundaries, row limits,
rewrite persistence, unchanged user content, and cancellation during inference.
Set `CHAT_TEST_DATABASE_URL` to an asyncpg PostgreSQL URL to run them. They use
disposable schemas; retrieval tests additionally need the pgvector extension.
Tests requiring PostgreSQL are skipped when that variable is absent.
