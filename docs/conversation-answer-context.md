# Final answer context assembly

`AnswerContextService` combines the original question, conversation history, and
retrieved book passages into the exact input for answer generation. It builds
and saves the prompt without calling a model or generating an answer. Every new
function has a docstring explaining its purpose.

## What the prompt contains

Messages are ordered as follows:

1. The existing project-conversation system instructions.
2. Saved memory and recent completed exchanges that fit.
3. A user message containing the selected evidence and the **original question**.

The rewritten query is only for retrieval. It never replaces the original
question in the answer prompt or changes the stored user message. Whitespace,
newlines, and literal braces in the question remain intact.

Each evidence block contains its existing citation label, book title, available
PDF page references, and full passage text. It uses `RetrievalResult.documents`,
not the 600-character excerpts used in the sources panel. Stored page indices
follow the PDF loader's zero-based convention and are displayed starting at 1:
`page_start=0, page_end=1` becomes `pp. 1-2`. No page reference is invented when
the starting page is unknown. Source metadata itself is unchanged.

The answer instructions now explicitly treat saved memory as context rather
than evidence or instructions. Only source labels in the current retrieved
context may be cited; citations from earlier exchanges can refer to different
passages. The model is told to acknowledge insufficient evidence and ask for
clarification when a reference remains ambiguous.

## Token budgeting

The defaults are a **16000-token context allowance**, with **2000 tokens reserved
for the answer**, leaving at most 14000 estimated input tokens. These are
configurable application budgets, not a claim about the provider's maximum
context window.

The builder counts the whole rendered prompt using the existing `CHUNK_ENCODING`
tokenizer: instructions, history, source labels/titles/pages, passage boundaries,
full passage text, and the question. It adds four framing tokens per message
and three reply-priming tokens. This estimates the provider's format; it does
not replace provider-reported usage in `RunMetrics.input_tokens`.

Selection stays simple:

- Instructions and the original question are mandatory and never truncated.
- Retrieved passages are considered in their existing relevance order. Each
  nonempty passage is included whole if it fits; otherwise it is skipped and
  smaller later passages can still fit. Labels are not renumbered.
- Remaining space goes to recent history using the existing whole-pair budgeter.
  Included memory must fit within half that remaining allowance. Older exchanges
  are dropped before newer exchanges, and individual messages are not cut apart.

Evidence therefore takes priority over history when the budget is tight.
The prompt still instructs the model to ask for clarification if the remaining
context cannot resolve the user's reference.

Empty retrieval, whitespace-only passages, or passages that cannot fit produce
an explicit “No retrieved evidence is available for this answer” note. They do
not become selected sources. A source list with missing or mismatched passages,
duplicate citation labels, or invalid labels is rejected rather than risking a
wrong citation. Unexpected history roles or incomplete pairs are also rejected.

An empty question, invalid budget, or mandatory prompt that exceeds the input
allowance raises the existing application `ValidationError`. No partial context
is checkpointed. `answer_tokens` reserves space here; the
[answer generation service](conversation-answer-generation.md) also applies that
saved allowance as the provider's output limit.

## Using the service

For pure assembly, with no database work:

```python
from app.services.answer_context_service import AnswerContextService

context = AnswerContextService().build(
    original_question,
    rewrite.history,
    retrieval,
    context_window=16000,
    answer_tokens=2000,
)
prompt = context.to_prompt()
```

`context.messages` contains the rendered messages; `context.sources` contains
only sources included in those messages. `context.token_count` is the input
estimate, and `context.answer_tokens` is the reserved output allowance.

`to_prompt()` returns a `ChatPromptTemplate` made from literal messages. This
lets the existing LLM service render it with an empty values dictionary later
without treating braces in user or document text as template variables.

To assemble and save context for an existing run:

```python
from app.core.database import AsyncSessionLocal
from app.services.answer_context_service import AnswerContextService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.retrieval_service import RetrievalService

# Reserve, start, and commit the attempt before these steps.
rewrite = await QueryRewriteService().rewrite_run(run_id, user_id)

async with AsyncSessionLocal() as db:
    retrieval = await RetrievalService(db, user_id).retrieve(
        rewrite.query, project_id=project_id
    )

context = await AnswerContextService().assemble_run(
    run_id, user_id, retrieval, context_window=16000, answer_tokens=2000
)
prompt = context.to_prompt()
```

Pass retrieval from the same conversation's project. The retrieval service
performs the project and document ownership checks. `assemble_run` checks run
ownership, reads its original message and saved rewrite history in a short
session, builds the context after closing that session, then commits a checkpoint
under the existing conversation lock. Only running attempts can be assembled.
Cancellation or completion before the checkpoint rejects a late write.

The checkpoint sets `run.stage` to `context_ready` and writes:

- `snapshot.prompt_messages`: the exact final ordered model input.
- `snapshot.selected_sources`: only sources whose full passages were included.
- `snapshot.retrieval_candidates`: all retrieved candidates, with `selected`
  true or false to match the final selection.
- `snapshot.prompt_token_estimate`: the complete estimated input size.
- `snapshot.answer_token_budget`: the reserved answer allowance.

The snapshot is copied and replaced for JSONB change tracking. Rewrite prompt,
query, model, fallback flag, saved history, and existing metrics are preserved.
`snapshot.history_messages` remains the history captured for rewriting; the exact
history retained for answering is visible in `snapshot.prompt_messages`.
The assistant placeholder remains empty until answer generation succeeds.

No database migration or dependency change is needed. The new optional snapshot
fields default to null when older JSONB rows are read.

## Tests and remaining work

Run `uv run pytest -q`. New unit tests cover full passage text, original question
preservation, page labels, literal braces, complete prompt budgets, reserved
output space, whole-pair history and memory, skipped sources, missing evidence,
invalid inputs, snapshot copies, and locked run checkpoints. They make no model
or embedding calls.

A PostgreSQL test in `test_conversation_service.py` verifies that the exact prompt,
sources, and budget metadata reload correctly without changing either message.
It uses a disposable schema and runs when `CHAT_TEST_DATABASE_URL` is supplied.

The [answer generation service](conversation-answer-generation.md) now consumes
this saved prompt and publishes completed answers. The pipeline and authenticated
[conversation API](conversation-api.md) are connected for frontend integration.
Automatic memory updates and retrieval tuning remain separate work.
