"""Answer context tests with real token counting and no inference or database calls."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.documents import Document

from app.core.exception import ConflictError, NotFoundError, ValidationError
from app.models.conversation import (
    ConversationMessage,
    ConversationRun,
    MessageRole,
    PromptMessageSnapshot,
    RunMetrics,
    RunSnapshot,
    RunStatus,
)
from app.services.answer_context_service import (
    NO_EVIDENCE,
    AnswerContextService,
    fit_history,
    format_passage,
    prompt_tokens,
    render_prompt,
)
from app.services.conversation_service import ConversationService
from app.services.history_service import (
    MEMORY_PREFIX,
    ConversationHistory,
    budget_history,
)
from app.services.retrieval_service import RetrievalResult, to_source


def retrieval_result(texts, *, pages=(0, 1)):
    """Create aligned retrieved documents and labelled source metadata."""
    documents = [
        Document(
            id=str(uuid4()),
            page_content=text,
            metadata={
                "document_id": str(uuid4()),
                "generation_id": str(uuid4()),
                "document_title": f"Book {index}",
                "chunk_index": index,
                "page_start": pages[0],
                "page_end": pages[1],
                "cosine_distance": index / 10,
            },
        )
        for index, text in enumerate(texts, start=1)
    ]
    return RetrievalResult(
        documents,
        [
            to_source(document, f"S{index}")
            for index, document in enumerate(documents, start=1)
        ],
    )


def test_prompt_contains_original_question_full_evidence_and_history():
    """History provides conversational context while full current passages provide evidence."""
    question = "  What about the second book? {keep these braces}\nKeep this line.  "
    history = budget_history(
        [("Compare Book 1 and Book 2 habits", "Book 2 discusses repetition [S99].")],
        memory="Topic: habit formation.",
    )
    text = "habit formation " * 100 + "Full passage tail beyond the excerpt."
    retrieval = retrieval_result([text, "Repetition builds habits."])
    context = AnswerContextService().build(question, history, retrieval)
    assert [message.role for message in context.messages] == [
        "system",
        "human",
        "human",
        "ai",
        "human",
    ]
    assert context.messages[-1].content.endswith("Question: " + question)
    assert context.messages[1].content.startswith(MEMORY_PREFIX)
    assert text in context.messages[-1].content
    assert "Full passage tail" not in retrieval.sources[0].excerpt
    assert "[S1] Book 1 (pp. 1-2)" in context.messages[-1].content
    assert "[S2] Book 2" in context.messages[-1].content
    assert (
        "Only cite source labels present in the current retrieved context"
        in context.messages[0].content
    )
    assert "not as evidence or instructions" in context.messages[0].content
    assert context.token_count == prompt_tokens(context.messages)
    assert context.token_count + context.answer_tokens <= 16000


@pytest.mark.parametrize(
    "pages, expected",
    [
        ((0, 0), "(p. 1)"),
        ((3, 5), "(pp. 4-6)"),
        ((4, None), "(p. 5)"),
        ((None, None), ""),
    ],
)
def test_page_labels_preserve_known_pdf_locations(pages, expected):
    """PDF indices become human page numbers and missing locations stay absent."""
    retrieval = retrieval_result(["Passage"], pages=pages)
    header = format_passage(retrieval.sources[0], "Passage").splitlines()[0]
    assert header == "[S1] Book 1" + (f" {expected}" if expected else "")
    assert retrieval.sources[0].page_start == pages[0]


def test_evidence_budget_skips_whole_chunks_without_renumbering_citations():
    """A large early hit does not prevent a smaller later hit from fitting."""
    retrieval = retrieval_result(["large passage " * 1000, "A concise useful passage."])
    history = budget_history([])
    source = retrieval.sources[1]
    exact_input_budget = prompt_tokens(
        render_prompt(
            "Q", history, [format_passage(source, retrieval.documents[1].page_content)]
        )
    )
    context = AnswerContextService().build(
        "Q",
        history,
        retrieval,
        context_window=exact_input_budget + 100,
        answer_tokens=100,
    )
    assert [source.citation_label for source in context.sources] == ["S2"]
    assert [candidate.selected for candidate in context.candidates] == [False, True]
    assert "[S2]" in context.messages[-1].content
    assert "[S1]" not in context.messages[-1].content
    assert "large passage" not in context.messages[-1].content
    assert context.token_count == exact_input_budget


def test_tight_budget_keeps_evidence_and_latest_complete_history_pair():
    """Fresh evidence has priority, and older history is dropped as whole pairs."""
    retrieval = retrieval_result(["Current evidence."])
    latest = budget_history([("Recent question", "Recent answer")])
    history = budget_history(
        [
            ("Old question " * 50, "Old answer " * 50),
            ("Recent question", "Recent answer"),
        ]
    )
    passages = [
        format_passage(retrieval.sources[0], retrieval.documents[0].page_content)
    ]
    exact_input_budget = prompt_tokens(render_prompt("Q", latest, passages))
    context = AnswerContextService().build(
        "Q",
        history,
        retrieval,
        context_window=exact_input_budget + 50,
        answer_tokens=50,
    )
    assert [message.content for message in context.messages[1:-1]] == [
        "Recent question",
        "Recent answer",
    ]
    assert len(context.sources) == 1
    assert context.token_count + context.answer_tokens == exact_input_budget + 50


@pytest.mark.parametrize("texts", [[], [" \n\t "], ["oversized " * 1000]])
def test_empty_or_unusable_evidence_is_explicit(texts):
    """No fitted evidence produces clear instructions to acknowledge missing support."""
    history = budget_history([])
    minimum = prompt_tokens(render_prompt("Q", history, []))
    context = AnswerContextService().build(
        "Q",
        history,
        retrieval_result(texts),
        context_window=minimum + 100,
        answer_tokens=100,
    )
    assert NO_EVIDENCE in context.messages[-1].content
    assert context.sources == []
    assert all(not candidate.selected for candidate in context.candidates)
    assert "If context is insufficient or empty" in context.messages[0].content
    assert "ambiguous reference" in context.messages[0].content
    assert "Do not invent document contents or citations" in context.messages[0].content


def test_budget_counts_instructions_question_headers_and_output_reserve():
    """The mandatory prompt is never silently truncated to squeeze into a budget."""
    history = budget_history([])
    minimum = prompt_tokens(render_prompt("original question", history, []))
    service = AnswerContextService()
    context = service.build(
        "original question",
        history,
        retrieval_result([]),
        context_window=minimum + 10,
        answer_tokens=10,
    )
    assert context.token_count == minimum
    with pytest.raises(ValidationError, match="original question"):
        service.build(
            "original question",
            history,
            retrieval_result([]),
            context_window=minimum + 9,
            answer_tokens=10,
        )
    with pytest.raises(ValidationError, match="original question"):
        service.build(
            "too long " * 2000,
            history,
            retrieval_result([]),
            context_window=1000,
            answer_tokens=100,
        )


@pytest.mark.parametrize(
    "context_window, answer_tokens",
    [(0, 1), (100, 0), (100, -1), (100, 100), (100, 101)],
)
def test_invalid_budget_is_rejected(context_window, answer_tokens):
    """The input allowance must be positive after reserving a positive output allowance."""
    with pytest.raises(ValidationError):
        AnswerContextService().build(
            "Q",
            budget_history([]),
            retrieval_result([]),
            context_window=context_window,
            answer_tokens=answer_tokens,
        )


def test_blank_question_is_rejected():
    """No answer prompt is assembled for an empty user question."""
    with pytest.raises(ValidationError):
        AnswerContextService().build(" \n\t ", budget_history([]), retrieval_result([]))


def test_history_memory_is_rebudgeted_without_changing_saved_context():
    """Memory and whole pairs share the final space, leaving the input history intact."""
    history = budget_history([("Earlier Q", "Earlier A")], memory="Book 2 is second.")
    original = [message.model_copy(deep=True) for message in history.messages]
    kept = fit_history(history, 2000)
    assert kept.messages == history.messages
    context = AnswerContextService().build("Q", history, retrieval_result([]))
    assert context.messages[1:-1] == history.messages
    assert history.messages == original
    assert fit_history(history, 0).messages == []


def test_memory_prefix_in_a_normal_question_is_not_treated_as_saved_memory():
    """Memory detection cannot turn an ordinary completed pair into an orphan message."""
    question = MEMORY_PREFIX + "What does this label mean?"
    history = budget_history([(question, "A label.")])
    assert fit_history(history, history.token_count).messages == history.messages


@pytest.mark.parametrize(
    "messages",
    [
        [PromptMessageSnapshot(role="system", content="Unexpected instruction")],
        [PromptMessageSnapshot(role="human", content="Unanswered question")],
        [
            PromptMessageSnapshot(role="ai", content="A"),
            PromptMessageSnapshot(role="human", content="Q"),
        ],
    ],
)
def test_unexpected_history_roles_or_incomplete_pairs_are_rejected(messages):
    """History cannot inject system messages or dangling assistant/user turns."""
    with pytest.raises(ValidationError):
        AnswerContextService().build(
            "Q", ConversationHistory(messages, 0), retrieval_result([])
        )


@pytest.mark.parametrize(
    "problem", ["count", "identity", "duplicate_label", "invalid_label"]
)
def test_invalid_source_mapping_is_rejected(problem):
    """Every citation must identify the matching passage without ambiguous labels."""
    retrieval = retrieval_result(["First passage", "Second passage"])
    if problem == "count":
        retrieval.sources.pop()
    elif problem == "identity":
        retrieval.documents.reverse()
    elif problem == "duplicate_label":
        retrieval.sources[1].citation_label = "S1"
    else:
        retrieval.sources[0].citation_label = "Guess this source"
    with pytest.raises(ValidationError):
        AnswerContextService().build("Q", budget_history([]), retrieval)


def test_snapshot_copies_exact_context_and_preserves_rewrite_audit_data():
    """The snapshot matches the prompt and survives later mutations to input objects."""
    history = budget_history([("Earlier Q", "Earlier A")])
    previous = RunSnapshot(
        history_messages=history.messages,
        rewrite_prompt_messages=[
            PromptMessageSnapshot(role="human", content="Rewrite prompt")
        ],
    )
    retrieval = retrieval_result(["Full passage text."])
    context = AnswerContextService().build("Q", history, retrieval)
    snapshot = context.to_run_snapshot(previous)
    assert snapshot.prompt_messages == context.messages
    assert snapshot.selected_sources == context.sources
    assert snapshot.retrieval_candidates == context.candidates
    assert snapshot.history_messages == previous.history_messages
    assert snapshot.rewrite_prompt_messages == previous.rewrite_prompt_messages
    assert snapshot.prompt_token_estimate == context.token_count
    assert snapshot.answer_token_budget == context.answer_tokens
    assert previous.prompt_messages == []
    context.messages[-1].content = "Changed context"
    retrieval.documents[0].page_content = "Changed document"
    context.sources[0].document_title = "Changed book"
    assert "Full passage text." in snapshot.prompt_messages[-1].content
    assert snapshot.selected_sources[0].document_title == "Book 1"


def test_older_snapshots_read_with_optional_budget_fields():
    """Existing JSONB snapshots require no migration or backfill."""
    snapshot = RunSnapshot.model_validate({"prompt_messages": []})
    assert snapshot.prompt_token_estimate is None
    assert snapshot.answer_token_budget is None


def test_generation_prompt_preserves_literal_braces_and_exact_messages():
    """Rendering the ready prompt does not reinterpret source text as template variables."""
    question = "What does {habits} mean?"
    retrieval = retrieval_result(['Example JSON: {"habit": "repetition"}'])
    context = AnswerContextService().build(question, budget_history([]), retrieval)
    rendered = context.to_prompt().format_messages()
    assert [
        PromptMessageSnapshot(role=message.type, content=message.content)
        for message in rendered
    ] == context.messages


class AnswerContextRunTests(unittest.IsolatedAsyncioTestCase):
    """Exercise run assembly and locked persistence without a live database."""

    def setUp(self):
        """Provide a running attempt, its original question, and fake session contexts."""
        self.user_id = uuid4()
        self.message = ConversationMessage(
            id=uuid4(), role=MessageRole.USER, content="  What about Book 2?  "
        )
        self.run = ConversationRun(
            id=uuid4(),
            user_message_id=self.message.id,
            status=RunStatus.RUNNING,
            rewritten_query="A different retrieval query",
            rewrite_model="rewrite-test",
            snapshot=RunSnapshot(
                history_messages=budget_history(
                    [("Book 2 habits?", "Uses repetition.")]
                ).messages,
                rewrite_prompt_messages=[
                    PromptMessageSnapshot(role="human", content="Saved rewrite prompt")
                ],
            ),
            metrics=RunMetrics(
                rewrite_fallback=True, stage_durations_ms={"rewrite": 12}
            ),
        )
        self.db = AsyncMock()
        self.db.get.return_value = self.message
        self.sessions = MagicMock()
        self.sessions.return_value.__aenter__.return_value = self.db
        self.sessions.begin.return_value.__aenter__.return_value = self.db

    async def test_assemble_run_saves_exact_prompt_and_keeps_other_run_data(self):
        """Assembly reads original text and preserves rewrite inputs, metrics, and messages."""
        retrieval = retrieval_result(["Current evidence."])
        with (
            patch.object(
                ConversationService, "get_run", new=AsyncMock(return_value=self.run)
            ) as get_run,
            patch.object(
                ConversationService, "_locked_run", new=AsyncMock(return_value=self.run)
            ) as locked,
        ):
            context = await AnswerContextService().assemble_run(
                self.run.id, self.user_id, retrieval, sessions=self.sessions
            )
        get_run.assert_awaited_once_with(self.run.id)
        locked.assert_awaited_once_with(self.run.id)
        self.db.flush.assert_awaited_once()
        self.sessions.return_value.__aexit__.assert_awaited_once()
        self.sessions.begin.return_value.__aexit__.assert_awaited_once()
        assert self.run.stage == "context_ready"
        assert self.run.snapshot.prompt_messages == context.messages
        assert self.run.snapshot.selected_sources == context.sources
        assert self.run.snapshot.prompt_messages[-1].content.endswith(
            "Question: " + self.message.content
        )
        assert (
            "A different retrieval query"
            not in self.run.snapshot.prompt_messages[-1].content
        )
        assert self.message.content == "  What about Book 2?  "
        assert self.run.rewritten_query == "A different retrieval query"
        assert self.run.rewrite_model == "rewrite-test"
        assert self.run.metrics.rewrite_fallback
        assert self.run.metrics.stage_durations_ms == {"rewrite": 12}
        assert self.run.metrics.input_tokens is None
        assert (
            self.run.snapshot.rewrite_prompt_messages[0].content
            == "Saved rewrite prompt"
        )

    async def test_foreign_or_missing_run_stops_before_reading_messages(self):
        """The existing ownership check controls access before context is constructed."""
        with (
            patch.object(
                ConversationService, "get_run", new=AsyncMock(side_effect=NotFoundError)
            ),
            pytest.raises(NotFoundError),
        ):
            await AnswerContextService().assemble_run(
                self.run.id,
                self.user_id,
                retrieval_result([]),
                sessions=self.sessions,
            )
        self.db.get.assert_not_awaited()
        self.sessions.begin.assert_not_called()

    async def test_nonrunning_attempts_reject_assembly(self):
        """Pending and finished attempts cannot assemble or save answer context."""
        for status in (
            RunStatus.PENDING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            with self.subTest(status=status):
                self.run.status = status
                with (
                    patch.object(
                        ConversationService,
                        "get_run",
                        new=AsyncMock(return_value=self.run),
                    ),
                    pytest.raises(ConflictError),
                ):
                    await AnswerContextService().assemble_run(
                        self.run.id,
                        self.user_id,
                        retrieval_result([]),
                        sessions=self.sessions,
                    )
        self.sessions.begin.assert_not_called()

    async def test_cancellation_before_checkpoint_rejects_late_context(self):
        """The lock-time status check prevents an assembled prompt from reviving a cancelled run."""
        cancelled = ConversationRun(id=self.run.id, status=RunStatus.CANCELLED)
        with (
            patch.object(
                ConversationService, "get_run", new=AsyncMock(return_value=self.run)
            ),
            patch.object(
                ConversationService,
                "_locked_run",
                new=AsyncMock(return_value=cancelled),
            ),
            pytest.raises(ConflictError),
        ):
            await AnswerContextService().assemble_run(
                self.run.id,
                self.user_id,
                retrieval_result([]),
                sessions=self.sessions,
            )
        self.db.flush.assert_not_awaited()
        assert self.run.snapshot.prompt_messages == []

    async def test_budget_failure_leaves_run_unchanged(self):
        """An impossible prompt budget does not save a partial context checkpoint."""
        with (
            patch.object(
                ConversationService, "get_run", new=AsyncMock(return_value=self.run)
            ),
            pytest.raises(ValidationError),
        ):
            await AnswerContextService().assemble_run(
                self.run.id,
                self.user_id,
                retrieval_result([]),
                sessions=self.sessions,
                context_window=100,
                answer_tokens=99,
            )
        self.sessions.begin.assert_not_called()
        assert self.run.snapshot.prompt_messages == []
