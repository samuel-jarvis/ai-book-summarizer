"""Build and checkpoint a budgeted answer prompt without calling the model."""

import re
from dataclasses import dataclass
from uuid import UUID

from langchain_core.messages import convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import AsyncSessionLocal
from app.core.exception import ConflictError, ValidationError
from app.models.conversation import (
    CandidateSnapshot,
    ConversationMessage,
    PromptMessageSnapshot,
    RunSnapshot,
    RunStatus,
    SourceSnapshot,
)
from app.services.conversation_service import ConversationService
from app.services.history_service import (
    MEMORY_PREFIX,
    ConversationHistory,
    budget_history,
    message_tokens,
)
from app.services.llm.prompts import PROJECT_CONVERSATION_PROMPT
from app.services.retrieval_service import RetrievalResult

NO_EVIDENCE = "No retrieved evidence is available for this answer."


def format_passage(source: SourceSnapshot, text: str) -> str:
    """Label a full passage, displaying stored zero-based PDF pages as one-based."""
    header = f"[{source.citation_label}] {source.document_title}"
    if source.page_start is not None and source.page_start >= 0:
        start = source.page_start + 1
        if source.page_end is not None and source.page_end > source.page_start:
            header += f" (pp. {start}-{source.page_end + 1})"
        else:
            header += f" (p. {start})"
    return f"{header}\n<passage>\n{text}\n</passage>"


def render_prompt(
    question: str, history: ConversationHistory, passages: list[str]
) -> list[PromptMessageSnapshot]:
    """Render instructions, chronological history, evidence, and the exact question."""
    messages = PROJECT_CONVERSATION_PROMPT.format_messages(
        history=[(message.role, message.content) for message in history.messages],
        context="\n\n".join(passages) if passages else NO_EVIDENCE,
        question=question,
    )
    return [
        PromptMessageSnapshot(role=message.type, content=message.content)
        for message in messages
    ]


def prompt_tokens(messages: list[PromptMessageSnapshot]) -> int:
    """Estimate the complete input, including message framing and reply priming."""
    return sum(message_tokens(message) for message in messages) + 3


def fit_history(history: ConversationHistory, token_budget: int) -> ConversationHistory:
    """Reuse whole-pair history budgeting and reject unexpected message roles."""
    messages = history.messages
    memory = None
    if (
        len(messages) % 2
        and messages[0].role == "human"
        and messages[0].content.startswith(MEMORY_PREFIX)
    ):
        memory = messages[0].content[len(MEMORY_PREFIX) :]
        messages = messages[1:]
    if len(messages) % 2 or any(
        message.role != ("human" if index % 2 == 0 else "ai")
        for index, message in enumerate(messages)
    ):
        raise ValidationError(
            "Answer history must contain completed user/assistant pairs."
        )
    exchanges = [
        (messages[index].content, messages[index + 1].content)
        for index in range(0, len(messages), 2)
    ]
    return budget_history(exchanges, memory=memory, token_budget=token_budget)


@dataclass(frozen=True)
class AnswerContext:
    """Exact rendered input, citation selection, and the reserved output allowance."""

    messages: list[PromptMessageSnapshot]
    sources: list[SourceSnapshot]
    candidates: list[CandidateSnapshot]
    token_count: int
    answer_tokens: int

    def to_prompt(self) -> ChatPromptTemplate:
        """Return literal messages so braces in questions or passages stay unchanged."""
        return ChatPromptTemplate.from_messages(
            convert_to_messages(
                [(message.role, message.content) for message in self.messages]
            )
        )

    def to_run_snapshot(self, previous: RunSnapshot | None = None) -> RunSnapshot:
        """Capture the exact prompt and selection while preserving rewrite inputs."""
        snapshot = previous.model_copy(deep=True) if previous else RunSnapshot()
        snapshot.prompt_messages = [
            message.model_copy(deep=True) for message in self.messages
        ]
        snapshot.selected_sources = [
            source.model_copy(deep=True) for source in self.sources
        ]
        snapshot.retrieval_candidates = [
            candidate.model_copy(deep=True) for candidate in self.candidates
        ]
        snapshot.prompt_token_estimate = self.token_count
        snapshot.answer_token_budget = self.answer_tokens
        return snapshot


class AnswerContextService:
    """Assemble evidence first, then fit recent history within the remaining space."""

    def build(
        self,
        question: str,
        history: ConversationHistory,
        retrieval: RetrievalResult,
        *,
        context_window: int = 16000,
        answer_tokens: int = 2000,
    ) -> AnswerContext:
        """Keep instructions and the question intact; omit whole chunks that do not fit."""
        if not question.strip():
            raise ValidationError("Question must contain non-whitespace text.")
        if answer_tokens < 1 or context_window <= answer_tokens:
            raise ValidationError(
                "Context window must exceed a positive answer token allowance."
            )
        limit = context_window - answer_tokens
        empty_history = ConversationHistory([], 0)
        messages = render_prompt(question, empty_history, [])
        if prompt_tokens(messages) > limit:
            raise ValidationError(
                "Instructions and the original question exceed the prompt budget."
            )
        if len(retrieval.documents) != len(retrieval.sources):
            raise ValidationError("Retrieved documents and sources must match.")

        passages = []
        sources = []
        labels = set()
        for document, source in zip(
            retrieval.documents, retrieval.sources, strict=True
        ):
            if str(source.chunk_id) != document.id:
                raise ValidationError("A source does not match its retrieved passage.")
            if (
                not re.fullmatch(r"S[1-9][0-9]*", source.citation_label)
                or source.citation_label in labels
            ):
                raise ValidationError(
                    "Source citation labels must be unique S-number labels."
                )
            labels.add(source.citation_label)
            if not document.page_content.strip():
                continue
            passage = format_passage(source, document.page_content)
            candidate_messages = render_prompt(
                question, empty_history, [*passages, passage]
            )
            if prompt_tokens(candidate_messages) <= limit:
                passages.append(passage)
                sources.append(source.model_copy(deep=True))
                messages = candidate_messages

        remaining = limit - prompt_tokens(messages)
        kept_history = fit_history(history, remaining)
        messages = render_prompt(question, kept_history, passages)
        selected = {source.chunk_id for source in sources}
        candidates = retrieval.to_run_snapshot().retrieval_candidates
        for candidate in candidates:
            candidate.selected = candidate.chunk_id in selected
        return AnswerContext(
            messages, sources, candidates, prompt_tokens(messages), answer_tokens
        )

    async def assemble_run(
        self,
        run_id: UUID,
        user_id: UUID,
        retrieval: RetrievalResult,
        *,
        sessions: async_sessionmaker[AsyncSession] = AsyncSessionLocal,
        context_window: int = 16000,
        answer_tokens: int = 2000,
    ) -> AnswerContext:
        """Load the original message and saved history, build, and checkpoint the prompt."""
        async with sessions() as db:
            run = await ConversationService(db, user_id).get_run(run_id)
            if run.status != RunStatus.RUNNING:
                raise ConflictError(
                    "Only a running attempt can assemble answer context."
                )
            message = await db.get(ConversationMessage, run.user_message_id)
            question = message.content
            saved = run.snapshot.history_messages
            history = ConversationHistory(
                saved, sum(message_tokens(item) for item in saved)
            )
        context = self.build(
            question,
            history,
            retrieval,
            context_window=context_window,
            answer_tokens=answer_tokens,
        )
        async with sessions.begin() as db:
            await ConversationService(db, user_id).checkpoint(
                run_id, stage="context_ready", answer_context=context
            )
        return context
