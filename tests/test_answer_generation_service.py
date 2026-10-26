"""Generation lifecycle and provider payload tests without network or inference calls."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.core.exception import ConflictError, NotFoundError, ValidationError
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    MessageRole,
    PromptMessageSnapshot,
    RunMetrics,
    RunSnapshot,
    RunStatus,
)
from app.services.answer_generation_service import (
    AnswerGenerationError,
    AnswerGenerationService,
    answer_text,
)
from app.services.conversation_service import ConversationService
from app.services.llm.llm_service import LLMService


@pytest.mark.parametrize(
    "reason, text, code",
    [
        ("length", "Partial answer", "answer_truncated"),
        ("content_filter", "Filtered answer", "answer_filtered"),
        ("tool_calls", "Unexpected tools", "answer_unusable"),
        ("insufficient_system_resource", "Partial answer", "answer_unusable"),
        ("stop", " \n\t ", "answer_empty"),
    ],
)
def test_unusable_answers_are_rejected(reason, text, code):
    """Truncation, filtering, unexpected finishes, and empty output never become answers."""
    response = AIMessage(content=text, response_metadata={"finish_reason": reason})
    with pytest.raises(AnswerGenerationError) as error:
        answer_text(response)
    assert error.value.code == code


def test_refusal_and_tool_requests_are_not_published_as_answers():
    """A nominal stop reason cannot make a refusal or tool request into a completed answer."""
    responses = [
        AIMessage(content="Hidden refusal", additional_kwargs={"refusal": "refused"}),
        AIMessage(
            content="Call a tool",
            tool_calls=[{"name": "search", "args": {}, "id": "call_1"}],
        ),
    ]
    for response in responses:
        with pytest.raises(AnswerGenerationError, match="complete answer"):
            answer_text(response)


@pytest.mark.parametrize("reason", [None, "stop"])
def test_complete_answer_preserves_markdown_and_whitespace(reason):
    """Normal text is published without editing citations or formatting."""
    text = "  **Habits** improve with repetition [S1].\n"
    assert (
        answer_text(
            AIMessage(content=text, response_metadata={"finish_reason": reason})
        )
        == text
    )


class AnswerGenerationRunTests(unittest.IsolatedAsyncioTestCase):
    """Use actual conversation transitions with a fake database and controlled LLM."""

    def setUp(self):
        """Prepare an assembled run, visible messages, and mock transaction contexts."""
        self.user_id = uuid4()
        self.user = ConversationMessage(
            id=uuid4(), role=MessageRole.USER, content="What about {Book B}?"
        )
        self.assistant = ConversationMessage(
            id=uuid4(), role=MessageRole.ASSISTANT, content=""
        )
        self.run = ConversationRun(
            id=uuid4(),
            conversation_id=uuid4(),
            user_message_id=self.user.id,
            assistant_message_id=self.assistant.id,
            status=RunStatus.RUNNING,
            stage="context_ready",
            model="stored-answer-model",
            rewritten_query="Book B habit formation",
            rewrite_model="rewrite-model",
            snapshot=RunSnapshot(
                prompt_messages=[
                    PromptMessageSnapshot(
                        role="system", content="Use the supplied evidence."
                    ),
                    PromptMessageSnapshot(
                        role="human",
                        content='Evidence: {"habit": "practice"}\nQuestion: What about {Book B}?',
                    ),
                ],
                rewrite_prompt_messages=[
                    PromptMessageSnapshot(role="human", content="Saved rewrite")
                ],
                prompt_token_estimate=100,
                answer_token_budget=60,
            ),
            metrics=RunMetrics(
                rewrite_fallback=True, stage_durations_ms={"rewrite": 12}
            ),
        )
        self.conversation = SimpleNamespace(updated_at=None)
        self.db = AsyncMock()
        self.db.get.side_effect = self.get_row
        self.sessions = MagicMock()
        self.transaction = self.sessions.begin.return_value
        self.transaction.__aenter__.return_value = self.db
        self.response = AIMessage(
            content="Book B emphasizes repeated practice [S1].",
            response_metadata={"finish_reason": "stop", "request_id": "request-123"},
            usage_metadata={"input_tokens": 35, "output_tokens": 9, "total_tokens": 44},
        )
        self.llm = LLMService(model="injected-answer-model")
        self.llm.generate_response = AsyncMock(return_value=self.response)
        self.lock = patch.object(
            ConversationService, "_locked_run", new=AsyncMock(return_value=self.run)
        )
        self.lock.start()
        self.addCleanup(self.lock.stop)

    async def get_row(self, model, identity):
        """Return only the rows used when completing the visible assistant message."""
        if model is ConversationMessage:
            return self.user if identity == self.user.id else self.assistant
        if model is Conversation:
            return self.conversation
        raise AssertionError("Unexpected database row lookup")

    async def generate(self, llm=None):
        """Run answer generation using the supplied or default controlled provider."""
        return await AnswerGenerationService(llm or self.llm).generate_run(
            self.run.id, self.user_id, sessions=self.sessions
        )

    async def test_success_uses_saved_input_and_publishes_answer_with_usage(self):
        """The exact captured prompt and output allowance reach the model outside a transaction."""
        snapshot_before = self.run.snapshot.model_copy(deep=True)

        async def respond(prompt, values, **options):
            """Assert the claim was committed before the provider starts."""
            assert self.run.stage == "generating"
            assert self.transaction.__aexit__.await_count == 1
            assert self.assistant.content == ""
            return self.response

        self.llm.generate_response.side_effect = respond
        answer = await self.generate()
        prompt, values = self.llm.generate_response.call_args.args
        messages = prompt.format_messages(**values)
        assert [
            PromptMessageSnapshot(role=message.type, content=message.content)
            for message in messages
        ] == snapshot_before.prompt_messages
        assert self.llm.generate_response.call_args.kwargs == {
            "max_tokens": 60,
            "user_id": str(self.user_id),
        }
        assert answer == self.response.content
        assert self.run.status == RunStatus.COMPLETED
        assert self.run.stage == "completed"
        assert self.run.response_text == answer == self.assistant.content
        assert self.run.finished_at is not None
        assert self.conversation.updated_at is not None
        assert self.run.model == "injected-answer-model"
        assert self.run.metrics.input_tokens == 35
        assert self.run.metrics.output_tokens == 9
        assert self.run.metrics.finish_reason == "stop"
        assert self.run.metrics.provider_request_id == "request-123"
        assert self.run.metrics.rewrite_fallback
        assert self.run.metrics.stage_durations_ms == {"rewrite": 12}
        assert self.run.snapshot == snapshot_before
        assert self.user.content == "What about {Book B}?"
        assert self.run.rewritten_query == "Book B habit formation"
        assert self.run.rewrite_model == "rewrite-model"

    async def test_default_provider_uses_the_model_recorded_on_the_run(self):
        """No injected LLM means the model chosen when starting the run is respected."""
        with patch(
            "app.services.answer_generation_service.LLMService", return_value=self.llm
        ) as factory:
            await AnswerGenerationService().generate_run(
                self.run.id, self.user_id, sessions=self.sessions
            )
        factory.assert_called_once_with(model="stored-answer-model")
        assert self.run.model == "stored-answer-model"

    async def test_provider_failure_is_safe_and_does_not_publish(self):
        """Provider errors become a generic application error and a failed attempt."""
        self.llm.generate_response.side_effect = RuntimeError(
            "private account or API key details"
        )
        with pytest.raises(AnswerGenerationError) as error:
            await self.generate()
        assert error.value.code == "answer_generation_failed"
        assert "private" not in str(error.value)
        assert error.value.__suppress_context__
        assert self.run.status == RunStatus.FAILED
        assert self.run.error_code == "answer_generation_failed"
        assert self.run.finished_at is not None
        assert self.assistant.content == ""
        assert self.run.response_text is None
        assert "private" not in str(self.run.metrics)

    async def test_empty_or_truncated_output_fails_with_safe_metrics(self):
        """Unusable output records usage but leaves the assistant placeholder empty."""
        for reason, text, code in (
            ("length", "Partial", "answer_truncated"),
            ("stop", "  ", "answer_empty"),
        ):
            with self.subTest(reason=reason):
                self.run.status = RunStatus.RUNNING
                self.run.stage = "context_ready"
                self.response.response_metadata["finish_reason"] = reason
                self.response.content = text
                with pytest.raises(AnswerGenerationError) as error:
                    await self.generate()
                assert error.value.code == code
                assert self.run.status == RunStatus.FAILED
                assert self.run.metrics.finish_reason == reason
                assert self.run.metrics.input_tokens == 35
                assert self.assistant.content == ""

    async def test_missing_usage_stays_unknown_instead_of_using_estimated_tokens(self):
        """Provider-reported token counts are kept separate from context estimates."""
        self.llm.generate_response.return_value = AIMessage(
            content="A clarifying question."
        )
        await self.generate()
        assert self.run.metrics.input_tokens is None
        assert self.run.metrics.output_tokens is None
        assert self.run.metrics.finish_reason is None
        assert self.run.metrics.provider_request_id is None
        assert self.run.snapshot.prompt_token_estimate == 100

    async def test_missing_context_or_budget_stops_before_calling_provider(self):
        """Incomplete assembled context is rejected before claiming the run."""
        original = self.run.snapshot.model_copy(deep=True)
        for missing in ("prompt", "budget"):
            with self.subTest(missing=missing):
                self.run.snapshot = original.model_copy(deep=True)
                if missing == "prompt":
                    self.run.snapshot.prompt_messages = []
                else:
                    self.run.snapshot.answer_token_budget = None
                with pytest.raises(ValidationError):
                    await self.generate()
                assert self.run.stage == "context_ready"
        self.llm.generate_response.assert_not_awaited()

    async def test_missing_or_foreign_run_stops_before_calling_provider(self):
        """Generation relies on the existing locked ownership check."""
        with (
            patch.object(
                ConversationService,
                "_locked_run",
                new=AsyncMock(side_effect=NotFoundError),
            ),
            pytest.raises(NotFoundError),
        ):
            await self.generate()
        self.llm.generate_response.assert_not_awaited()

    async def test_finished_or_already_generating_attempt_cannot_call_provider(self):
        """Only a context-ready running attempt can be claimed."""
        for status, stage in (
            (RunStatus.PENDING, "context_ready"),
            (RunStatus.COMPLETED, "completed"),
            (RunStatus.FAILED, "failed"),
            (RunStatus.CANCELLED, "cancelled"),
            (RunStatus.RUNNING, "generating"),
            (RunStatus.RUNNING, "starting"),
        ):
            with self.subTest(status=status, stage=stage):
                self.run.status, self.run.stage = status, stage
                with pytest.raises(ConflictError):
                    await self.generate()
        self.llm.generate_response.assert_not_awaited()

    async def test_two_workers_cannot_generate_the_same_attempt(self):
        """A second worker is rejected while the first worker awaits its model response."""
        started, release = asyncio.Event(), asyncio.Event()

        async def wait_for_release(*args, **kwargs):
            """Keep the first provider call open until the duplicate claim is rejected."""
            started.set()
            await release.wait()
            return self.response

        self.llm.generate_response.side_effect = wait_for_release
        first = asyncio.create_task(self.generate())
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            with pytest.raises(ConflictError):
                await self.generate()
        finally:
            release.set()
            await first
        self.llm.generate_response.assert_awaited_once()

    async def test_claimed_prompt_cannot_be_replaced_by_late_context_or_rewrite(self):
        """Earlier pipeline stages cannot reset a claimed run or change its captured input."""
        self.run.stage = "generating"
        snapshot_before = self.run.snapshot.model_copy(deep=True)
        options = [
            {"stage": "context_ready", "snapshot": RunSnapshot()},
            {"stage": "rewritten", "rewritten_query": "Late query"},
            {"stage": "generating", "snapshot": RunSnapshot()},
        ]
        for option in options:
            with (
                self.subTest(option=option),
                pytest.raises(ConflictError, match="inputs cannot change"),
            ):
                await ConversationService(self.db, self.user_id).checkpoint(
                    self.run.id, **option
                )
        assert self.run.stage == "generating"
        assert self.run.snapshot == snapshot_before

    async def test_external_cancellation_rejects_late_completion(self):
        """An answer returned after cancellation never fills the visible message."""

        async def cancel_and_respond(*args, **kwargs):
            """Simulate a user cancellation while the model is working."""
            await ConversationService(self.db, self.user_id).cancel(self.run.id)
            return self.response

        self.llm.generate_response.side_effect = cancel_and_respond
        with pytest.raises(ConflictError):
            await self.generate()
        assert self.run.status == RunStatus.CANCELLED
        assert self.assistant.content == ""
        assert self.run.response_text is None

    async def test_provider_failure_after_cancellation_does_not_replace_cancelled_state(
        self,
    ):
        """Cancellation takes precedence even when the provider subsequently fails."""

        async def cancel_and_fail(*args, **kwargs):
            """Cancel the attempt before raising a controlled provider error."""
            await ConversationService(self.db, self.user_id).cancel(self.run.id)
            raise RuntimeError("private provider details")

        self.llm.generate_response.side_effect = cancel_and_fail
        with pytest.raises(AnswerGenerationError):
            await self.generate()
        assert self.run.status == RunStatus.CANCELLED
        assert self.run.error_code is None
        assert self.assistant.content == ""

    async def test_task_cancellation_marks_run_cancelled_and_propagates(self):
        """Cancelling the generation coroutine closes the attempt without publishing."""
        self.llm.generate_response.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await self.generate()
        assert self.run.status == RunStatus.CANCELLED
        assert self.assistant.content == ""
        assert self.run.error_code is None


class LLMResponseTests(unittest.IsolatedAsyncioTestCase):
    """Verify the LLM wrapper and serialized SDK body using local fake transports."""

    async def test_deepseek_output_limit_and_user_id_reach_actual_request_body(self):
        """The SDK sends max_tokens without ChatOpenAI renaming it to max_completion_tokens."""
        bodies = []

        def respond(request):
            """Capture the serialized request and return a local chat-completion response."""
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "answer-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Answer [S1]."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 7,
                        "total_tokens": 32,
                    },
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            model = ChatOpenAI(
                model="answer-model",
                api_key="test-key",
                base_url="https://provider.test/v1",
                http_async_client=client,
                use_responses_api=False,
                max_retries=0,
            )
            llm = LLMService(chat_model=model)
            response = await llm.generate_response(
                ChatPromptTemplate.from_messages([("human", "Question: {question}")]),
                {"question": "What about {Book B}?"},
                max_tokens=60,
                user_id="user-123",
            )
        assert len(bodies) == 1
        assert bodies[0]["max_tokens"] == 60
        assert "max_completion_tokens" not in bodies[0]
        assert bodies[0]["user_id"] == "user-123"
        assert bodies[0]["messages"][0]["content"] == "Question: What about {Book B}?"
        assert response.content == "Answer [S1]."
        assert response.usage_metadata["input_tokens"] == 25
        assert response.response_metadata["finish_reason"] == "stop"

    async def test_existing_generate_callers_still_receive_text(self):
        """Summary and query-rewrite callers retain the original string-returning API."""
        llm = LLMService(
            chat_model=FakeListChatModel(responses=["Existing text result"])
        )
        result = await llm.generate(
            ChatPromptTemplate.from_messages([("human", "Q")]), {}
        )
        assert result == "Existing text result"

    async def test_invalid_output_limit_does_not_invoke_model(self):
        """Reject nonpositive output budgets before any provider request is made."""
        llm = LLMService()
        with (
            patch.object(LLMService, "chat_model") as model,
            pytest.raises(ValueError, match="positive"),
        ):
            await llm.generate_response(
                ChatPromptTemplate.from_messages([("human", "Q")]), {}, max_tokens=0
            )
        model.ainvoke.assert_not_called()
