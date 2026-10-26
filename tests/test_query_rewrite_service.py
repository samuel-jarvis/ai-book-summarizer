"""Rewrite contract and rendered prompt tests with a local fake chat model."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.core.exception import ConflictError, ValidationError
from app.models.conversation import (
    ConversationRun,
    PromptMessageSnapshot,
    RunMetrics,
    RunSnapshot,
    RunStatus,
)
from app.services.conversation_service import ConversationService
from app.services.history_service import budget_history
from app.services.llm.llm_service import LLMService
from app.services.query_rewrite_service import (
    QueryRewriteService,
    RewriteResult,
    validate_rewrite,
)


class RewriteValidationTests(unittest.TestCase):
    """Reject ambiguous and malformed output before retrieval uses it."""

    def test_ready_query_is_trimmed(self):
        """A valid single-line query can have surrounding spaces."""
        self.assertEqual(
            validate_rewrite('{"status":"ready","query":" Book B habits? "}'),
            "Book B habits?",
        )

    def test_unusable_outputs_are_rejected(self):
        """Strict JSON, status, query length, and control characters are checked."""
        outputs = [
            "",
            "not JSON",
            "null",
            "[]",
            '```json\n{"status":"ready","query":"Q"}\n```',
            '{"query":"Q"}',
            '{"status":"unknown","query":"Q"}',
            '{"status":"ready","query":5}',
            '{"status":"ready","query":"Q","answer":"A"}',
            '{"status":"ambiguous","query":"Guessed book"}',
        ]
        outputs.extend(
            json.dumps({"status": "ready", "query": query})
            for query in (
                "",
                "   ",
                "???",
                "x" * 1001,
                "two\nlines",
                "tab\there",
                "control\x00",
                "line\u2028break",
            )
        )
        for output in outputs:
            with self.subTest(output=output[:80]):
                self.assertIsNone(validate_rewrite(output))


class QueryRewriteTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the actual prompt rendering and LLM abstraction without a provider."""

    def setUp(self):
        """Create history that clearly identifies two ordered books and a topic."""
        self.history = budget_history(
            [
                (
                    "Compare habit formation in Book A and Book B.",
                    "First, Book A emphasizes cues. Second, Book B emphasizes repetition.",
                )
            ]
        )

    def service(self, output):
        """Return a rewriter whose chat model emits controlled JSON."""
        llm = LLMService(
            model="rewrite-test", chat_model=FakeListChatModel(responses=[output])
        )
        llm.generate = AsyncMock(wraps=llm.generate)
        return QueryRewriteService(llm)

    async def test_followup_becomes_standalone_and_prompt_is_captured(self):
        """A second-book follow-up uses history while preserving the question."""
        question = "  What about the second book? {literal braces}  "
        service = self.service(
            '{"status":"ready","query":"How does Book B explain habit formation?"}'
        )
        result = await service.rewrite(question, self.history)
        self.assertEqual(result.query, "How does Book B explain habit formation?")
        self.assertFalse(result.fallback)
        self.assertEqual(result.model, "rewrite-test")
        self.assertEqual(
            [m.role for m in result.prompt_messages], ["system", "human", "ai", "human"]
        )
        self.assertEqual(
            result.prompt_messages[-1].content, "Latest user question:\n" + question
        )
        self.assertIn("multiple plausible meanings", result.prompt_messages[0].content)
        self.assertEqual(service.llm.generate.call_args.args[1]["question"], question)

    async def test_topic_change_and_standalone_question_keep_new_intent(self):
        """The prompt requires old context to be dropped for a clear topic change."""
        for question in ("How does Book C explain sleep?", "What is photosynthesis?"):
            with self.subTest(question=question):
                result = await self.service(
                    json.dumps({"status": "ready", "query": question})
                ).rewrite(question, self.history)
                self.assertEqual(result.query, question)
                self.assertFalse(result.fallback)
                self.assertIn("clear topic change", result.prompt_messages[0].content)

    async def test_ambiguous_and_malformed_results_fall_back_to_exact_question(self):
        """An uncertain referent or unusable reply never turns into a guessed query."""
        question = "  What does it say about that?  "
        for output in ('{"status":"ambiguous","query":""}', "bad output"):
            with self.subTest(output=output):
                result = await self.service(output).rewrite(question, self.history)
                self.assertEqual(result.query, question)
                self.assertTrue(result.fallback)
                self.assertTrue(result.prompt_messages)

    async def test_first_turn_or_empty_budget_skips_provider(self):
        """No context means a normal raw-query path rather than a failed rewrite."""
        service = self.service("unused")
        history = budget_history([("Q", "A")], token_budget=0)
        result = await service.rewrite("What about it?", history)
        self.assertEqual(result.query, "What about it?")
        self.assertEqual(result.prompt_messages, [])
        self.assertIsNone(result.model)
        self.assertFalse(result.fallback)
        service.llm.generate.assert_not_awaited()

    async def test_memory_alone_can_resolve_a_followup(self):
        """Available memory is part of the rendered input even without recent pairs."""
        history = budget_history(
            [], memory="Second book: Book B. Current topic: habit formation."
        )
        result = await self.service(
            '{"status":"ready","query":"Book B habit formation"}'
        ).rewrite("And the second book?", history)
        self.assertEqual(result.query, "Book B habit formation")
        self.assertIn("Saved conversation memory", result.prompt_messages[1].content)

    async def test_provider_error_falls_back_without_persisting_error_text(self):
        """Provider failure keeps the raw query and safe audit fields."""
        llm = LLMService(model="rewrite-test")
        llm.generate = AsyncMock(side_effect=RuntimeError("private provider details"))
        result = await QueryRewriteService(llm).rewrite("And Book B?", self.history)
        self.assertTrue(result.fallback)
        self.assertEqual(result.query, "And Book B?")
        self.assertNotIn("private provider", str(result))

    async def test_cancellation_propagates(self):
        """Cancelling inference stops the operation instead of returning a fallback."""
        llm = LLMService()
        llm.generate = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await QueryRewriteService(llm).rewrite("And Book B?", self.history)

    async def test_empty_question_is_rejected(self):
        """Blank questions cannot enter retrieval even when no history exists."""
        with self.assertRaises(ValidationError):
            await self.service("unused").rewrite(" \n ", budget_history([]))


class RewriteCheckpointTests(unittest.IsolatedAsyncioTestCase):
    """Check audit-state transitions without needing a PostgreSQL connection."""

    async def test_rewrite_checkpoint_merges_and_replaces_json_metadata(self):
        """Captured inputs and fallback persist while existing metadata survives."""
        snapshot = RunSnapshot(
            prompt_messages=[
                PromptMessageSnapshot(role="human", content="Existing context")
            ]
        )
        metrics = RunMetrics(stage_durations_ms={"earlier": 10})
        run = ConversationRun(
            id=uuid4(), status=RunStatus.RUNNING, snapshot=snapshot, metrics=metrics
        )
        history = budget_history([("Earlier question", "Earlier answer")])
        prompt = [PromptMessageSnapshot(role="human", content="What about it?")]
        result = RewriteResult(
            "What about it?", history, prompt, "rewrite-test", fallback=True
        )
        db = AsyncMock()
        service = ConversationService(db, uuid4())
        service._locked_run = AsyncMock(return_value=run)
        await service.checkpoint(run.id, stage="rewritten", rewrite_result=result)
        service._locked_run.assert_awaited_once_with(run.id)
        db.flush.assert_awaited_once()
        self.assertEqual(run.rewritten_query, "What about it?")
        self.assertEqual(run.rewrite_model, "rewrite-test")
        self.assertTrue(run.metrics.rewrite_fallback)
        self.assertEqual(run.snapshot.history_messages, history.messages)
        self.assertEqual(run.snapshot.rewrite_prompt_messages, prompt)
        self.assertEqual(run.snapshot.prompt_messages, snapshot.prompt_messages)
        self.assertEqual(run.metrics.stage_durations_ms, metrics.stage_durations_ms)
        self.assertIsNot(run.snapshot, snapshot)
        self.assertIsNot(run.metrics, metrics)
        self.assertEqual(snapshot.rewrite_prompt_messages, [])
        self.assertFalse(metrics.rewrite_fallback)
        prompt[0].content = "Changed later"
        self.assertEqual(
            run.snapshot.rewrite_prompt_messages[0].content, "What about it?"
        )

    async def test_skipped_rewrite_clears_previous_model_and_fallback(self):
        """A skipped rewrite has null model and false fallback even after prior data."""
        run = ConversationRun(
            id=uuid4(),
            status=RunStatus.RUNNING,
            snapshot=RunSnapshot(),
            metrics=RunMetrics(rewrite_fallback=True),
            rewrite_model="previous-model",
        )
        service = ConversationService(AsyncMock(), uuid4())
        service._locked_run = AsyncMock(return_value=run)
        result = RewriteResult("Original", budget_history([]), [], None)
        await service.checkpoint(run.id, stage="rewritten", rewrite_result=result)
        self.assertIsNone(run.rewrite_model)
        self.assertFalse(run.metrics.rewrite_fallback)

    async def test_finished_attempt_rejects_late_rewrite(self):
        """The status check rejects audit writes after completion or cancellation."""
        for status in (
            RunStatus.PENDING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            with self.subTest(status=status):
                run = ConversationRun(id=uuid4(), status=status)
                db = AsyncMock()
                service = ConversationService(db, uuid4())
                service._locked_run = AsyncMock(return_value=run)
                result = RewriteResult("Original", budget_history([]), [], None)
                with self.assertRaises(ConflictError):
                    await service.checkpoint(
                        run.id, stage="rewritten", rewrite_result=result
                    )
                db.flush.assert_not_awaited()
