"""Pipeline ordering and failure tests without database, Redis, or provider calls."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.exception import ConflictError, NotFoundError
from app.models.conversation import RunSnapshot
from app.services.answer_generation_service import AnswerGenerationError
from app.services.conversation_pipeline import ConversationPipeline
from app.services.retrieval_service import RetrievalResult


class ConversationPipelineTests(unittest.IsolatedAsyncioTestCase):
    """Use controlled stage services to verify the background workflow."""

    def setUp(self):
        """Supply fake short sessions and one service double per workflow stage."""
        self.sessions = MagicMock()
        self.db = AsyncMock()
        self.sessions.return_value.__aenter__.return_value = self.db
        self.sessions.begin.return_value.__aenter__.return_value = self.db
        self.run_id, self.user_id, self.conversation_id, self.project_id = [
            uuid4() for _ in range(4)
        ]
        self.run = SimpleNamespace(
            conversation_id=self.conversation_id, snapshot=RunSnapshot()
        )
        self.conversation = AsyncMock()
        self.conversation.start_run.return_value = self.run
        self.conversation.get_run.return_value = self.run
        self.conversation.get.return_value = SimpleNamespace(project_id=self.project_id)
        self.rewriter = AsyncMock()
        self.rewriter.rewrite_run.return_value = SimpleNamespace(
            query="Book B habit formation"
        )
        self.retrieval = AsyncMock()
        self.retrieval.retrieve.return_value = RetrievalResult([], [])
        self.context = AsyncMock()
        self.generation = AsyncMock()
        for name, value in (
            ("ConversationService", self.conversation),
            ("QueryRewriteService", self.rewriter),
            ("RetrievalService", self.retrieval),
            ("AnswerContextService", self.context),
            ("AnswerGenerationService", self.generation),
        ):
            patcher = patch(
                f"app.services.conversation_pipeline.{name}", return_value=value
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        self.pipeline = ConversationPipeline(sessions=self.sessions)

    async def test_workflow_uses_standalone_query_and_owned_project(self):
        """The stage handoff preserves the run and supplies retrieval with the rewritten query."""
        await self.pipeline.process(self.run_id, self.user_id)
        self.retrieval.retrieve.assert_awaited_once_with(
            "Book B habit formation", project_id=self.project_id
        )
        self.context.assemble_run.assert_awaited_once_with(
            self.run_id,
            self.user_id,
            self.retrieval.retrieve.return_value,
            sessions=self.sessions,
        )
        self.generation.generate_run.assert_awaited_once_with(
            self.run_id, self.user_id, sessions=self.sessions
        )
        stages = [
            call.kwargs["stage"] for call in self.conversation.checkpoint.call_args_list
        ]
        assert stages == ["retrieving", "retrieved"]
        self.conversation.fail_if_active.assert_not_awaited()

    async def test_redelivered_cancelled_or_deleted_task_is_harmless(self):
        """Only a successful pending-run claim is allowed to start provider work."""
        for error in (ConflictError(), NotFoundError()):
            with self.subTest(error=type(error).__name__):
                self.conversation.start_run.side_effect = error
                await self.pipeline.process(self.run_id, self.user_id)
        self.rewriter.rewrite_run.assert_not_awaited()

    async def test_stage_failure_records_only_safe_code_and_stops(self):
        """Retrieval failures never invoke answer generation or persist provider details."""
        self.retrieval.retrieve.side_effect = RuntimeError(
            "private embedding account details"
        )
        await self.pipeline.process(self.run_id, self.user_id)
        self.conversation.fail_if_active.assert_awaited_once_with(
            self.run_id, "retrieval_failed"
        )
        self.context.assemble_run.assert_not_awaited()
        self.generation.generate_run.assert_not_awaited()

    async def test_generation_error_keeps_its_more_specific_failure(self):
        """An already-recorded generation failure is not replaced by a generic pipeline code."""
        self.generation.generate_run.side_effect = AnswerGenerationError(
            "answer_truncated"
        )
        await self.pipeline.process(self.run_id, self.user_id)
        self.conversation.fail_if_active.assert_not_awaited()

    async def test_task_cancellation_is_recorded_and_propagated(self):
        """Worker shutdown can cancel a running workflow without publishing an answer."""
        self.rewriter.rewrite_run.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await self.pipeline.process(self.run_id, self.user_id)
        self.conversation.cancel.assert_awaited_once_with(self.run_id)
        self.generation.generate_run.assert_not_awaited()
