"""Real PostgreSQL lifecycle and concurrency tests in disposable schemas.

Set CHAT_TEST_DATABASE_URL to run. No inference or embedding calls are made.
"""

import asyncio
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

import app.models  # noqa: F401
from app.core.exception import ConflictError, NotFoundError
from app.models.conversation import (
    ConversationMessage,
    ConversationRun,
    PromptMessageSnapshot,
    RunSnapshot,
    RunStatus,
)
from app.models.project import Project
from app.models.user import User
from app.services.conversation_service import ConversationService
from app.services.history_service import HistoryService
from app.services.llm.llm_service import LLMService
from app.services.query_rewrite_service import QueryRewriteService


@unittest.skipUnless(os.getenv("CHAT_TEST_DATABASE_URL"), "PostgreSQL URL not supplied")
class ConversationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = "chat_test_" + uuid4().hex
        self.engine = create_async_engine(
            os.environ["CHAT_TEST_DATABASE_URL"],
            connect_args={"server_settings": {"search_path": f"{self.schema},public"}},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        spec = importlib.util.spec_from_file_location(
            "conversation_migration",
            Path("alembic/versions/a84365bf9fb4_add_conversations.py"),
        )
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)

        def create_tables(connection):
            User.__table__.create(connection)
            Project.__table__.create(connection)
            with Operations.context(MigrationContext.configure(connection)):
                migration.upgrade()

        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema))
            await connection.run_sync(create_tables)
        self.user_id, self.other_id = uuid4(), uuid4()
        async with self.sessions.begin() as db:
            db.add_all(
                [
                    User(id=self.user_id, email="owner@example.test"),
                    User(id=self.other_id, email="other@example.test"),
                ]
            )
            await db.flush()
            project = Project(user_id=self.user_id, title="Books")
            db.add(project)
            await db.flush()
            self.project_id = project.id
            self.conversation_id = (
                await ConversationService(db, self.user_id).create(
                    self.project_id, "Discussion"
                )
            ).id

    async def asyncTearDown(self):
        try:
            async with self.engine.begin() as connection:
                await connection.execute(
                    DropSchema(self.schema, cascade=True, if_exists=True)
                )
        finally:
            await self.engine.dispose()

    async def reserve(self, content="Explain habit formation", request_id=None):
        async with self.sessions.begin() as db:
            return await ConversationService(db, self.user_id).reserve_turn(
                self.conversation_id,
                content=content,
                client_request_id=request_id or uuid4(),
            )

    async def test_owner_checks_and_crud(self):
        async with self.sessions.begin() as db:
            other = ConversationService(db, self.other_id)
            with self.assertRaises(NotFoundError):
                await other.get(self.conversation_id)
            with self.assertRaises(NotFoundError):
                await other.create(self.project_id)
            owner = ConversationService(db, self.user_id)
            await owner.rename(self.conversation_id, " New title ")
            self.assertEqual(
                (await owner.list_conversations(self.project_id))[0].title, "New title"
            )
        turn = await self.reserve()
        async with self.sessions.begin() as db:
            with self.assertRaises(NotFoundError):
                await ConversationService(db, self.other_id).get_run(turn.run.id)
            owner = ConversationService(db, self.user_id)
            messages = await owner.list_messages(self.conversation_id, after_position=1)
            self.assertEqual([m.position for m in messages], [2])
            await owner.delete(self.conversation_id)
        async with self.sessions.begin() as db:
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(ConversationRun)), 0
            )
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(ConversationMessage)),
                0,
            )

    async def test_duplicate_and_conflicting_request(self):
        request_id = uuid4()
        first = await self.reserve(request_id=request_id)
        repeated = await self.reserve(request_id=request_id)
        self.assertTrue(repeated.reused)
        self.assertEqual(first.run.id, repeated.run.id)
        with self.assertRaises(ConflictError):
            await self.reserve("Changed question", request_id)
        with self.assertRaises(ConflictError):
            await self.reserve("Another message")

    async def test_completed_turn_snapshot_and_ordering(self):
        turn = await self.reserve()
        snapshot = RunSnapshot(
            rewrite_prompt_messages=[
                PromptMessageSnapshot(role="human", content="Rewrite this")
            ],
            prompt_messages=[
                PromptMessageSnapshot(role="human", content="Evidence and question")
            ],
        )
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(
                turn.run.id, model="test-model", pipeline_version="chat-v1"
            )
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).checkpoint(
                turn.run.id,
                stage="generating",
                snapshot=snapshot,
                rewritten_query="Habit formation in Book A",
                response_text="Partial",
            )
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            run = await service.get_run(turn.run.id)
            self.assertEqual(run.snapshot, snapshot)
            self.assertEqual(run.response_text, "Partial")
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                "",
            )
            await service.complete(turn.run.id, "Final answer [S1]")
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            run = await service.get_run(turn.run.id)
            self.assertEqual(run.status, RunStatus.COMPLETED)
            self.assertIsNotNone(run.finished_at)
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                "Final answer [S1]",
            )
            with self.assertRaises(ConflictError):
                await service.complete(turn.run.id, "Late write")
        next_turn = await self.reserve("And Book B?")
        self.assertEqual(next_turn.user_message.position, 3)
        self.assertEqual(next_turn.assistant_message.position, 4)

    async def test_cancel_retry_and_stale_completion(self):
        turn = await self.reserve()
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(turn.run.id, model="test", pipeline_version="v1")
            await service.checkpoint(
                turn.run.id, stage="generating", response_text="Partial"
            )
            await service.cancel(turn.run.id)
            await service.cancel(turn.run.id)
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            retry = await service.retry(turn.run.id)
            self.assertEqual(retry.attempt_number, 2)
            self.assertEqual(retry.assistant_message_id, turn.assistant_message.id)
            self.assertEqual(
                (await service.get_run(turn.run.id)).response_text, "Partial"
            )
            with self.assertRaises(ConflictError):
                await service.complete(turn.run.id, "Stale output")
            with self.assertRaises(ConflictError):
                await service.checkpoint(
                    turn.run.id, stage="generating", response_text="Stale"
                )
            await service.start_run(retry.id, model="test", pipeline_version="v1")
            await service.complete(retry.id, "Retry answer")

    async def test_failure_and_retry_of_older_turn_rejected(self):
        turn = await self.reserve()
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).fail(
                turn.run.id, "generation_timeout"
            )
        await self.reserve("New question")
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            with self.assertRaises(ConflictError):
                await service.retry(turn.run.id)

    async def test_reservation_rolls_back_all_rows(self):
        with self.assertRaises(RuntimeError):
            async with self.sessions.begin() as db:
                await ConversationService(db, self.user_id).reserve_turn(
                    self.conversation_id, content="Question", client_request_id=uuid4()
                )
                raise RuntimeError("Simulated failure before commit")
        async with self.sessions.begin() as db:
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(ConversationMessage)),
                0,
            )
            self.assertEqual(
                await db.scalar(select(func.count()).select_from(ConversationRun)), 0
            )

    async def test_concurrent_submissions_serialize(self):
        gate = asyncio.Event()

        async def submit(request_id):
            await gate.wait()
            try:
                return await self.reserve(request_id=request_id)
            except ConflictError:
                return None

        tasks = [asyncio.create_task(submit(uuid4())) for _ in range(2)]
        gate.set()
        results = await asyncio.gather(*tasks)
        self.assertEqual(sum(result is not None for result in results), 1)

    async def test_concurrent_duplicate_submissions_reuse_run(self):
        request_id = uuid4()
        results = await asyncio.gather(
            self.reserve(request_id=request_id), self.reserve(request_id=request_id)
        )
        self.assertEqual(results[0].run.id, results[1].run.id)
        self.assertEqual(sum(result.reused for result in results), 1)

    async def completed_turn(self, question, answer):
        """Create a successful exchange with no provider calls."""
        turn = await self.reserve(question)
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(turn.run.id, model="test", pipeline_version="v1")
            await service.complete(turn.run.id, answer)
        return turn

    async def test_history_excludes_failed_cancelled_empty_and_current_turns(self):
        """History uses completed pairs only and respects owner and turn boundaries."""
        failed = await self.reserve("Failed question")
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(failed.run.id, model="test", pipeline_version="v1")
            await service.checkpoint(
                failed.run.id,
                stage="generating",
                response_text="Partial failed response",
            )
            await service.fail(failed.run.id, "test_failure")
        await self.completed_turn(
            "Compare Book A and Book B habits", "Book B emphasizes repetition."
        )
        cancelled = await self.reserve("Cancelled question")
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).cancel(cancelled.run.id)
        empty = await self.completed_turn("Empty assistant", "Initially valid")
        async with self.sessions.begin() as db:
            assistant = await db.get(ConversationMessage, empty.assistant_message.id)
            assistant.content = " \n\t "
        current = await self.reserve("What about the second book?")
        async with self.sessions() as db:
            history = await HistoryService(db, self.user_id).load(
                self.conversation_id, before_position=current.user_message.position
            )
            self.assertEqual(
                [m.content for m in history.messages],
                ["Compare Book A and Book B habits", "Book B emphasizes repetition."],
            )
            with self.assertRaises(NotFoundError):
                await HistoryService(db, self.other_id).load(
                    self.conversation_id, before_position=current.user_message.position
                )

    async def test_history_includes_successful_retry_once_and_limits_recent_pairs(self):
        """Failed retry attempts add no history and the row limit selects recent pairs."""
        await self.completed_turn("Old topic", "Old answer")
        turn = await self.reserve("Book B habits?")
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.fail(turn.run.id, "test_failure")
            retry = await service.retry(turn.run.id)
            await service.start_run(retry.id, model="test", pipeline_version="v1")
            await service.complete(retry.id, "Book B uses repetition.")
        async with self.sessions() as db:
            history = await HistoryService(db, self.user_id).load(
                self.conversation_id, before_position=5, max_exchanges=1
            )
            self.assertEqual(
                [m.content for m in history.messages],
                ["Book B habits?", "Book B uses repetition."],
            )

    async def test_history_memory_boundary_and_oversized_memory(self):
        """Included memory replaces covered turns; omitted memory leaves them available."""
        first = await self.completed_turn("First topic", "First answer")
        await self.completed_turn("Recent topic", "Recent answer")
        async with self.sessions.begin() as db:
            conversation = await ConversationService(db, self.user_id).get(
                self.conversation_id
            )
            conversation.memory_summary = "Earlier topic: first book."
            conversation.memory_through_position = first.assistant_message.position
        async with self.sessions() as db:
            history = await HistoryService(db, self.user_id).load(
                self.conversation_id, before_position=5
            )
            self.assertEqual(len(history.messages), 3)
            self.assertEqual(
                [m.content for m in history.messages[1:]],
                ["Recent topic", "Recent answer"],
            )
        async with self.sessions.begin() as db:
            conversation = await ConversationService(db, self.user_id).get(
                self.conversation_id
            )
            conversation.memory_summary = "Oversized memory " * 1000
        async with self.sessions() as db:
            history = await HistoryService(db, self.user_id).load(
                self.conversation_id, before_position=5, token_budget=100
            )
            self.assertEqual(
                [m.content for m in history.messages],
                ["First topic", "First answer", "Recent topic", "Recent answer"],
            )

    async def test_rewrite_run_persists_inputs_and_preserves_user_message(self):
        """The rewrite is saved separately and unrelated snapshot/metrics fields survive."""
        await self.completed_turn(
            "Compare Book A and Book B habits", "The second book is Book B."
        )
        question = "  What about the second book?  "
        turn = await self.reserve(question)
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(
                turn.run.id, model="answer-test", pipeline_version="v1"
            )
            run = await service.get_run(turn.run.id)
            snapshot = run.snapshot.model_copy(deep=True)
            snapshot.prompt_messages = [
                PromptMessageSnapshot(role="human", content="Keep this")
            ]
            metrics = run.metrics.model_copy(deep=True)
            metrics.stage_durations_ms = {"previous": 5.0}
            await service.checkpoint(
                turn.run.id, stage="starting", snapshot=snapshot, metrics=metrics
            )
        llm = LLMService(model="rewrite-test")
        llm.generate = AsyncMock(
            return_value='{"status":"ready","query":"How does Book B explain habit formation?"}'
        )
        result = await QueryRewriteService(llm).rewrite_run(
            turn.run.id, self.user_id, sessions=self.sessions
        )
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(turn.run.id)
            self.assertEqual(run.rewritten_query, result.query)
            self.assertEqual(run.rewrite_model, "rewrite-test")
            self.assertEqual(
                run.snapshot.rewrite_prompt_messages, result.prompt_messages
            )
            self.assertEqual(run.snapshot.history_messages, result.history.messages)
            self.assertEqual(run.snapshot.prompt_messages[0].content, "Keep this")
            self.assertEqual(run.metrics.stage_durations_ms, {"previous": 5.0})
            self.assertFalse(run.metrics.rewrite_fallback)
            self.assertEqual(run.stage, "rewritten")
            self.assertEqual(
                (await db.get(ConversationMessage, turn.user_message.id)).content,
                question,
            )

    async def test_rewrite_run_saves_fallback_and_skips_first_turn(self):
        """First turns skip the model; later ambiguous output is persisted as fallback."""
        first = await self.reserve("First question")
        llm = LLMService(model="rewrite-test")
        llm.generate = AsyncMock(return_value='{"status":"ambiguous","query":""}')
        service = QueryRewriteService(llm)
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).start_run(
                first.run.id, model="test", pipeline_version="v1"
            )
        await service.rewrite_run(first.run.id, self.user_id, sessions=self.sessions)
        llm.generate.assert_not_awaited()
        async with self.sessions.begin() as db:
            run = await ConversationService(db, self.user_id).get_run(first.run.id)
            self.assertEqual(run.rewritten_query, "First question")
            self.assertIsNone(run.rewrite_model)
            self.assertEqual(run.snapshot.rewrite_prompt_messages, [])
            await ConversationService(db, self.user_id).complete(
                first.run.id, "First answer"
            )
        second = await self.reserve("What does it mean?")
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).start_run(
                second.run.id, model="test", pipeline_version="v1"
            )
        await service.rewrite_run(second.run.id, self.user_id, sessions=self.sessions)
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(second.run.id)
            self.assertTrue(run.metrics.rewrite_fallback)
            self.assertEqual(run.rewritten_query, "What does it mean?")
            self.assertEqual(run.rewrite_model, "rewrite-test")
            self.assertTrue(run.snapshot.rewrite_prompt_messages)

    async def test_cancel_during_rewrite_rejects_stale_checkpoint(self):
        """A cancelled attempt cannot receive a rewrite that finishes later."""
        await self.completed_turn("Book B habits", "Book B uses repetition.")
        turn = await self.reserve("And the second book?")
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).start_run(
                turn.run.id, model="test", pipeline_version="v1"
            )

        async def cancel_then_return(*args):
            """Simulate cancellation while no inference transaction is held open."""
            async with self.sessions.begin() as db:
                await ConversationService(db, self.user_id).cancel(turn.run.id)
            return '{"status":"ready","query":"Book B habits"}'

        llm = LLMService(model="rewrite-test")
        llm.generate = AsyncMock(side_effect=cancel_then_return)
        with self.assertRaises(ConflictError):
            await QueryRewriteService(llm).rewrite_run(
                turn.run.id, self.user_id, sessions=self.sessions
            )
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(turn.run.id)
            self.assertEqual(run.status, RunStatus.CANCELLED)
            self.assertIsNone(run.rewritten_query)


if __name__ == "__main__":
    unittest.main()
