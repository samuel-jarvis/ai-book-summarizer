"""Real PostgreSQL lifecycle and concurrency tests in disposable schemas.

Set CHAT_TEST_DATABASE_URL to run. No inference or embedding calls are made.
"""

import asyncio
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import httpx
from alembic.migration import MigrationContext
from alembic.operations import Operations
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
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
from app.services.answer_context_service import AnswerContextService
from app.services.answer_generation_service import (
    AnswerGenerationError,
    AnswerGenerationService,
)
from app.services.conversation_service import ConversationService
from app.services.history_service import HistoryService, budget_history
from app.services.llm.llm_service import LLMService
from app.services.query_rewrite_service import QueryRewriteService
from app.services.retrieval_service import RetrievalResult, to_source


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

    async def test_answer_context_checkpoint_reloads_exact_prompt_and_sources(self):
        """The database stores the final input independently of rewrite and message text."""
        turn = await self.reserve("  How does Book B explain habits? {literal}  ")
        history = budget_history([("Book B habits?", "Book B emphasizes practice.")])
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            await service.start_run(turn.run.id, model="test", pipeline_version="v1")
            await service.checkpoint(
                turn.run.id,
                stage="rewritten",
                rewritten_query="Book B habit formation",
                snapshot=RunSnapshot(
                    history_messages=history.messages,
                    rewrite_prompt_messages=[
                        PromptMessageSnapshot(
                            role="human", content="Saved rewrite input"
                        )
                    ],
                ),
            )
        document = Document(
            id=str(uuid4()),
            page_content="Habits develop through repeated practice.",
            metadata={
                "document_id": str(uuid4()),
                "generation_id": str(uuid4()),
                "document_title": "Book B",
                "chunk_index": 0,
                "page_start": 0,
                "page_end": 1,
                "cosine_distance": 0.1,
            },
        )
        retrieval = RetrievalResult([document], [to_source(document, "S1")])
        context = await AnswerContextService().assemble_run(
            turn.run.id, self.user_id, retrieval, sessions=self.sessions
        )
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(turn.run.id)
            self.assertEqual(run.stage, "context_ready")
            self.assertEqual(run.snapshot.prompt_messages, context.messages)
            self.assertEqual(run.snapshot.selected_sources, context.sources)
            self.assertTrue(run.snapshot.retrieval_candidates[0].selected)
            self.assertEqual(run.snapshot.history_messages, history.messages)
            self.assertEqual(
                run.snapshot.rewrite_prompt_messages[0].content, "Saved rewrite input"
            )
            self.assertEqual(run.snapshot.prompt_token_estimate, context.token_count)
            self.assertEqual(run.snapshot.answer_token_budget, 2000)
            self.assertEqual(run.rewritten_query, "Book B habit formation")
            self.assertEqual(
                (await db.get(ConversationMessage, turn.user_message.id)).content,
                "  How does Book B explain habits? {literal}  ",
            )
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                "",
            )

    async def context_ready_turn(self):
        """Create an assembled attempt using actual context persistence and no evidence calls."""
        turn = await self.reserve("How does Book B explain habit formation?")
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).start_run(
                turn.run.id, model="test-answer", pipeline_version="v1"
            )
        await AnswerContextService().assemble_run(
            turn.run.id, self.user_id, RetrievalResult([], []), sessions=self.sessions
        )
        return turn

    async def test_generated_answer_and_reported_usage_reload(self):
        """Successful generation atomically publishes the assistant and persists provider usage."""
        turn = await self.context_ready_turn()
        llm = LLMService(model="test-answer")
        llm.generate_response = AsyncMock(
            return_value=AIMessage(
                content="Please provide the book passage so I can explain its approach.",
                response_metadata={"finish_reason": "stop"},
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 15,
                    "total_tokens": 115,
                },
            )
        )
        service = AnswerGenerationService(llm)
        with self.assertRaises(NotFoundError):
            await service.generate_run(
                turn.run.id, self.other_id, sessions=self.sessions
            )
        llm.generate_response.assert_not_awaited()
        answer = await service.generate_run(
            turn.run.id, self.user_id, sessions=self.sessions
        )
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(turn.run.id)
            self.assertEqual(run.status, RunStatus.COMPLETED)
            self.assertEqual(run.response_text, answer)
            self.assertEqual(run.metrics.input_tokens, 100)
            self.assertEqual(run.metrics.output_tokens, 15)
            self.assertEqual(run.metrics.finish_reason, "stop")
            self.assertEqual(run.model, "test-answer")
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                answer,
            )
            self.assertEqual(
                (await db.get(ConversationMessage, turn.user_message.id)).content,
                "How does Book B explain habit formation?",
            )
            self.assertTrue(run.snapshot.prompt_messages)

    async def test_generation_failure_preserves_placeholder_and_can_be_retried(self):
        """A failed attempt stores a safe code; retry generates into the existing assistant slot."""
        turn = await self.context_ready_turn()
        llm = LLMService(model="test-answer")
        llm.generate_response = AsyncMock(
            side_effect=RuntimeError("private provider details")
        )
        with self.assertRaises(AnswerGenerationError):
            await AnswerGenerationService(llm).generate_run(
                turn.run.id, self.user_id, sessions=self.sessions
            )
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            failed = await service.get_run(turn.run.id)
            self.assertEqual(failed.status, RunStatus.FAILED)
            self.assertEqual(failed.error_code, "answer_generation_failed")
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                "",
            )
            retry = await service.retry(turn.run.id)
            await service.start_run(
                retry.id, model="test-answer", pipeline_version="v1"
            )
        await AnswerContextService().assemble_run(
            retry.id, self.user_id, RetrievalResult([], []), sessions=self.sessions
        )
        llm.generate_response = AsyncMock(
            return_value=AIMessage(content="Which passage should I explain?")
        )
        answer = await AnswerGenerationService(llm).generate_run(
            retry.id, self.user_id, sessions=self.sessions
        )
        async with self.sessions() as db:
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                answer,
            )
            self.assertEqual(
                (
                    await ConversationService(db, self.user_id).get_run(turn.run.id)
                ).status,
                RunStatus.FAILED,
            )
            self.assertEqual(
                (await ConversationService(db, self.user_id).get_run(retry.id)).status,
                RunStatus.COMPLETED,
            )

    async def test_concurrent_generation_calls_claim_the_attempt_once(self):
        """Database locking lets only one worker call the provider for a ready attempt."""
        turn = await self.context_ready_turn()
        started, release = asyncio.Event(), asyncio.Event()

        async def respond_after_release(*args, **kwargs):
            """Hold inference open while another worker attempts the same claim."""
            started.set()
            await release.wait()
            return AIMessage(content="Please provide evidence for Book B.")

        llm = LLMService(model="test-answer")
        llm.generate_response = AsyncMock(side_effect=respond_after_release)
        service = AnswerGenerationService(llm)
        first = asyncio.create_task(
            service.generate_run(turn.run.id, self.user_id, sessions=self.sessions)
        )
        try:
            await asyncio.wait_for(started.wait(), timeout=10)
            with self.assertRaises(ConflictError):
                await service.generate_run(
                    turn.run.id, self.user_id, sessions=self.sessions
                )
        finally:
            release.set()
            await first
        llm.generate_response.assert_awaited_once()

    async def test_cancellation_during_answer_generation_prevents_publication(self):
        """A committed user cancellation rejects the model's later answer."""
        turn = await self.context_ready_turn()

        async def cancel_and_respond(*args, **kwargs):
            """Cancel from a separate transaction while no inference transaction is open."""
            async with self.sessions.begin() as db:
                await ConversationService(db, self.user_id).cancel(turn.run.id)
            return AIMessage(content="Late answer")

        llm = LLMService(model="test-answer")
        llm.generate_response = AsyncMock(side_effect=cancel_and_respond)
        with self.assertRaises(ConflictError):
            await AnswerGenerationService(llm).generate_run(
                turn.run.id, self.user_id, sessions=self.sessions
            )
        async with self.sessions() as db:
            run = await ConversationService(db, self.user_id).get_run(turn.run.id)
            self.assertEqual(run.status, RunStatus.CANCELLED)
            self.assertIsNone(run.response_text)
            self.assertEqual(
                (await db.get(ConversationMessage, turn.assistant_message.id)).content,
                "",
            )

    async def test_conversation_http_pipeline_and_queue_failure_recovery(self):
        """Exercise real HTTP routes and all pipeline stages against PostgreSQL with fake providers."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from app.api.deps import get_current_user
        from app.core.database import get_db
        from app.main import app
        from app.services.conversation_pipeline import ConversationPipeline

        overrides = app.dependency_overrides.copy()
        identity = {"user_id": self.user_id}

        async def database():
            """Use independent real PostgreSQL sessions for each HTTP request."""
            async with self.sessions() as db:
                yield db

        async def current_user():
            """Keep HTTP authentication deterministic while exercising real ownership checks."""
            return SimpleNamespace(id=identity["user_id"])

        async def provider_response(prompt, values, **options):
            """Return deterministic rewrite JSON or a cited answer without paid inference."""
            if options.get("max_tokens"):
                return AIMessage(
                    content="Book B emphasizes repeated practice [S1].",
                    response_metadata={"finish_reason": "stop"},
                )
            return AIMessage(
                content='{"status":"ready","query":"Book B habit formation through repetition"}'
            )

        llm = LLMService(model="test-pipeline")
        llm.generate_response = AsyncMock(side_effect=provider_response)
        pipeline = ConversationPipeline(sessions=self.sessions, llm=llm)

        async def dispatch(run_id, user_id):
            """Execute the queued task inline with independent pipeline sessions for this test."""
            await pipeline.process(UUID(run_id), UUID(user_id))

        document = Document(
            id=str(uuid4()),
            page_content="Book B explains habit formation through repeated practice.",
            metadata={
                "document_id": str(uuid4()),
                "generation_id": str(uuid4()),
                "document_title": "Book B",
                "chunk_index": 0,
                "page_start": 0,
                "page_end": 0,
                "cosine_distance": 0.1,
            },
        )
        retrieval = RetrievalResult([document], [to_source(document, "S1")])
        app.dependency_overrides[get_db] = database
        app.dependency_overrides[get_current_user] = current_user
        try:
            with (
                patch(
                    "app.services.conversation_dispatcher.answer_conversation_task.kiq",
                    new=AsyncMock(side_effect=dispatch),
                ) as queue,
                patch(
                    "app.services.conversation_pipeline.RetrievalService.retrieve",
                    new=AsyncMock(return_value=retrieval),
                ) as search,
            ):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test/api/v1",
                ) as client:
                    created = await client.post(
                        "/conversations",
                        json={
                            "project_id": str(self.project_id),
                            "title": "Book discussion",
                        },
                    )
                    self.assertEqual(created.status_code, 201, created.text)
                    conversation_id = created.json()["data"]["id"]
                    request_id = str(uuid4())
                    first = {
                        "content": "  Explain Book B habit formation.  ",
                        "client_request_id": request_id,
                    }
                    sent = await client.post(
                        f"/conversations/{conversation_id}/messages", json=first
                    )
                    self.assertEqual(sent.status_code, 202, sent.text)
                    run_id = sent.json()["data"]["run"]["id"]
                    polled = await client.get(f"/conversation-runs/{run_id}")
                    self.assertEqual(
                        polled.json()["data"]["status"], "completed", polled.text
                    )
                    self.assertEqual(
                        polled.json()["data"]["sources"][0]["citation_label"], "S1"
                    )
                    duplicate = await client.post(
                        f"/conversations/{conversation_id}/messages", json=first
                    )
                    self.assertTrue(duplicate.json()["data"]["reused"])
                    self.assertEqual(queue.await_count, 1)
                    conflict = await client.post(
                        f"/conversations/{conversation_id}/messages",
                        json={**first, "content": "Changed intent"},
                    )
                    self.assertEqual(conflict.status_code, 409)
                    followup = await client.post(
                        f"/conversations/{conversation_id}/messages",
                        json={
                            "content": "What about repetition?",
                            "client_request_id": str(uuid4()),
                        },
                    )
                    self.assertEqual(followup.status_code, 202, followup.text)
                    followup_id = followup.json()["data"]["run"]["id"]
                    second = (
                        await client.get(f"/conversation-runs/{followup_id}")
                    ).json()["data"]
                    self.assertEqual(second["status"], "completed")
                    self.assertEqual(
                        second["rewritten_query"],
                        "Book B habit formation through repetition",
                    )
                    debug = (
                        await client.get(f"/conversation-runs/{followup_id}/debug")
                    ).json()["data"]
                    self.assertEqual(len(debug["snapshot"]["history_messages"]), 2)
                    self.assertEqual(
                        search.call_args.args[0], second["rewritten_query"]
                    )
                    history = await client.get(
                        f"/conversations/{conversation_id}/messages"
                    )
                    self.assertEqual(
                        [
                            message["position"]
                            for message in history.json()["data"]["messages"]
                        ],
                        [1, 2, 3, 4],
                    )
                    self.assertEqual(
                        history.json()["data"]["messages"][0]["content"],
                        first["content"],
                    )
                    self.assertEqual(
                        history.json()["data"]["messages"][3]["latest_run"]["status"],
                        "completed",
                    )

                    queue.side_effect = RuntimeError("private broker details")
                    third = {
                        "content": "Clarify practice.",
                        "client_request_id": str(uuid4()),
                    }
                    unavailable = await client.post(
                        f"/conversations/{conversation_id}/messages", json=third
                    )
                    self.assertEqual(unavailable.status_code, 503)
                    self.assertNotIn("private broker", unavailable.text)
                    recovered = await client.post(
                        f"/conversations/{conversation_id}/messages", json=third
                    )
                    failed = recovered.json()["data"]["run"]
                    self.assertTrue(recovered.json()["data"]["reused"])
                    self.assertEqual(failed["status"], "failed")
                    self.assertEqual(failed["error_code"], "queue_unavailable")
                    queue.side_effect = dispatch
                    retried = await client.post(
                        f"/conversation-runs/{failed['id']}/retry"
                    )
                    self.assertEqual(retried.status_code, 202, retried.text)
                    self.assertEqual(retried.json()["data"]["attempt_number"], 2)
                    retry_id = retried.json()["data"]["id"]
                    self.assertEqual(
                        (await client.get(f"/conversation-runs/{retry_id}")).json()[
                            "data"
                        ]["status"],
                        "completed",
                    )
                    self.assertEqual(
                        retried.json()["data"]["assistant_message_id"],
                        failed["assistant_message_id"],
                    )

                    identity["user_id"] = self.other_id
                    for path in (
                        f"/conversations/{conversation_id}",
                        f"/conversations/{conversation_id}/messages",
                        f"/conversation-runs/{run_id}",
                        f"/conversation-runs/{run_id}/sources",
                    ):
                        self.assertEqual((await client.get(path)).status_code, 404)
                    identity["user_id"] = self.user_id
                    self.assertEqual(
                        (
                            await client.delete(f"/conversations/{conversation_id}")
                        ).status_code,
                        200,
                    )
                    self.assertEqual(
                        (await client.get(f"/conversation-runs/{run_id}")).status_code,
                        404,
                    )
        finally:
            app.dependency_overrides.clear()
            app.dependency_overrides.update(overrides)


if __name__ == "__main__":
    unittest.main()
