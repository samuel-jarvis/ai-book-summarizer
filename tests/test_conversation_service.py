"""Real PostgreSQL lifecycle and concurrency tests in disposable schemas.

Set CHAT_TEST_DATABASE_URL to run. No inference or embedding calls are made.
"""

import asyncio
import importlib.util
import os
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
