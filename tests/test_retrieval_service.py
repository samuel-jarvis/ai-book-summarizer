"""Retrieval tests: snapshot mapping without a database, then pgvector search.

Set CHAT_TEST_DATABASE_URL to run the PostgreSQL tests. The query embedding is
patched with controlled vectors, so no provider calls are made.
"""

import os
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema

import app.models  # noqa: F401
from app.core.config import settings
from app.core.database import Base
from app.core.exception import NotFoundError
from app.models.conversation import (
    Conversation,
    ConversationMessage,
    ConversationRun,
    PromptMessageSnapshot,
    RunSnapshot,
)
from app.models.document import Document, EmbeddingStatus
from app.models.document_embedding import DocumentEmbedding, DocumentEmbeddingMetadata
from app.models.embedding_generation import EmbeddingGeneration
from app.models.project import Project
from app.models.project_documents import ProjectDocument
from app.models.user import User
from app.services.conversation_service import ConversationService
from app.services.retrieval_service import RetrievalResult, RetrievalService, to_source


def vector(x=1.0, y=0.0):
    return [x, y] + [0.0] * (settings.EMBEDDING_DIMENSION - 2)


class RetrievalResultTests(unittest.TestCase):
    def setUp(self):
        self.document = LangChainDocument(
            id=str(uuid4()),
            page_content="x" * 700,
            metadata={
                "document_id": str(uuid4()),
                "generation_id": str(uuid4()),
                "document_title": "Book A",
                "chunk_index": 0,
                "page_start": 0,
                "page_end": 1,
                "cosine_distance": 0.1,
            },
        )

    def test_source_is_labelled_and_excerpt_truncated(self):
        source = to_source(self.document, "S1")
        self.assertEqual(source.citation_label, "S1")
        self.assertEqual(source.document_title, "Book A")
        self.assertEqual(len(source.excerpt), 600)

    def test_snapshot_keeps_other_fields_and_does_not_mutate_previous(self):
        result = RetrievalResult([self.document], [to_source(self.document, "S1")])
        previous = RunSnapshot(
            rewrite_prompt_messages=[PromptMessageSnapshot(role="human", content="Q")]
        )
        snapshot = result.to_run_snapshot(previous)
        self.assertEqual(
            snapshot.rewrite_prompt_messages, previous.rewrite_prompt_messages
        )
        self.assertEqual(previous.selected_sources, [])
        self.assertTrue(snapshot.retrieval_candidates[0].selected)
        snapshot.selected_sources[0].excerpt = "Changed"
        self.assertNotEqual(result.sources[0].excerpt, "Changed")


@unittest.skipUnless(os.getenv("CHAT_TEST_DATABASE_URL"), "PostgreSQL URL not supplied")
class RetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = "retrieval_test_" + uuid4().hex
        self.engine = create_async_engine(
            os.environ["CHAT_TEST_DATABASE_URL"],
            connect_args={"server_settings": {"search_path": f"{self.schema},public"}},
            execution_options={"schema_translate_map": {None: self.schema}},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.addAsyncCleanup(self.cleanup_schema)
        tables = [
            User,
            Project,
            Document,
            ProjectDocument,
            EmbeddingGeneration,
            DocumentEmbedding,
            Conversation,
            ConversationMessage,
            ConversationRun,
        ]
        async with self.engine.begin() as connection:
            await connection.execute(CreateSchema(self.schema))
            await connection.run_sync(
                lambda conn: Base.metadata.create_all(
                    conn, tables=[model.__table__ for model in tables], checkfirst=False
                )
            )
        self.user_id, self.other_id, self.project_id, self.other_project_id = [
            uuid4() for _ in range(4)
        ]
        async with self.sessions.begin() as db:
            db.add_all(
                [
                    User(id=self.user_id, email="owner@example.test"),
                    User(id=self.other_id, email="other@example.test"),
                ]
            )
            await db.flush()
            db.add_all(
                [
                    Project(id=self.project_id, user_id=self.user_id, title="Books"),
                    Project(
                        id=self.other_project_id, user_id=self.other_id, title="Private"
                    ),
                ]
            )
        embed = patch(
            "app.services.indexing.vector_store.embed_query",
            new=AsyncMock(return_value=vector()),
        )
        embed.start()
        self.addCleanup(embed.stop)

    async def cleanup_schema(self):
        try:
            async with self.engine.begin() as connection:
                await connection.execute(
                    DropSchema(self.schema, cascade=True, if_exists=True)
                )
        finally:
            await self.engine.dispose()

    async def document(self, title="Book A", *, user_id=None, project_id=None):
        async with self.sessions.begin() as db:
            doc = Document(
                user_id=user_id or self.user_id,
                title=title,
                file_size_bytes=1,
                content_hash="a" * 64,
                embedding_status=EmbeddingStatus.COMPLETED,
            )
            db.add(doc)
            await db.flush()
            db.add(
                ProjectDocument(
                    project_id=project_id or self.project_id, document_id=doc.id
                )
            )
            return doc.id

    async def index(self, document_id, chunks=None, *, active=True, model=None):
        chunks = (
            chunks
            if chunks is not None
            else [("Habit formation requires consistent practice.", vector())]
        )
        async with self.sessions.begin() as db:
            generation = EmbeddingGeneration(
                document_id=document_id,
                model=model or settings.EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSION,
                chunking_version="test-v1",
                status=EmbeddingStatus.COMPLETED,
                is_active=active,
            )
            db.add(generation)
            await db.flush()
            chunk_ids = []
            for index, (content, embedding) in enumerate(chunks):
                chunk = DocumentEmbedding(
                    document_id=document_id,
                    generation_id=generation.id,
                    content=content,
                    chunk_index=index,
                    embedding=embedding,
                    embedding_metadata=DocumentEmbeddingMetadata(
                        page_start=index, page_end=index, token_count=10
                    ),
                )
                db.add(chunk)
                await db.flush()
                chunk_ids.append(chunk.id)
            return generation.id, chunk_ids

    async def retrieve(self, project_id=None, **kwargs):
        async with self.sessions() as db:
            return await RetrievalService(db, self.user_id).retrieve(
                "habit formation", project_id=project_id or self.project_id, **kwargs
            )

    async def test_nearest_chunks_first_with_titles_and_labels(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a, chunks=[("A less related passage", vector(0.8, 0.6))])
        await self.index(b)
        result = await self.retrieve()
        self.assertEqual(
            [s.document_title for s in result.sources], ["Book B", "Book A"]
        )
        self.assertEqual([s.citation_label for s in result.sources], ["S1", "S2"])
        self.assertAlmostEqual(result.sources[0].cosine_distance, 0.0)
        self.assertAlmostEqual(result.sources[1].cosine_distance, 0.2, places=5)
        self.assertEqual(result.sources[0].page_start, 0)

    async def test_document_filter_narrows_search(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a)
        await self.index(b)
        result = await self.retrieve(document_ids=[a])
        self.assertEqual({s.document_id for s in result.sources}, {a})

    async def test_only_active_compatible_generations_are_searched(self):
        """Exclude inactive versions and incompatible models without creating two active versions per book."""
        doc = await self.document()
        generation_id, active_ids = await self.index(
            doc, chunks=[("Current content", vector(0.8, 0.2))]
        )
        await self.index(doc, chunks=[("Old perfect match", vector())], active=False)
        incompatible = await self.document("Incompatible model")
        await self.index(incompatible, model="old-embedding-model")
        result = await self.retrieve()
        self.assertEqual([s.chunk_id for s in result.sources], active_ids)
        self.assertEqual(result.sources[0].generation_id, generation_id)

    async def test_other_users_data_is_not_exposed(self):
        with self.assertRaises(NotFoundError):
            await self.retrieve(project_id=self.other_project_id)
        private = await self.document(user_id=self.other_id, project_id=self.project_id)
        await self.index(private)
        self.assertEqual((await self.retrieve()).sources, [])

    async def test_retrieval_snapshot_can_be_checkpointed_and_reloaded(self):
        doc = await self.document()
        await self.index(doc)
        async with self.sessions.begin() as db:
            service = ConversationService(db, self.user_id)
            conversation = await service.create(self.project_id)
            turn = await service.reserve_turn(
                conversation.id, content="Question", client_request_id=uuid4()
            )
            await service.start_run(
                turn.run.id, model="test", pipeline_version="chat-v1"
            )
            run_id = turn.run.id
        result = await self.retrieve()
        async with self.sessions.begin() as db:
            await ConversationService(db, self.user_id).checkpoint(
                run_id,
                stage="retrieved",
                snapshot=result.to_run_snapshot(RunSnapshot()),
                rewritten_query="habit formation",
            )
        async with self.sessions() as db:
            run = await db.scalar(
                select(ConversationRun).where(ConversationRun.id == run_id)
            )
            self.assertTrue(run.snapshot.retrieval_candidates[0].selected)
            self.assertEqual(run.snapshot.selected_sources[0].citation_label, "S1")


if __name__ == "__main__":
    unittest.main()
