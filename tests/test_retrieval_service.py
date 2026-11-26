"""Retrieval tests: snapshot mapping without a database, then pgvector search.

Set CHAT_TEST_DATABASE_URL to run the PostgreSQL tests. The query embedding is
patched with controlled vectors, so no provider calls are made.
"""

import os
import unittest
from collections import Counter
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
from app.services.indexing.vector_store import merge_balanced, merge_groups
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


def chunk(chunk_id, book, distance):
    return LangChainDocument(
        id=chunk_id,
        page_content=chunk_id,
        metadata={"document_id": book, "cosine_distance": distance},
    )


class MergeBalancedTests(unittest.TestCase):
    def ids(self, docs):
        return [doc.id for doc in docs]

    def test_each_book_keeps_its_best_chunks_and_the_rest_fill_by_distance(self):
        a = [chunk(f"a{i}", "A", 0.1 + i / 100) for i in range(6)]
        b = [chunk("b0", "B", 0.6), chunk("b1", "B", 0.7)]
        merged = merge_balanced(a + b, k=6, min_per_document=2)
        self.assertEqual(self.ids(merged), ["a0", "a1", "a2", "a3", "b0", "b1"])

    def test_zero_minimum_is_plain_nearest_k(self):
        a = [chunk(f"a{i}", "A", 0.1 + i / 100) for i in range(6)]
        b = [chunk("b0", "B", 0.6)]
        self.assertEqual(
            self.ids(merge_balanced(a + b, k=6, min_per_document=0)),
            ["a0", "a1", "a2", "a3", "a4", "a5"],
        )

    def test_book_with_fewer_chunks_than_the_minimum_gets_what_exists(self):
        a = [chunk(f"a{i}", "A", 0.1 + i / 100) for i in range(6)]
        merged = merge_balanced(a + [chunk("b0", "B", 0.9)], k=6, min_per_document=2)
        self.assertEqual(self.ids(merged), ["a0", "a1", "a2", "a3", "a4", "b0"])

    def test_reservations_beyond_k_keep_the_nearest_books_chunks(self):
        docs = [chunk(f"{book}{i}", book, 0.1 * n + i / 100)
                for n, book in enumerate("ABCD") for i in range(2)]
        merged = merge_balanced(docs, k=5, min_per_document=2)
        self.assertEqual(self.ids(merged), ["A0", "A1", "B0", "B1", "C0"])

    def test_duplicates_from_the_two_searches_are_collapsed(self):
        merged = merge_balanced(
            [chunk("a0", "A", 0.1), chunk("a0", "A", 0.1), chunk("b0", "B", 0.5)],
            k=6,
            min_per_document=2,
        )
        self.assertEqual(self.ids(merged), ["a0", "b0"])


class MergeGroupsTests(unittest.TestCase):
    def ids(self, docs):
        return [doc.id for doc in docs]

    def test_each_query_keeps_an_equal_share_of_its_own_nearest(self):
        first = [chunk(f"a{i}", "A", 0.1 + i / 100) for i in range(6)]
        second = [chunk(f"b{i}", "B", 0.5 + i / 100) for i in range(6)]
        merged = merge_groups([first, second], k=6)
        self.assertEqual(self.ids(merged), ["a0", "a1", "a2", "b0", "b1", "b2"])

    def test_a_chunk_found_by_both_queries_frees_a_slot_for_the_next_nearest(self):
        shared = chunk("s0", "A", 0.2)
        first = [shared] + [chunk(f"a{i}", "A", 0.3 + i / 100) for i in range(5)]
        second = [shared] + [chunk(f"b{i}", "B", 0.4 + i / 100) for i in range(5)]
        merged = merge_groups([first, second], k=6)
        self.assertEqual(self.ids(merged), ["s0", "a0", "a1", "a2", "b0", "b1"])

    def test_three_groups_split_the_slots_and_short_groups_are_topped_up(self):
        groups = [
            [chunk("a0", "A", 0.1)],
            [chunk(f"b{i}", "B", 0.2 + i / 100) for i in range(6)],
            [chunk(f"c{i}", "C", 0.3 + i / 100) for i in range(6)],
        ]
        merged = merge_groups(groups, k=6)
        self.assertEqual(self.ids(merged), ["a0", "b0", "b1", "b2", "c0", "c1"])

    def test_more_groups_than_slots_still_returns_at_most_k(self):
        groups = [[chunk(f"g{n}", "A", n / 10)] for n in range(4)]
        self.assertEqual(self.ids(merge_groups(groups, k=2)), ["g0", "g1"])


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

    async def test_every_book_keeps_a_minimum_share_of_the_results(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a, chunks=[(f"A{i}", vector(1.0, 0.01 * i)) for i in range(6)])
        await self.index(b, chunks=[(f"B{i}", vector(0.2, 1.0)) for i in range(3)])
        plain = await self.retrieve(min_per_document=0)
        self.assertEqual({s.document_id for s in plain.sources}, {a})
        balanced = await self.retrieve()
        self.assertEqual(Counter(s.document_id for s in balanced.sources), {a: 4, b: 2})
        distances = [s.cosine_distance for s in balanced.sources]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(
            [s.citation_label for s in balanced.sources],
            [f"S{n}" for n in range(1, 7)],
        )

    async def test_split_question_searches_each_part_on_its_own(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a, chunks=[(f"A{i}", vector(1.0, 0.01 * i)) for i in range(6)])
        await self.index(b, chunks=[(f"B{i}", vector(0.01 * i, 1.0)) for i in range(6)])
        by_query = {"about a": vector(1.0, 0.0), "about b": vector(0.0, 1.0)}
        embed = AsyncMock(side_effect=lambda query: by_query[query])
        with patch("app.services.indexing.vector_store.embed_query", new=embed):
            result = await self.retrieve(queries=["about a", "about b"])
        self.assertEqual(Counter(s.document_id for s in result.sources), {a: 3, b: 3})
        self.assertEqual(result.queries, ["about a", "about b"])
        self.assertEqual(embed.await_count, 2)
        distances = [s.cosine_distance for s in result.sources]
        self.assertEqual(distances, sorted(distances))
        snapshot = result.to_run_snapshot(RunSnapshot())
        self.assertEqual(snapshot.search_queries, ["about a", "about b"])

    async def test_one_query_list_uses_the_normal_search_and_records_the_query(self):
        a = await self.document("Book A")
        await self.index(a)
        result = await self.retrieve(queries=["only one"])
        self.assertEqual(result.queries, ["habit formation"])
        self.assertEqual(len(result.sources), 1)

    async def test_single_book_project_is_unchanged_by_balancing(self):
        a = await self.document("Book A")
        await self.index(a, chunks=[(f"A{i}", vector(1.0, 0.1 * i)) for i in range(4)])
        plain = await self.retrieve(min_per_document=0)
        balanced = await self.retrieve(min_per_document=2)
        self.assertEqual(
            [s.chunk_id for s in balanced.sources], [s.chunk_id for s in plain.sources]
        )

    async def test_book_with_a_single_chunk_contributes_just_that_chunk(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a, chunks=[(f"A{i}", vector(1.0, 0.01 * i)) for i in range(6)])
        await self.index(b, chunks=[("B0", vector(0.2, 1.0))])
        counts = Counter(s.document_id for s in (await self.retrieve()).sources)
        self.assertEqual(counts, {a: 5, b: 1})

    async def test_document_filter_still_applies_with_balancing(self):
        a, b = await self.document("Book A"), await self.document("Book B")
        await self.index(a, chunks=[(f"A{i}", vector(1.0, 0.01 * i)) for i in range(4)])
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
