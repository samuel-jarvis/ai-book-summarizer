"""Rerank contract, relabelling and fallbacks with a local fake chat model."""

import asyncio
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from langchain_core.documents import Document as LangChainDocument
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.models.conversation import RunSnapshot
from app.services.llm.llm_service import LLMService
from app.services.rerank_service import RerankService, format_passages, validate_ranking
from app.services.retrieval_service import RetrievalResult, to_source


def docs(count):
    return [
        LangChainDocument(
            id=str(uuid4()),
            page_content=f"passage {n}",
            metadata={
                "document_id": str(uuid4()),
                "generation_id": str(uuid4()),
                "document_title": "Book A" if n % 2 else "Book B",
                "chunk_index": n,
                "page_start": n,
                "page_end": n + 1,
                "cosine_distance": 0.1 + n / 100,
            },
        )
        for n in range(count)
    ]


def result(count):
    documents = docs(count)
    sources = [to_source(doc, f"S{n}") for n, doc in enumerate(documents, start=1)]
    return RetrievalResult(documents, sources)


class ValidateRankingTests(unittest.TestCase):
    def test_numbers_become_zero_based_indexes_and_are_cut_to_keep(self):
        self.assertEqual(validate_ranking('{"ranking":[5,2,7,1]}', 8, 3), [4, 1, 6])

    def test_unusable_outputs_are_rejected(self):
        outputs = [
            "",
            "not JSON",
            "[]",
            '```json\n{"ranking":[1]}\n```',
            "{}",
            '{"ranking":[]}',
            '{"ranking":"1"}',
            '{"ranking":["1"]}',
            '{"ranking":[1,1]}',
            '{"ranking":[0]}',
            '{"ranking":[9]}',
            '{"ranking":[1],"answer":"x"}',
        ]
        for output in outputs:
            with self.subTest(output=output[:50]):
                self.assertIsNone(validate_ranking(output, 8, 3))


class RerankTests(unittest.IsolatedAsyncioTestCase):
    def service(self, output):
        llm = LLMService(model="rerank-test", chat_model=FakeListChatModel(responses=[output]))
        llm.generate = AsyncMock(wraps=llm.generate)
        return RerankService(llm)

    async def test_picks_come_first_relabelled_and_the_nearest_fill_the_rest(self):
        original = result(8)
        service = self.service('{"ranking":[5,2]}')
        reranked = await service.rerank("question?", original, keep=3)
        ids = [doc.id for doc in original.documents]
        self.assertEqual([doc.id for doc in reranked.documents], [ids[4], ids[1], ids[0]])
        self.assertEqual([s.citation_label for s in reranked.sources], ["S1", "S2", "S3"])
        self.assertEqual(
            [str(s.chunk_id) for s in reranked.sources], [ids[4], ids[1], ids[0]]
        )
        values = service.llm.generate.call_args.args[1]
        self.assertEqual((values["keep"], values["question"]), (3, "question?"))
        self.assertIn("[1] (Book B, pp. 1-2)\npassage 0", values["passages"])

    async def test_every_original_hit_stays_in_the_snapshot_with_its_selection(self):
        original = result(8)
        reranked = await self.service('{"ranking":[5,2,7]}').rerank("q", original, keep=3)
        snapshot = reranked.to_run_snapshot(RunSnapshot())
        self.assertEqual(len(snapshot.retrieval_candidates), 8)
        selected = [c for c in snapshot.retrieval_candidates if c.selected]
        self.assertEqual(
            {str(c.chunk_id) for c in selected},
            {original.documents[i].id for i in (4, 1, 6)},
        )
        self.assertEqual(
            [s.citation_label for s in snapshot.selected_sources], ["S1", "S2", "S3"]
        )

    async def test_unusable_output_and_provider_errors_fall_back_to_nearest(self):
        original = result(8)
        bad = await self.service("sorry").rerank("q", original, keep=3)
        self.assertEqual([d.id for d in bad.documents], [d.id for d in original.documents[:3]])
        failing = self.service("{}")
        failing.llm.generate = AsyncMock(side_effect=RuntimeError("private account detail"))
        fallback = await failing.rerank("q", original, keep=3)
        self.assertEqual(
            [d.id for d in fallback.documents], [d.id for d in original.documents[:3]]
        )
        self.assertEqual(len(fallback.candidates), 8)

    async def test_short_result_is_returned_unchanged_without_a_model_call(self):
        original = result(3)
        service = self.service('{"ranking":[1]}')
        self.assertIs(await service.rerank("q", original, keep=6), original)
        service.llm.generate.assert_not_awaited()

    async def test_cancellation_propagates(self):
        service = self.service("{}")
        service.llm.generate = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await service.rerank("q", result(8), keep=3)

    def test_passages_carry_book_and_pages(self):
        text = format_passages(docs(2))
        self.assertIn("[2] (Book A, pp. 2-3)\npassage 1", text)


if __name__ == "__main__":
    unittest.main()
