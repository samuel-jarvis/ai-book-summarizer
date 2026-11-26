"""Question-splitting contract and fallbacks with a local fake chat model."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.services.llm.llm_service import LLMService
from app.services.query_plan_service import QueryPlanService, validate_plan


class ValidatePlanTests(unittest.TestCase):
    """Reject unusable output; keep clean, distinct queries."""

    def test_queries_are_trimmed_deduplicated_and_capped(self):
        output = json.dumps({"queries": [" A one ", "a ONE", "B two", "C three", "D four"]})
        self.assertEqual(validate_plan(output, 3), ["A one", "B two", "C three"])

    def test_unusable_outputs_are_rejected(self):
        outputs = [
            "",
            "not JSON",
            "[]",
            '```json\n{"queries":["Q"]}\n```',
            "{}",
            '{"queries":[]}',
            '{"queries":"Q"}',
            '{"queries":[5]}',
            '{"queries":["Q"],"answer":"A"}',
            json.dumps({"queries": ["ok", ""]}),
            json.dumps({"queries": ["ok", "???"]}),
            json.dumps({"queries": ["two\nlines"]}),
            json.dumps({"queries": ["x" * 501]}),
            json.dumps({"queries": [f"q{i}" for i in range(11)]}),
        ]
        for output in outputs:
            with self.subTest(output=output[:60]):
                self.assertIsNone(validate_plan(output, 3))


class QueryPlanTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the real prompt rendering through the LLM abstraction, no provider."""

    def service(self, output):
        llm = LLMService(model="plan-test", chat_model=FakeListChatModel(responses=[output]))
        llm.generate = AsyncMock(wraps=llm.generate)
        return QueryPlanService(llm)

    async def test_multi_part_question_becomes_several_queries(self):
        service = self.service('{"queries":["Time Machine old books","Griffin books Marvel"]}')
        plan = await service.plan("Compare the books in both stories", max_queries=3)
        self.assertEqual(plan.queries, ["Time Machine old books", "Griffin books Marvel"])
        self.assertFalse(plan.fallback)
        self.assertEqual(plan.model, "plan-test")
        values = service.llm.generate.call_args.args[1]
        self.assertEqual(values, {"question": "Compare the books in both stories", "max_queries": 3})

    async def test_single_topic_keeps_the_original_wording(self):
        service = self.service('{"queries":["a paraphrase"]}')
        plan = await service.plan("What colour was the lamp?", max_queries=3)
        self.assertEqual(plan.queries, ["What colour was the lamp?"])
        self.assertFalse(plan.fallback)

    async def test_bad_output_and_provider_errors_fall_back_to_the_question(self):
        bad = await self.service("sorry").plan("Q about books", max_queries=3)
        self.assertEqual((bad.queries, bad.fallback), (["Q about books"], True))
        failing = self.service("{}")
        failing.llm.generate = AsyncMock(side_effect=RuntimeError("private account detail"))
        plan = await failing.plan("Q about books", max_queries=3)
        self.assertEqual((plan.queries, plan.fallback), (["Q about books"], True))

    async def test_limit_of_one_skips_the_model_call(self):
        service = self.service('{"queries":["a","b"]}')
        plan = await service.plan("Q", max_queries=1)
        self.assertEqual((plan.queries, plan.model, plan.fallback), (["Q"], None, False))
        service.llm.generate.assert_not_awaited()

    async def test_cancellation_propagates(self):
        service = self.service("{}")
        service.llm.generate = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await service.plan("Q about books", max_queries=3)


if __name__ == "__main__":
    unittest.main()
