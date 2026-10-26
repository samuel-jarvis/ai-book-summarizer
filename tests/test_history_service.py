"""History budgeting and SQL scope checks without a database or inference calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy.dialects import postgresql

from app.core.exception import ValidationError
from app.models.conversation import PromptMessageSnapshot
from app.services.history_service import (
    MEMORY_PREFIX,
    HistoryService,
    budget_history,
    message_tokens,
)


class HistoryBudgetTests(unittest.TestCase):
    """Check whole-pair selection using the configured tokenizer."""

    def test_newest_pairs_fit_and_remain_chronological(self):
        """Older turns are dropped before newer turns and no pair is split."""
        exchanges = [("old question", "old answer"), ("new question", "new answer")]
        budget = sum(
            message_tokens(PromptMessageSnapshot(role="human", content=text))
            for text in exchanges[-1]
        )
        history = budget_history(exchanges, token_budget=budget)
        self.assertEqual([m.content for m in history.messages], list(exchanges[-1]))
        self.assertEqual([m.role for m in history.messages], ["human", "ai"])
        self.assertEqual(history.token_count, budget)

    def test_memory_and_pairs_share_one_budget(self):
        """Saved memory consumes the same budget as the recent exchanges."""
        history = budget_history(
            [("Q", "A")] * 20, memory="Book B is second.", token_budget=100
        )
        self.assertTrue(history.messages[0].content.startswith(MEMORY_PREFIX))
        self.assertLessEqual(history.token_count, 100)
        self.assertEqual(
            history.token_count, sum(message_tokens(m) for m in history.messages)
        )
        self.assertEqual((len(history.messages) - 1) % 2, 0)

    def test_oversized_memory_is_omitted_whole(self):
        """Avoid cutting a saved statement into a misleading partial fact."""
        history = budget_history(
            [("Q", "A")], memory="large memory " * 100, token_budget=100
        )
        self.assertEqual([m.content for m in history.messages], ["Q", "A"])

    def test_oversized_newest_pair_does_not_pull_in_stale_turns(self):
        """An oversized latest exchange prevents disconnected older context."""
        history = budget_history(
            [("Q", "A"), ("big " * 200, "answer")], token_budget=50
        )
        self.assertEqual(history.messages, [])

    def test_empty_pairs_zero_budget_and_negative_budget(self):
        """Empty messages are excluded and invalid budgets fail explicitly."""
        history = budget_history([("\n", "A"), ("Q", " \t")])
        self.assertEqual(history.messages, [])
        self.assertEqual(budget_history([("Q", "A")], token_budget=0).token_count, 0)
        with self.assertRaises(ValidationError):
            budget_history([], token_budget=-1)

    def test_special_token_text_is_treated_as_user_text(self):
        """User text resembling tokenizer special tokens must not raise."""
        history = budget_history([("What is <|endoftext|>?", "A marker.")])
        self.assertEqual(len(history.messages), 2)


class HistoryLoadTests(unittest.IsolatedAsyncioTestCase):
    """Inspect the loader's bounded owner-scoped SQL and memory boundaries."""

    async def test_completed_pairs_are_loaded_newest_first_then_reordered(self):
        """Only completed, nonempty pairs before the current turn are queried."""
        db = AsyncMock()
        rows = MagicMock()
        rows.all.return_value = [("new Q", "new A"), ("old Q", "old A")]
        db.execute.return_value = rows
        conversation = SimpleNamespace(
            memory_summary=None, memory_through_position=None
        )
        conversation_id, user_id = uuid4(), uuid4()
        with patch(
            "app.services.history_service.ConversationService.get",
            new=AsyncMock(return_value=conversation),
        ) as get:
            history = await HistoryService(db, user_id).load(
                conversation_id, before_position=7, max_exchanges=2
            )
        get.assert_awaited_once_with(conversation_id)
        self.assertEqual(
            [m.content for m in history.messages], ["old Q", "old A", "new Q", "new A"]
        )
        statement = db.execute.call_args.args[0].compile(dialect=postgresql.dialect())
        self.assertIn("conversation_runs.status =", str(statement))
        self.assertIn("btrim", str(statement))
        self.assertIn("ORDER BY", str(statement))
        self.assertIn(7, statement.params.values())
        self.assertIn(2, statement.params.values())

    async def test_future_or_missing_memory_boundary_is_ignored(self):
        """Memory without a valid earlier boundary cannot enter the prompt."""
        for through in (None, 5, 7):
            with self.subTest(through=through):
                db = AsyncMock()
                db.execute.return_value = MagicMock()
                db.execute.return_value.all.return_value = []
                conversation = SimpleNamespace(
                    memory_summary="Future topic", memory_through_position=through
                )
                with patch(
                    "app.services.history_service.ConversationService.get",
                    new=AsyncMock(return_value=conversation),
                ):
                    history = await HistoryService(db, uuid4()).load(
                        uuid4(), before_position=5
                    )
                self.assertEqual(history.messages, [])

    async def test_invalid_limits_are_rejected_before_querying(self):
        """Negative budgets and invalid positions or limits fail before DB work."""
        for options in (
            {"before_position": 0},
            {"token_budget": -1},
            {"max_exchanges": 0},
            {"max_exchanges": 201},
        ):
            with self.subTest(options=options):
                db = AsyncMock()
                with self.assertRaises(ValidationError):
                    await HistoryService(db, uuid4()).load(
                        uuid4(), **({"before_position": 1} | options)
                    )
                db.execute.assert_not_awaited()
