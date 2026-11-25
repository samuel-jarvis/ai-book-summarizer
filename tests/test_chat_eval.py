"""Chat benchmark judge and summary, with a fake model so no provider is called."""

import json
import unittest

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.services.llm.llm_service import LLMService
from evals.chat.cases import Case
from evals.chat.compare import case_changes, metric_rows, warnings
from evals.chat.judge import AnswerVerdict, Judge, JudgeError, parse, render
from evals.chat.run_info import cases_hash
from evals.chat.summary import format_summary, snapshot, summarize

ANSWER_CASE = Case.model_validate({
    "id": "a", "type": "follow_up", "prior_turns": ["Who is Weena?"],
    "question": "What did she give him?", "expected_behavior": "answer",
    "required_facts": [{
        "fact": "A garland of flowers.",
        "evidence": [{"document": "tm.pdf", "pdf_page": 3, "quote": "a garland"}],
    }],
})
ABSTAIN_CASE = Case.model_validate({
    "id": "b", "type": "unanswerable", "prior_turns": [], "question": "Which day?",
    "expected_behavior": "abstain", "required_facts": [],
    "abstention_reason": "The book gives the year but not the day.",
})
SOURCES = [
    {"citation_label": "S1", "document": "tm.pdf", "pdf_page_start": 3, "pdf_page_end": 4,
     "text": "She gave me a garland."},
    {"citation_label": None, "document": "tm.pdf", "pdf_page_start": 9, "pdf_page_end": 9,
     "text": "NOT IN PROMPT"},
]


def judge_replying(reply: str) -> Judge:
    model = FakeListChatModel(responses=[reply])
    return Judge(llm=LLMService(model="fake-judge", chat_model=model))


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_answer_case_fills_both_scores(self):
        reply = 'Sure!\n```json\n{"correct": true, "supported": false, "reason": "ok"}\n```'
        scores = await judge_replying(reply).grade(ANSWER_CASE, "A garland [S1].", SOURCES)
        self.assertIs(scores["answer_correct"], True)
        self.assertIs(scores["citations_support_claims"], False)
        self.assertEqual(scores["judge"], {"model": "fake-judge", "reason": "ok"})

    async def test_abstain_case_scores_the_decline(self):
        reply = '{"abstained": false, "reason": "invented a date"}'
        scores = await judge_replying(reply).grade(ABSTAIN_CASE, "It was 12 May.", SOURCES)
        self.assertIs(scores["answer_correct"], False)
        self.assertIsNone(scores["citations_support_claims"])

    async def test_unusable_reply_raises(self):
        with self.assertRaises(JudgeError):
            await judge_replying("I think it is fine.").grade(ANSWER_CASE, "x", SOURCES)
        with self.assertRaises(JudgeError):
            await judge_replying('{"correct": "maybe"}').grade(ANSWER_CASE, "x", SOURCES)

    def test_prompt_shows_only_selected_passages_and_the_whole_conversation(self):
        _, values = render(ANSWER_CASE, "A garland [S1].", SOURCES)
        self.assertIn("[S1] tm.pdf pp. 3-4", values["passages"])
        self.assertNotIn("NOT IN PROMPT", values["passages"])
        self.assertIn("Who is Weena?", values["conversation"])
        self.assertIn("A garland of flowers.", values["facts"])

    def test_parse_ignores_text_around_the_object(self):
        verdict = parse('Verdict: {"correct": true, "supported": true} done', AnswerVerdict)
        self.assertTrue(verdict.correct)


def row(kind, retrieval=None, prompt=None, citations=None, correct=None, failed=False, case_id=None):
    scores = None if failed else {
        "retrieval": retrieval and {"found": retrieval[0], "total": retrieval[1]},
        "prompt_evidence": prompt and {"found": prompt[0], "total": prompt[1]},
        "citation_labels_valid": citations,
        "answer_correct": correct,
        "citations_support_claims": None,
    }
    return {
        "case": {"id": case_id or kind, "type": kind},
        "scores": scores,
        "last_turn_sources": [],
    }


class SummaryTests(unittest.TestCase):
    def test_counts_by_type_and_overall(self):
        rows = [
            row("direct", (1, 1), (1, 1), True, True),
            row("direct", (0, 1), (0, 1), True, False),
            row("unanswerable", None, None, True, True),
            row("comparison", failed=True),
        ]
        summary = summarize(rows)
        self.assertEqual(summary["direct"]["retrieval"], (1, 2))
        self.assertEqual(summary["direct"]["correct"], (1, 2))
        self.assertEqual(summary["unanswerable"]["retrieval"], (0, 0))
        self.assertEqual(summary["comparison"]["failed"], 1)
        self.assertEqual(summary["all"]["cases"], 4)
        self.assertEqual(summary["all"]["citations"], (3, 3))

    def test_unjudged_reports_render_as_na(self):
        text = format_summary(summarize([row("direct", (1, 1), (1, 1), True, None)]))
        self.assertIn("1/1 100%", text)
        self.assertIn("N/A", text)


class TrackingTests(unittest.TestCase):
    def snap(self, rows, name, **run):
        rows = [{**r, "run": run} for r in rows]
        return snapshot(rows, name, note="n")

    def test_snapshot_keeps_scores_and_run_facts_only(self):
        rows = [row("direct", (1, 1), (1, 1), True, True, case_id="d1"), row("comparison", failed=True)]
        snap = self.snap(rows, "x", git_commit="abc")
        self.assertEqual(snap["run"]["git_commit"], "abc")
        self.assertEqual(snap["run"]["note"], "n")
        self.assertEqual(snap["cases"]["d1"]["retrieval"], "1/1")
        self.assertEqual(snap["cases"]["comparison"]["status"], "failed")
        self.assertNotIn("last_turn_sources", json.dumps(snap))

    def test_compare_reports_points_and_flipped_cases(self):
        before = self.snap([row("comparison", (1, 4), (1, 4), True, False, case_id="c1")], "a")
        after = self.snap([row("comparison", (3, 4), (3, 4), True, True, case_id="c1")], "b")
        after = json.loads(json.dumps(after))  # as saved to and loaded from disk
        lines = metric_rows(before, after)
        self.assertTrue(any("retrieval" in line and "+50 pts" in line for line in lines))
        self.assertEqual(
            case_changes(before, after),
            ["c1 (comparison): retrieval 1/4 -> 3/4; in_prompt 1/4 -> 3/4; correct False -> True"],
        )

    def test_compare_warns_when_labels_or_tree_differ(self):
        before = self.snap([row("direct", (1, 1), (1, 1), True, True, case_id="d")], "a", cases_hash="1")
        after = self.snap(
            [row("direct", (1, 1), (1, 1), True, True, case_id="d")], "b", cases_hash="2", git_dirty=True
        )
        notes = warnings(before, after)
        self.assertEqual(len(notes), 2)

    def test_cases_hash_changes_with_the_labels(self):
        self.assertEqual(cases_hash([ANSWER_CASE]), cases_hash([ANSWER_CASE]))
        self.assertNotEqual(cases_hash([ANSWER_CASE]), cases_hash([ABSTAIN_CASE]))


if __name__ == "__main__":
    unittest.main()
