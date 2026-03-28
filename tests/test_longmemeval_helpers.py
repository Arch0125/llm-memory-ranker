import unittest

from benchmarks.longmemeval import (
    build_query_text,
    contains_match_score,
    exact_match_score,
    iter_history_memories,
    normalize_answer,
    selected_session_recall,
    summarize_records,
    token_f1_score,
)
from memory.types import MemoryHit, MemoryRecord


def make_hit(session_id):
    return MemoryHit(
        record=MemoryRecord(
            memory_id=f"m-{session_id}",
            user_id="u1",
            memory_type="ephemeral",
            text="memory",
            created_at="2026-03-28T00:00:00+00:00",
            last_accessed_at="2026-03-28T00:00:00+00:00",
            metadata={"session_id": session_id},
        ),
        score=0.9,
        embedding_model="hash-384",
        age_days=0,
    )


class LongMemEvalHelpersTest(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "question_id": "q1",
            "question_type": "single-session-preference",
            "question": "What food does the user prefer?",
            "answer": "sushi",
            "question_date": "2024-01-10",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024-01-01", "2024-01-05"],
            "haystack_sessions": [
                [{"role": "user", "content": "I prefer sushi.", "has_answer": True}],
                [{"role": "assistant", "content": "You prefer sushi."}],
            ],
            "answer_session_ids": ["s1"],
        }

    def test_iter_history_memories_turn_granularity_keeps_metadata(self):
        memories = list(
            iter_history_memories(
                self.instance,
                granularity="turn",
                include_assistant_turns=True,
            )
        )
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["metadata"]["session_id"], "s1")
        self.assertEqual(memories[0]["memory_type"], "preference")
        self.assertTrue(memories[0]["metadata"]["has_answer"])

    def test_iter_history_memories_session_granularity_collapses_session(self):
        memories = list(
            iter_history_memories(
                self.instance,
                granularity="session",
                include_assistant_turns=True,
            )
        )
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["memory_type"], "ephemeral")
        self.assertIn("Session s1 on 2024-01-01", memories[0]["text"])

    def test_answer_metrics(self):
        self.assertEqual(normalize_answer("The Sushi!"), "sushi")
        self.assertEqual(exact_match_score("Sushi", "the sushi"), 1.0)
        self.assertEqual(contains_match_score("The answer is sushi", "sushi"), 1.0)
        self.assertGreater(token_f1_score("the user prefers sushi", "sushi"), 0.0)

    def test_query_text_includes_question_date(self):
        query = build_query_text(self.instance, include_question_date=True)
        self.assertIn("Question date: 2024-01-10", query)
        self.assertIn("What food does the user prefer?", query)

    def test_selected_session_recall(self):
        hits = [make_hit("s1"), make_hit("s2")]
        self.assertEqual(selected_session_recall(hits, ["s1"]), 1.0)

    def test_summarize_records(self):
        summary = summarize_records(
            [
                {
                    "question_type": "single-session-preference",
                    "exact_match": 1.0,
                    "contains_match": 1.0,
                    "token_f1": 1.0,
                    "abstention_accuracy": None,
                    "selected_memory_count": 2,
                    "selected_session_recall": 1.0,
                },
                {
                    "question_type": "single-session-preference",
                    "exact_match": 0.0,
                    "contains_match": 1.0,
                    "token_f1": 0.5,
                    "abstention_accuracy": None,
                    "selected_memory_count": 1,
                    "selected_session_recall": 0.0,
                },
            ]
        )
        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["by_question_type"]["single-session-preference"]["examples"], 2)
        self.assertAlmostEqual(summary["avg_selected_memory_count"], 1.5)


if __name__ == "__main__":
    unittest.main()
