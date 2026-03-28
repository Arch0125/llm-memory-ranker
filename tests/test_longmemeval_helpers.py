import unittest

from benchmarks.longmemeval import (
    acceptable_answers,
    analyze_question,
    build_query_text,
    contains_match_score,
    exact_match_score,
    iter_history_memories,
    normalize_answer,
    postprocess_prediction,
    select_bundled_hits,
    selected_session_recall,
    summarize_records,
    token_f1_score,
    assess_answerability,
)
from memory.types import MemoryHit, MemoryRecord


def make_hit(
    session_id,
    memory_type="event",
    text="memory",
    event_date="2024-01-01",
    entities=None,
    granularity="fact",
    score=0.9,
    confidence=0.8,
):
    return MemoryHit(
        record=MemoryRecord(
            memory_id=f"m-{session_id}-{memory_type}-{granularity}",
            user_id="u1",
            memory_type=memory_type,
            text=text,
            created_at="2026-03-28T00:00:00+00:00",
            last_accessed_at="2026-03-28T00:00:00+00:00",
            importance=0.8,
            metadata={
                "session_id": session_id,
                "event_date": event_date,
                "entities": entities or [],
                "granularity": granularity,
                "fact_text": text,
            },
        ),
        score=score,
        embedding_model="temporal-hash-512",
        age_days=0,
        critic_label="use",
        critic_confidence=confidence,
    )


class LongMemEvalHelpersTest(unittest.TestCase):
    def setUp(self):
        self.instance = {
            "question_id": "q1",
            "question_type": "temporal-reasoning",
            "question": "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
            "answer": "'Data Analysis using Python' webinar",
            "question_date": "2024/01/10 (Wed) 10:00",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/01/01 (Mon) 09:00", "2024/01/05 (Fri) 09:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "I attended the Data Analysis using Python webinar.", "has_answer": True}],
                [{"role": "user", "content": "I attended the Effective Time Management workshop.", "has_answer": True}],
            ],
            "answer_session_ids": ["s1", "s2"],
        }

    def test_iter_history_memories_turn_granularity_keeps_temporal_metadata(self):
        memories = list(
            iter_history_memories(
                self.instance,
                granularity="turn",
                include_assistant_turns=True,
            )
        )
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["metadata"]["session_id"], "s1")
        self.assertEqual(memories[0]["metadata"]["event_date"], "2024-01-01")
        self.assertEqual(memories[0]["metadata"]["granularity"], "fact")

    def test_iter_history_memories_session_granularity_yields_episode_memories(self):
        memories = list(
            iter_history_memories(
                self.instance,
                granularity="session",
                include_assistant_turns=True,
            )
        )
        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["memory_type"], "episode")
        self.assertIn("Episode summary", memories[0]["text"])

    def test_iter_history_memories_hybrid_emits_global_timeline(self):
        memories = list(
            iter_history_memories(
                self.instance,
                granularity="hybrid",
                include_assistant_turns=True,
            )
        )
        self.assertEqual(memories[0]["metadata"]["granularity"], "timeline-global")
        self.assertIn("Global timeline evidence", memories[0]["text"])

    def test_analyze_question_extracts_targets_and_temporal_kind(self):
        plan = analyze_question(self.instance)
        self.assertTrue(plan.is_temporal)
        self.assertEqual(plan.reasoning_kind, "ordering")
        self.assertEqual(
            plan.targets,
            ["Effective Time Management", "Data Analysis using Python"],
        )

    def test_answer_metrics_support_acceptable_variants(self):
        self.assertEqual(normalize_answer("The Sushi!"), "sushi")
        self.assertEqual(acceptable_answers("7 days. 8 days (including the last day) is also acceptable."), ["7 days", "8 days"])
        self.assertEqual(exact_match_score("8 days", "7 days. 8 days (including the last day) is also acceptable."), 1.0)
        self.assertEqual(contains_match_score("The answer is sushi", "sushi"), 1.0)
        self.assertGreater(token_f1_score("the user prefers sushi", "sushi"), 0.0)

    def test_query_text_includes_normalized_question_date(self):
        query = build_query_text(self.instance, include_question_date=True)
        self.assertIn("Question date: 2024-01-10", query)
        self.assertIn(self.instance["question"], query)

    def test_selected_session_recall(self):
        hits = [make_hit("s1"), make_hit("s2")]
        self.assertEqual(selected_session_recall(hits, ["s1"]), 1.0)

    def test_select_bundled_hits_picks_temporal_target_pair(self):
        plan = analyze_question(self.instance)
        hits = [
            make_hit(
                "s1",
                text="2024-01-01 | Attended the Data Analysis using Python webinar.",
                event_date="2024-01-01",
                entities=["data analysis using python"],
            ),
            make_hit(
                "s2",
                text="2024-01-05 | Attended the Effective Time Management workshop.",
                event_date="2024-01-05",
                entities=["effective time management"],
            ),
            make_hit(
                "global",
                memory_type="timeline",
                granularity="timeline-global",
                text="Global timeline evidence.",
                event_date="2024-01-01",
                entities=["data analysis using python", "effective time management"],
                score=0.95,
            ),
        ]

        selected, _ = select_bundled_hits(plan, hits, max_items=4, max_tokens=400, encode=None)
        answerability = assess_answerability(plan, selected)

        self.assertGreaterEqual(len(selected), 2)
        self.assertTrue(answerability["sufficient"])

    def test_postprocess_prediction_extracts_target_name(self):
        plan = analyze_question(self.instance)
        prediction = "You attended the Data Analysis using Python webinar first on 2024/01/01."
        self.assertEqual(postprocess_prediction(plan, prediction), "Data Analysis using Python")

    def test_summarize_records(self):
        summary = summarize_records(
            [
                {
                    "question_type": "temporal-reasoning",
                    "exact_match": 1.0,
                    "contains_match": 1.0,
                    "token_f1": 1.0,
                    "abstention_accuracy": None,
                    "selected_memory_count": 2,
                    "selected_session_recall": 1.0,
                    "answerable": True,
                },
                {
                    "question_type": "temporal-reasoning",
                    "exact_match": 0.0,
                    "contains_match": 1.0,
                    "token_f1": 0.5,
                    "abstention_accuracy": None,
                    "selected_memory_count": 1,
                    "selected_session_recall": 0.0,
                    "answerable": False,
                },
            ]
        )
        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["by_question_type"]["temporal-reasoning"]["examples"], 2)
        self.assertAlmostEqual(summary["avg_selected_memory_count"], 1.5)
        self.assertAlmostEqual(summary["avg_answerable"], 0.5)


if __name__ == "__main__":
    unittest.main()
