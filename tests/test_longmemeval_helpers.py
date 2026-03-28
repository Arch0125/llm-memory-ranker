import unittest

from benchmarks.longmemeval import (
    acceptable_answers,
    analyze_question,
    build_structured_event_view,
    build_query_text,
    contains_match_score,
    exact_match_score,
    iter_history_memories,
    normalize_answer,
    postprocess_prediction,
    solve_temporal_question,
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
    date_confidence=0.9,
    date_source="explicit-date",
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
                "event_dates": [event_date] if event_date else [],
                "entities": entities or [],
                "granularity": granularity,
                "fact_text": text,
                "event_aliases": entities or [],
                "date_confidence": date_confidence,
                "date_source": date_source,
                "event_items": [
                    {
                        "label": text,
                        "normalized_label": normalize_answer(text),
                        "aliases": entities or [],
                        "activity_type": memory_type,
                        "event_date": event_date,
                        "date_source": date_source,
                        "date_confidence": date_confidence,
                        "date_candidates": (
                            [{"date": event_date, "source": date_source, "confidence": date_confidence}]
                            if event_date
                            else []
                        ),
                        "session_id": session_id,
                        "turn_index": 0,
                        "role": "user",
                        "has_answer": True,
                    }
                ],
            },
        ),
        score=score,
        embedding_model="temporal-hash-512",
        age_days=0,
        critic_label="use",
        critic_confidence=confidence,
    )


def make_aggregate_hit(
    text="Aggregate memory",
    aggregate_kind="count_distinct",
    aggregate_answer="2",
    aggregate_confidence=0.86,
    entries=None,
    score=0.95,
    confidence=0.9,
):
    return MemoryHit(
        record=MemoryRecord(
            memory_id=f"agg-{aggregate_kind}",
            user_id="u1",
            memory_type="aggregate",
            text=text,
            created_at="2026-03-28T00:00:00+00:00",
            last_accessed_at="2026-03-28T00:00:00+00:00",
            importance=0.95,
            metadata={
                "granularity": "aggregate",
                "summary": text,
                "aggregate_kind": aggregate_kind,
                "aggregate_answer": aggregate_answer,
                "aggregate_confidence": aggregate_confidence,
                "aggregate_mode": f"multi-session-{aggregate_kind}",
                "aggregate_entries": entries or [],
                "event_aliases": [],
                "entities": [],
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
        self.assertIn("event_items", memories[0]["metadata"])
        self.assertIn("date_source", memories[0]["metadata"])

    def test_iter_history_memories_turn_granularity_prefers_explicit_turn_date(self):
        instance = {
            "question_id": "q-explicit",
            "question_type": "temporal-reasoning",
            "question": "When did I attend the workshop?",
            "answer": "2023-01-10",
            "question_date": "2023/01/20 (Fri) 10:00",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/01/13 (Fri) 18:07"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I attended the workshop on January 10th before preparing for the team meeting.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-01-10")
        self.assertIn("2023-01-10", memories[0]["text"])

    def test_iter_history_memories_turn_granularity_derives_relative_date(self):
        instance = {
            "question_id": "q-relative",
            "question_type": "temporal-reasoning",
            "question": "When did I buy the coffee maker?",
            "answer": "2023-05-01",
            "question_date": "2023/05/25 (Thu) 10:00",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/22 (Mon) 09:38"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I bought the coffee maker about three weeks ago and have been using it every morning since then.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-05-01")

    def test_iter_history_memories_turn_granularity_derives_article_relative_date(self):
        instance = {
            "question_id": "q-relative-article",
            "question_type": "temporal-reasoning",
            "question": "When did I set up the smart thermostat?",
            "answer": "2023-04-24",
            "question_date": "2023/05/24 (Wed) 10:00",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/24 (Wed) 02:06"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "Also, since I set up my smart thermostat a month ago, I've noticed that it's been learning my schedule.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-04-24")

    def test_iter_history_memories_turn_granularity_keeps_target_terms_in_fact_text(self):
        instance = {
            "question_id": "q-target-terms",
            "question_type": "temporal-reasoning",
            "question": "Which seeds were started first, the tomatoes or the marigolds?",
            "answer": "Tomatoes",
            "question_date": "2023/03/10 (Fri) 08:29",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/03/10 (Fri) 00:33"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I'm planning to plant out my seedlings soon and I'm wondering if you can tell me the average soil temperature for my area. By the way, I've been starting seeds indoors under grow lights in my basement since February 20th - tomatoes, peppers, and cucumbers are all doing well, about 2-3 inches tall now.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertIn("tomatoes", memories[0]["text"].lower())
        self.assertIn("tomatoes", memories[0]["metadata"]["fact_text"].lower())
        self.assertIn("tomatoes", memories[0]["metadata"]["event_aliases"])

    def test_iter_history_memories_turn_granularity_prefers_relative_anchor_date(self):
        instance = {
            "question_id": "q-relative-anchor",
            "question_type": "temporal-reasoning",
            "question": "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
            "answer": "7 days",
            "question_date": "2023/12/10 (Sun) 23:13",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/12/10 (Sun) 23:13"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I attended the annual Holiday Market at the local mall a week before Black Friday, and I saw some unique handmade jewelry there.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-11-17")
        self.assertEqual(memories[0]["metadata"]["date_source"], "relative-before-black-friday")

    def test_iter_history_memories_turn_granularity_extracts_got_purchase_target(self):
        instance = {
            "question_id": "q-got-purchase",
            "question_type": "temporal-reasoning",
            "question": "When did I get the iPhone 13 Pro?",
            "answer": "2023-11-24",
            "question_date": "2023/12/10 (Sun) 23:13",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/12/10 (Sun) 11:49"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I'm looking to upgrade my phone case and screen protector. By the way, I got my iPhone 13 Pro at a discounted price of $800 from Best Buy on Black Friday.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-11-24")
        self.assertIn("iphone 13 pro", " ".join(memories[0]["metadata"]["event_aliases"]))

    def test_iter_history_memories_turn_granularity_parses_day_of_month(self):
        instance = {
            "question_id": "q-day-month",
            "question_type": "temporal-reasoning",
            "question": "When did I attend the BBQ party?",
            "answer": "June 3",
            "question_date": "2023/07/01 (Sat) 22:22",
            "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/07/01 (Sat) 22:22"],
            "haystack_sessions": [[
                {
                    "role": "user",
                    "content": "I attended a backyard BBQ party at my colleague's house on the 3rd of June, and they had an amazing selection of BBQ sauces.",
                    "has_answer": True,
                }
            ]],
            "answer_session_ids": ["s1"],
        }
        memories = list(iter_history_memories(instance, granularity="turn", include_assistant_turns=False))
        self.assertEqual(memories[0]["metadata"]["event_date"], "2023-06-03")

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

    def test_analyze_question_keeps_date_questions_as_date_reasoning(self):
        instance = {
            "question_id": "q-date-kind",
            "question_type": "temporal-reasoning",
            "question": "What was the date on which I attended the first BBQ event in June?",
            "answer": "June 3rd",
            "question_date": "2024/06/20 (Thu) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        self.assertEqual(plan.reasoning_kind, "date")
        self.assertEqual(plan.filter_month, "06")

    def test_analyze_question_multi_session_extracts_plan(self):
        instance = {
            "question_id": "q-multi",
            "question_type": "multi-session",
            "question": "How many different doctors have I visited in the last two months?",
            "answer": "2",
            "question_date": "2024/04/01 (Mon) 09:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        self.assertTrue(plan.is_multi_session)
        self.assertFalse(plan.is_temporal)
        self.assertEqual(plan.reasoning_kind, "multi-session")
        self.assertEqual(plan.multi_session_kind, "count_distinct")
        self.assertIn("doctor", plan.multi_session_focus_terms)

    def test_answer_metrics_support_acceptable_variants(self):
        self.assertEqual(normalize_answer("The Sushi!"), "sushi")
        self.assertEqual(acceptable_answers("7 days. 8 days (including the last day) is also acceptable."), ["7 days", "8 days"])
        self.assertEqual(acceptable_answers(3), ["3"])
        self.assertEqual(exact_match_score("8 days", "7 days. 8 days (including the last day) is also acceptable."), 1.0)
        self.assertEqual(exact_match_score("3", 3), 1.0)
        self.assertEqual(exact_match_score("June 3", "June 3rd"), 1.0)
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

    def test_build_structured_event_view_prefers_target_facts(self):
        plan = analyze_question(self.instance)
        hits = [
            make_hit(
                "s1",
                text="2024-01-01 | Attended the Data Analysis using Python webinar.",
                event_date="2024-01-01",
                entities=["data analysis using python", "data analysis using python webinar"],
            ),
            make_hit(
                "s2",
                text="2024-01-05 | Attended the Effective Time Management workshop.",
                event_date="2024-01-05",
                entities=["effective time management", "effective time management workshop"],
            ),
        ]
        events = build_structured_event_view(plan, hits, limit=4)
        self.assertEqual(len(events), 2)
        self.assertIn("data analysis using python", events[0]["aliases"])

    def test_solve_temporal_question_ordering(self):
        plan = analyze_question(self.instance)
        hits = [
            make_hit(
                "s1",
                text="2024-01-01 | Attended the Data Analysis using Python webinar.",
                event_date="2024-01-01",
                entities=["data analysis using python", "data analysis using python webinar"],
            ),
            make_hit(
                "s2",
                text="2024-01-05 | Attended the Effective Time Management workshop.",
                event_date="2024-01-05",
                entities=["effective time management", "effective time management workshop"],
            ),
        ]
        result = solve_temporal_question(plan, hits)
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer, "Data Analysis using Python")

    def test_solve_temporal_question_difference_uses_question_date_for_ago(self):
        instance = {
            "question_id": "q-ago",
            "question_type": "temporal-reasoning",
            "question": "How many months ago did I book the Airbnb in San Francisco?",
            "answer": "2 months",
            "question_date": "2024/03/20 (Wed) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        hits = [
            make_hit(
                "s1",
                text="2024-01-05 | I booked the Airbnb in San Francisco.",
                event_date="2024-01-05",
                entities=["book the airbnb in san francisco", "airbnb in san francisco"],
            )
        ]
        result = solve_temporal_question(plan, hits)
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer, "2 months")

    def test_solve_temporal_question_difference_uses_black_friday_anchor(self):
        instance = {
            "question_id": "q-black-friday",
            "question_type": "temporal-reasoning",
            "question": "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
            "answer": "7 days",
            "question_date": "2023/12/10 (Sun) 23:13",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        hits = [
            make_hit(
                "s1",
                text="2023-11-17 | I attended the annual Holiday Market at the local mall.",
                event_date="2023-11-17",
                entities=["holiday market"],
                date_confidence=0.86,
                date_source="relative-before-black-friday",
            ),
            make_hit(
                "s2",
                text="2023-11-24 | I got my iPhone 13 Pro from Best Buy on Black Friday.",
                event_date="2023-11-24",
                entities=["iphone 13 pro"],
                date_confidence=0.88,
                date_source="named-anchor-black-friday",
            ),
        ]
        result = solve_temporal_question(plan, hits)
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer, "7 days")

    def test_postprocess_prediction_extracts_target_name(self):
        plan = analyze_question(self.instance)
        prediction = "You attended the Data Analysis using Python webinar first on 2024/01/01."
        self.assertEqual(postprocess_prediction(plan, prediction), "Data Analysis using Python")

    def test_postprocess_prediction_extracts_target_with_internal_article(self):
        instance = {
            "question_id": "q4",
            "question_type": "temporal-reasoning",
            "question": "Which event happened first, the purchase of the coffee maker or the malfunction of the stand mixer?",
            "answer": "The malfunction of the stand mixer",
            "question_date": "2023/05/25 (Thu) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        prediction = "Based on the evidence, the purchase of the coffee maker happened first."
        self.assertEqual(postprocess_prediction(plan, prediction), "purchase of the coffee maker")

    def test_postprocess_prediction_avoids_year_as_duration(self):
        instance = {
            "question_id": "q2",
            "question_type": "temporal-reasoning",
            "question": "How many days before the team meeting did I attend the workshop?",
            "answer": "7 days",
            "question_date": "2024/01/10 (Wed) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        prediction = (
            "On 2023-01-13, you mentioned preparing for the meeting.\n"
            "Since there is no specific date for the workshop attendance, it is not possible to determine the exact number of days."
        )
        self.assertEqual(
            postprocess_prediction(plan, prediction),
            "Insufficient evidence",
        )

    def test_postprocess_prediction_extracts_month_day_date(self):
        instance = {
            "question_id": "q3",
            "question_type": "temporal-reasoning",
            "question": "What was the date on which I attended the first BBQ event in June?",
            "answer": "June 3rd",
            "question_date": "2024/06/20 (Thu) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        prediction = "You attended the first BBQ event in June on June 3rd."
        self.assertEqual(postprocess_prediction(plan, prediction), "June 3")

    def test_postprocess_prediction_extracts_day_month_date(self):
        instance = {
            "question_id": "q3b",
            "question_type": "temporal-reasoning",
            "question": "What was the date on which I attended the first BBQ event in June?",
            "answer": "June 3rd",
            "question_date": "2024/06/20 (Thu) 10:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        prediction = "You attended the first BBQ event in June on the 3rd of June."
        self.assertEqual(postprocess_prediction(plan, prediction), "June 3")

    def test_answerability_allows_global_timeline_backstop(self):
        plan = analyze_question(self.instance)
        selected = [
            make_hit(
                "global",
                memory_type="timeline",
                granularity="timeline-global",
                text="Global timeline evidence.",
                event_date="2024-01-01",
                entities=["data analysis using python", "effective time management workshop"],
                score=0.95,
            ),
            make_hit(
                "s1",
                text="2024-01-01 | Attended the Data Analysis using Python webinar.",
                event_date="2024-01-01",
                entities=["data analysis using python webinar"],
            ),
            make_hit(
                "s2",
                text="2024-01-05 | Attended the Effective Time Management workshop.",
                event_date="2024-01-05",
                entities=["effective time management workshop"],
            ),
        ]
        answerability = assess_answerability(plan, selected)
        self.assertTrue(answerability["sufficient"])

    def test_iter_history_memories_hybrid_builds_multi_session_aggregate_memory(self):
        instance = {
            "question_id": "q-ms-agg",
            "question_type": "multi-session",
            "question": "How many different doctors have I visited in the last two months?",
            "answer": "2",
            "question_date": "2024/04/01 (Mon) 09:00",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/03/12 (Tue) 10:00", "2024/03/22 (Fri) 11:00"],
            "haystack_sessions": [
                [{"role": "user", "content": "I visited Dr. Patel for a persistent cough.", "has_answer": True}],
                [{"role": "user", "content": "I had an appointment with Dr. Chen for my skin rash.", "has_answer": True}],
            ],
            "answer_session_ids": ["s1", "s2"],
        }
        memories = list(iter_history_memories(instance, granularity="hybrid", include_assistant_turns=False))
        aggregate = next(memory for memory in memories if memory["memory_type"] == "aggregate")
        self.assertEqual(aggregate["metadata"]["aggregate_kind"], "count_distinct")
        self.assertEqual(aggregate["metadata"]["aggregate_answer"], "2")

    def test_select_bundled_hits_prefers_multi_session_aggregate(self):
        instance = {
            "question_id": "q-ms-select",
            "question_type": "multi-session",
            "question": "How many different doctors have I visited in the last two months?",
            "answer": "2",
            "question_date": "2024/04/01 (Mon) 09:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        aggregate = make_aggregate_hit(
            text="Aggregate memory for multi-session doctor visits.",
            aggregate_kind="count_distinct",
            aggregate_answer="2",
            aggregate_confidence=0.86,
            entries=[
                {"entry_id": "e1", "session_id": "s1", "event_date": "2024-03-12"},
                {"entry_id": "e2", "session_id": "s2", "event_date": "2024-03-22"},
            ],
        )
        facts = [
            make_hit("s1", text="2024-03-12 | I visited Dr. Patel for a cough.", entities=["dr patel", "doctor"]),
            make_hit("s2", text="2024-03-22 | I saw Dr. Chen about a rash.", entities=["dr chen", "doctor"]),
        ]
        selected, _ = select_bundled_hits(plan, [facts[0], aggregate, facts[1]], max_items=5, max_tokens=400, encode=None)
        answerability = assess_answerability(plan, selected)
        self.assertEqual(selected[0].record.memory_type, "aggregate")
        self.assertTrue(answerability["sufficient"])

    def test_solve_temporal_question_multi_session_uses_aggregate_answer(self):
        instance = {
            "question_id": "q-ms-money",
            "question_type": "multi-session",
            "question": "What is the total amount I spent on bike accessories this year?",
            "answer": "$150",
            "question_date": "2024/12/31 (Tue) 09:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        aggregate = make_aggregate_hit(
            text="Aggregate memory for bike accessory spending.",
            aggregate_kind="sum_money",
            aggregate_answer="$150",
            aggregate_confidence=0.88,
            entries=[
                {"entry_id": "e1", "session_id": "s1", "event_date": "2024-02-10"},
                {"entry_id": "e2", "session_id": "s2", "event_date": "2024-05-03"},
            ],
        )
        result = solve_temporal_question(plan, [aggregate])
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer, "$150")

    def test_solve_temporal_question_multi_session_falls_back_to_fact_entries(self):
        instance = {
            "question_id": "q-ms-fallback",
            "question_type": "multi-session",
            "question": "How many different doctors have I visited in the last two months?",
            "answer": "2",
            "question_date": "2024/04/01 (Mon) 09:00",
            "haystack_session_ids": [],
            "haystack_dates": [],
            "haystack_sessions": [],
            "answer_session_ids": [],
        }
        plan = analyze_question(instance)
        hits = [
            make_hit(
                "s1",
                text="2024-03-12 | I visited Dr. Patel for a persistent cough.",
                event_date="2024-03-12",
                entities=["doctor", "dr patel"],
            ),
            make_hit(
                "s2",
                text="2024-03-22 | I had an appointment with Dr. Chen for my skin rash.",
                event_date="2024-03-22",
                entities=["doctor", "dr chen"],
            ),
        ]
        result = solve_temporal_question(plan, hits)
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer, "2")

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
