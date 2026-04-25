"""Smoke tests for the generic memory engine refactor.

These exercise the new modules without depending on benchmark data:

- `memory.fusion` (RRF + weighted)
- `memory.bm25` and `memory.expansion`
- `memory.cache`
- `memory.granularity`
- `memory.solver` deterministic decisions
- `memory.adapters` registry round-trip
- The `MemoryAwareInference` pipeline with the upgraded retrieval flags
"""
from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime

from memory import (
    InMemoryEmbeddingCache,
    MemoryAwareConfig,
    MemoryAwareInference,
    QueryPlan,
    SQLiteMemoryStore,
    build_embedder,
)
from memory.adapters import get_adapter, list_adapters
from memory.bm25 import bm25_search
from memory.cache import CachedEmbedder
from memory.expansion import expand_query
from memory.fusion import reciprocal_rank_fusion, weighted_score_fusion
from memory.granularity import (
    build_episode_memory,
    build_fact_memories,
    build_global_timeline_memory,
    build_timeline_memory,
)
from memory.recency import apply_recency_bias
from memory.solver import solve
from memory.types import MemoryHit, MemoryRecord


def _record(memory_id, text, importance=0.5, memory_type="fact", metadata=None):
    return MemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        memory_type=memory_type,
        text=text,
        source_turn_id=None,
        created_at="2024-01-01T00:00:00+00:00",
        last_accessed_at="2024-01-01T00:00:00+00:00",
        times_retrieved=0,
        importance=importance,
        decay_score=0.0,
        status="active",
        version_group_id=None,
        metadata=metadata or {},
    )


def _hit(memory_id, score, text="text", **md):
    return MemoryHit(
        record=_record(memory_id, text, metadata=md),
        score=score,
        embedding_model="test",
        age_days=0,
    )


class FusionTests(unittest.TestCase):
    def test_rrf_promotes_items_in_multiple_lists(self):
        list_a = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.4)]
        list_b = [_hit("b", 0.9), _hit("d", 0.6), _hit("a", 0.4)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        ids = [hit.record.memory_id for hit in fused]
        # Items present in both lists should be in the top-2.
        self.assertIn("b", ids[:2])
        self.assertIn("a", ids[:2])

    def test_weighted_fusion_normalizes_score_scales(self):
        list_a = [_hit("a", 100.0), _hit("b", 80.0), _hit("c", 50.0)]
        list_b = [_hit("b", 0.95), _hit("c", 0.30)]
        fused = weighted_score_fusion([list_a, list_b], weights=[1.0, 1.0])
        ids = [hit.record.memory_id for hit in fused]
        # `b` is strong in both lists and should win even though list_a's raw
        # scores are 100x list_b's.
        self.assertEqual(ids[0], "b")


class CacheTests(unittest.TestCase):
    def test_cached_embedder_avoids_recompute(self):
        cache = InMemoryEmbeddingCache()
        underlying = build_embedder("hash-32")
        cached = CachedEmbedder(underlying, cache)
        cached.embed("hello")
        cached.embed("hello")
        cached.embed("world")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 2)


class GranularityTests(unittest.TestCase):
    def setUp(self):
        self.session = {
            "session_id": "s1",
            "session_date": "2024-09-12",
            "turns": [
                {"role": "user", "content": "Headed to Tokyo for my birthday with Alice."},
                {"role": "assistant", "content": "Have a great trip!"},
                {"role": "user", "content": "Booked a sushi restaurant for dinner.", "has_answer": True},
            ],
        }

    def test_episode_memory_includes_session_date(self):
        memory = build_episode_memory(self.session)
        self.assertEqual(memory["memory_type"], "episode")
        self.assertIn("2024-09-12", memory["text"])
        self.assertIn("Tokyo", memory["text"])
        self.assertEqual(memory["metadata"]["session_id"], "s1")

    def test_fact_memories_emit_per_turn(self):
        memories = build_fact_memories(self.session)
        self.assertEqual(len(memories), 3)
        self.assertEqual(memories[0]["metadata"]["granularity"], "fact")
        # The has-answer turn gets boosted importance.
        boosted = [m for m in memories if m["metadata"].get("has_answer")]
        self.assertTrue(boosted)
        self.assertGreaterEqual(boosted[0]["importance"], 0.7)

    def test_global_timeline_memory(self):
        sessions = [
            {**self.session},
            {
                "session_id": "s2",
                "session_date": "2024-10-01",
                "turns": [{"role": "user", "content": "Started a new running routine."}],
            },
        ]
        global_memory = build_global_timeline_memory(sessions)
        self.assertIsNotNone(global_memory)
        self.assertEqual(global_memory["memory_type"], "timeline-global")
        self.assertIn("2024-09-12", global_memory["text"])
        self.assertIn("2024-10-01", global_memory["text"])


class SolverTests(unittest.TestCase):
    def _hit_with_date(self, mid, label, event_date, aliases=()):
        return _hit(
            mid,
            0.8,
            text=label,
            event_date=event_date,
            session_date=event_date,
            session_id=mid,
            granularity="fact",
            event_aliases=list(aliases),
            entities=list(aliases),
            summary=label,
            fact_text=label,
            date_confidence=0.9,
        )

    def test_solver_resolves_ordering(self):
        plan = QueryPlan(
            question="Which came first, Alice's wedding or Bob's wedding?",
            query_text="Which came first, Alice's wedding or Bob's wedding?",
            reasoning_kind="ordering",
            targets=["Alice", "Bob"],
            ordering_direction="first",
        )
        hits = [
            self._hit_with_date("h1", "Alice and Carol got married.", "2024-04-12", aliases=["alice"]),
            self._hit_with_date("h2", "Bob got married last spring.", "2025-05-30", aliases=["bob"]),
        ]
        result = solve(plan, hits)
        self.assertTrue(result.resolved)
        self.assertEqual(result.answer.lower(), "alice")

    def test_solver_resolves_difference(self):
        plan = QueryPlan(
            question="How many days between the move-in and the renewal?",
            query_text="How many days between the move-in and the renewal?",
            reasoning_kind="difference",
            unit_hint="days",
            targets=["move-in", "renewal"],
        )
        hits = [
            self._hit_with_date("h1", "Move-in date set.", "2024-01-10", aliases=["move-in"]),
            self._hit_with_date("h2", "Renewal next year.", "2025-01-10", aliases=["renewal"]),
        ]
        result = solve(plan, hits)
        self.assertTrue(result.resolved)
        self.assertIn("days", result.answer)


class ExpansionTests(unittest.TestCase):
    def test_expansion_emits_reformulation_and_entities(self):
        variants = expand_query(
            "How many days between Alice's wedding and Bob's wedding?",
            entities=["Alice", "Bob"],
            anchor_date="2025-06-01",
            targets=["Alice", "Bob"],
        )
        self.assertIn("How many days between Alice's wedding and Bob's wedding?", variants)
        self.assertGreaterEqual(len(variants), 2)


class AdapterRegistryTests(unittest.TestCase):
    def test_adapters_are_registered(self):
        get_adapter("longmemeval")
        get_adapter("locomo")
        get_adapter("memorybench")
        names = list_adapters()
        self.assertIn("longmemeval", names)
        self.assertIn("locomo", names)
        self.assertIn("memorybench", names)


class PipelineUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(os.path.join(self.tmp.name, "memory.sqlite"))
        self.embedder = build_embedder("hash-64")
        self.config = MemoryAwareConfig(
            user_id="user-1",
            top_k=8,
            max_items=3,
            similarity_threshold=0.05,
            critic_threshold=0.4,
            maybe_threshold=0.3,
            memory_token_budget=200,
            fusion_strategy="rrf",
            use_bm25=True,
            use_query_expansion=True,
            diversity=0.4,
        )
        self.memory = MemoryAwareInference(
            store=self.store, embedder=self.embedder, config=self.config,
        )
        for text, mtype in [
            ("Working on the memory-aware inference project with retrieval gating.", "project"),
            ("Prefers sushi for dinner recommendations.", "preference"),
            ("Booked Alice's wedding ceremony for 2024-04-12.", "fact"),
            ("Bob's wedding ceremony is on 2025-05-30.", "fact"),
        ]:
            self.memory.remember(text, memory_type=mtype, importance=0.7)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_rrf_with_bm25_returns_relevant_memory(self):
        plan = QueryPlan(
            question="Debug the memory retrieval gating bug in the current project.",
            query_text="Debug the memory retrieval gating bug in the current project.",
            reasoning_kind="factual",
        )
        ranked = self.memory.rank_hits(plan.query_text, plan=plan)
        kinds = [hit.record.memory_type for hit in ranked]
        self.assertIn("project", kinds)


class RecencyBiasTests(unittest.TestCase):
    def _ku_plan(self):
        return QueryPlan(
            question="What is my current job title?",
            query_text="What is my current job title?",
            reasoning_kind="knowledge-update",
            question_type="knowledge-update",
        )

    def test_promotes_newer_memory_when_plan_matches(self):
        old = _hit("old", 0.80, text="Senior Engineer", event_date="2024-01-01")
        new = _hit("new", 0.78, text="Staff Engineer", event_date="2024-12-01")
        result = apply_recency_bias([old, new], strength=0.4, plan=self._ku_plan())
        self.assertEqual(result[0].record.memory_id, "new")
        # Older memory's score is left unchanged.
        self.assertAlmostEqual(old.score, 0.80, places=6)
        self.assertGreater(new.score, 0.78)

    def test_no_op_when_plan_kind_not_in_trigger_set(self):
        old = _hit("old", 0.80, text="A", event_date="2024-01-01")
        new = _hit("new", 0.78, text="B", event_date="2024-12-01")
        plan = QueryPlan(
            question="What did Alice say?",
            query_text="What did Alice say?",
            reasoning_kind="factual",
            question_type="single-session-user",
        )
        apply_recency_bias([old, new], strength=0.4, plan=plan)
        self.assertEqual(old.score, 0.80)
        self.assertEqual(new.score, 0.78)

    def test_no_op_when_no_dated_memories(self):
        a = _hit("a", 0.80)
        b = _hit("b", 0.78)
        apply_recency_bias([a, b], strength=0.5, plan=self._ku_plan())
        self.assertEqual(a.score, 0.80)
        self.assertEqual(b.score, 0.78)


class YesNoDetectionTests(unittest.TestCase):
    def test_first_person_yes_no_detected(self):
        from memory.prompting import is_yes_no_question

        for q in [
            "Do I have a spare screwdriver?",
            "Did I pay the electric bill last month?",
            "Have I tried Emma's recipes?",
            "Is my partner vegan?",
            "Was it raining when we met?",
            "Should I cancel my subscription?",
        ]:
            self.assertTrue(is_yes_no_question(q), q)

    def test_open_request_not_treated_as_yes_no(self):
        from memory.prompting import is_yes_no_question

        for q in [
            "Can you suggest a good restaurant?",
            "Could you recommend a podcast?",
            "Would you list some hotels in Miami?",
            "Will you describe my project?",
        ]:
            self.assertFalse(is_yes_no_question(q), q)

    def test_open_questions_not_treated_as_yes_no(self):
        from memory.prompting import is_yes_no_question

        self.assertFalse(is_yes_no_question(""))
        self.assertFalse(is_yes_no_question("What is my partner's name?"))
        self.assertFalse(is_yes_no_question("Where did Rachel move to?"))


class PromptingTests(unittest.TestCase):
    def test_answer_instruction_for_plan_routes_preference(self):
        from memory.prompting import (
            answer_instruction_for_plan,
            final_answer_instruction,
            preference_answer_instruction,
        )

        pref_plan = QueryPlan(
            question="What kind of restaurant should I try this weekend?",
            query_text="restaurant recommendation",
            reasoning_kind="preference",
            question_type="single-session-preference",
        )
        factual_plan = QueryPlan(
            question="When did I move to Berlin?",
            query_text="move to Berlin",
            reasoning_kind="factual",
            question_type="single-session-user",
        )

        self.assertEqual(
            answer_instruction_for_plan(pref_plan), preference_answer_instruction()
        )
        self.assertEqual(
            answer_instruction_for_plan(factual_plan), final_answer_instruction()
        )
        self.assertEqual(answer_instruction_for_plan(None), final_answer_instruction())
        # The preference fragment talks about preferences and explicitly tells
        # the model NOT to emit the terse 'Final answer:' line.
        pref = preference_answer_instruction()
        self.assertIn("would prefer", pref)
        self.assertIn("Do NOT end", pref)

    def test_answer_instruction_for_plan_routes_yes_no(self):
        from memory.prompting import (
            answer_instruction_for_plan,
            yes_no_answer_instruction,
        )

        yn_plan = QueryPlan(
            question="Do I have a spare screwdriver?",
            query_text="spare screwdriver",
            reasoning_kind="factual",
            question_type="knowledge-update",
        )
        # Multi-session yes/no opener should still get the normal final-answer
        # instruction, since multi-session implies an aggregated value.
        ms_plan = QueryPlan(
            question="Did I spend more than $100 on coffee this year?",
            query_text="coffee spending",
            reasoning_kind="multi-session",
            question_type="multi-session",
        )

        self.assertEqual(
            answer_instruction_for_plan(yn_plan), yes_no_answer_instruction()
        )
        self.assertNotEqual(
            answer_instruction_for_plan(ms_plan), yes_no_answer_instruction()
        )


class PreferencePostprocessTests(unittest.TestCase):
    def test_postprocess_keeps_multi_sentence_for_preference(self):
        from benchmarks.longmemeval import QuestionPlan, postprocess_prediction

        plan = QuestionPlan(
            question_id="q1",
            question_type="single-session-preference",
            question="What kind of restaurant should I try?",
            query_text="restaurant",
            question_date="",
            normalized_question_date="",
            reasoning_kind="preference",
            is_temporal=False,
            unit_hint="",
            targets=[],
            normalized_targets=[],
            query_entities=[],
            question_month="",
            ordering_direction="first",
            filter_month="",
            is_multi_session=False,
            multi_session_kind="",
            multi_session_subject="",
            multi_session_actions=[],
            multi_session_focus_terms=[],
            range_start="",
            range_end="",
            requires_distinct=False,
            requires_current_state=False,
        )
        raw = (
            "The user would prefer a quiet ramen shop with vegetarian options "
            "and counter seating, since they have mentioned enjoying calm "
            "evenings and avoiding loud groups. They would not prefer a "
            "loud sports bar.\nFinal answer: ramen"
        )
        cleaned = postprocess_prediction(plan, raw)
        self.assertIn("would prefer", cleaned)
        self.assertIn("would not prefer", cleaned)
        self.assertNotIn("Final answer", cleaned)


class LongMemEvalAnalyzerTests(unittest.TestCase):
    def test_ssp_question_gets_preference_reasoning_kind(self):
        from benchmarks.longmemeval import analyze_question

        plan = analyze_question(
            {
                "question_id": "ssp1",
                "question": "What kind of book should I read this weekend?",
                "question_date": "2024-05-01",
                "question_type": "single-session-preference",
                "haystack_sessions": [],
            }
        )
        self.assertEqual(plan.reasoning_kind, "preference")
        self.assertFalse(plan.is_multi_session)

    def test_ssu_question_keeps_factual_reasoning_kind(self):
        from benchmarks.longmemeval import analyze_question

        plan = analyze_question(
            {
                "question_id": "ssu1",
                "question": "What is my partner's name?",
                "question_date": "2024-05-01",
                "question_type": "single-session-user",
                "haystack_sessions": [],
            }
        )
        self.assertEqual(plan.reasoning_kind, "factual")


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(os.path.join(self.tmp.name, "memory.sqlite"))
        self.embedder = build_embedder("hash-32")
        self.engine = MemoryAwareInference(
            store=self.store,
            embedder=self.embedder,
            config=MemoryAwareConfig(user_id="u"),
        )
        self.engine.remember("Alice loves Tokyo sushi at Sukiyabashi.", "fact", importance=0.7)
        self.engine.remember("Bob is into mountain biking on weekends.", "fact", importance=0.6)
        self.engine.remember("Carol opened a new bookstore on Maple street.", "fact", importance=0.6)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_bm25_search_finds_token_matches(self):
        hits = bm25_search(self.store, "Alice Tokyo sushi", "u", top_k=3)
        self.assertTrue(hits)
        self.assertIn("Alice", hits[0].record.text)


if __name__ == "__main__":
    unittest.main()
