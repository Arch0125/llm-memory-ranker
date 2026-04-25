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
