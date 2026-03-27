import unittest
from datetime import timedelta

from memory.retrieve import gate_hits
from memory.types import MemoryHit, MemoryRecord
from memory.utils import iso_timestamp, utc_now


def make_hit(memory_id, memory_type, score, importance, age_days):
    now = utc_now()
    record = MemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        memory_type=memory_type,
        text=f"{memory_type} memory",
        created_at=iso_timestamp(now - timedelta(days=age_days)),
        last_accessed_at=iso_timestamp(now - timedelta(days=age_days)),
        importance=importance,
    )
    return MemoryHit(record=record, score=score, embedding_model="hash-64", age_days=age_days)


class GateTests(unittest.TestCase):
    def test_gate_filters_low_similarity_and_stale_ephemera(self):
        low_similarity = make_hit("m1", "project", 0.05, 0.9, 1)
        stale_ephemeral = make_hit("m2", "ephemeral", 0.8, 0.3, 10)
        stable_project = make_hit("m3", "project", 0.7, 0.95, 200)

        kept = gate_hits(
            [low_similarity, stale_ephemeral, stable_project],
            sim_threshold=0.18,
            max_age_days=None,
            stable_importance_threshold=0.8,
        )

        self.assertEqual([hit.record.memory_id for hit in kept], ["m3"])


if __name__ == "__main__":
    unittest.main()
