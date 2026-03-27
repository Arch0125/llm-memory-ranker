import unittest
from datetime import timedelta

from memory.types import MemoryHit, MemoryRecord
from memory.utils import iso_timestamp, utc_now
from prompt.budget import select_memories


def make_hit(memory_id, text, score, label, confidence, importance=0.5, age_days=0):
    now = utc_now()
    record = MemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        memory_type="project",
        text=text,
        created_at=iso_timestamp(now - timedelta(days=age_days)),
        last_accessed_at=iso_timestamp(now - timedelta(days=age_days)),
        importance=importance,
    )
    return MemoryHit(
        record=record,
        score=score,
        embedding_model="hash-64",
        age_days=age_days,
        critic_label=label,
        critic_confidence=confidence,
    )


class BudgetTests(unittest.TestCase):
    def test_select_memories_prefers_higher_ranked_hits_under_budget(self):
        best = make_hit(
            "best",
            "Working on the memory-aware inference project.",
            score=0.91,
            label="use",
            confidence=0.84,
            importance=0.9,
        )
        worse = make_hit(
            "worse",
            "A lower value candidate that should be dropped when the budget is tight.",
            score=0.44,
            label="maybe",
            confidence=0.53,
            importance=0.4,
        )

        selected, used_tokens = select_memories([worse, best], max_items=2, max_tokens=22)

        self.assertEqual([hit.record.memory_id for hit in selected], ["best"])
        self.assertGreater(used_tokens, 0)


if __name__ == "__main__":
    unittest.main()
