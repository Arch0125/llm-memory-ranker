import unittest
from datetime import timedelta

from memory.policies import DecayConfig, decay_and_prune
from memory.types import MemoryRecord
from memory.utils import iso_timestamp, utc_now


def make_record(memory_id, importance, age_days, times_retrieved=0):
    now = utc_now()
    return MemoryRecord(
        memory_id=memory_id,
        user_id="user-1",
        memory_type="ephemeral",
        text=f"record {memory_id}",
        created_at=iso_timestamp(now - timedelta(days=age_days)),
        last_accessed_at=iso_timestamp(now - timedelta(days=age_days)),
        importance=importance,
        times_retrieved=times_retrieved,
    )


class PolicyTests(unittest.TestCase):
    def test_decay_archives_old_low_importance_memories(self):
        old_low_value = make_record("m1", importance=0.2, age_days=200)
        old_high_value = make_record("m2", importance=0.9, age_days=200, times_retrieved=6)

        updated = decay_and_prune(
            [old_low_value, old_high_value],
            cfg=DecayConfig(hard_ttl_days=120, keep_if_importance_at_least=0.8),
        )

        self.assertEqual(updated[0].status, "archived")
        self.assertEqual(updated[1].status, "active")
        self.assertGreater(updated[1].decay_score, updated[0].decay_score)


if __name__ == "__main__":
    unittest.main()
