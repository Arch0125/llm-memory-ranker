import tempfile
import unittest

from memory import SQLiteMemoryStore, build_embedder


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(f"{self.tmp.name}/memory.sqlite")
        self.embedder = build_embedder("hash-64")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_search_ranks_relevant_memory_first_and_filters_by_user(self):
        project_text = "Working on a memory-aware inference project."
        unrelated_text = "Favorite pizza topping is mushroom."
        other_user_text = "Unrelated memory from another user."

        project = self.store.add_memory(
            user_id="user-1",
            text=project_text,
            memory_type="project",
            importance=0.9,
            embedding=self.embedder.embed(project_text),
            embedding_model=self.embedder.model_name,
        )
        self.store.add_memory(
            user_id="user-1",
            text=unrelated_text,
            memory_type="preference",
            importance=0.4,
            embedding=self.embedder.embed(unrelated_text),
            embedding_model=self.embedder.model_name,
        )
        self.store.add_memory(
            user_id="user-2",
            text=other_user_text,
            memory_type="project",
            importance=0.9,
            embedding=self.embedder.embed(other_user_text),
            embedding_model=self.embedder.model_name,
        )

        hits = self.store.search(
            vector=self.embedder.embed("continue the memory inference project"),
            user_id="user-1",
            top_k=5,
            model_name=self.embedder.model_name,
        )

        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0].record.memory_id, project.memory_id)
        self.assertTrue(all(hit.record.user_id == "user-1" for hit in hits))


if __name__ == "__main__":
    unittest.main()
