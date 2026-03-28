import tempfile
import unittest

from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(f"{self.tmp.name}/memory.sqlite")
        self.embedder = build_embedder("hash-64")
        self.memory = MemoryAwareInference(
            store=self.store,
            embedder=self.embedder,
            config=MemoryAwareConfig(
                user_id="user-1",
                top_k=8,
                max_items=3,
                similarity_threshold=0.18,
                critic_threshold=0.58,
                maybe_threshold=0.48,
                memory_token_budget=96,
            ),
        )
        self.memory.remember(
            "Working on the memory-aware inference project with retrieval gating.",
            memory_type="project",
            importance=0.95,
        )
        self.memory.remember(
            "Prefers sushi for dinner recommendations.",
            memory_type="preference",
            importance=0.85,
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_pipeline_suppresses_irrelevant_preference_memory(self):
        prompt, _, selected = self.memory.prepare_prompt(
            "Debug the memory retrieval gating bug in the current project."
        )

        selected_types = [hit.record.memory_type for hit in selected]
        self.assertIn("project", selected_types)
        self.assertNotIn("preference", selected_types)
        self.assertIn("memory-aware inference project", prompt)
        self.assertNotIn("sushi", prompt.lower())

    def test_pipeline_uses_preference_when_query_is_applicable(self):
        _, _, selected = self.memory.prepare_prompt(
            "Recommend a sushi dinner plan that fits my preferences."
        )

        self.assertIn("preference", [hit.record.memory_type for hit in selected])

    def test_pipeline_can_render_charset_safe_prompt_for_char_models(self):
        allowed_chars = set("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        self.memory.config.memory_token_budget = 400

        def strict_encode(text):
            for char in text:
                if char not in allowed_chars:
                    raise KeyError(char)
            return list(text)

        prompt, _, selected = self.memory.prepare_prompt(
            "Debug the memory retrieval gating bug in the current project.",
            encode=strict_encode,
            plain_text_prompt=True,
            allowed_chars=allowed_chars,
        )

        self.assertTrue(selected)
        self.assertTrue(all(char in allowed_chars for char in prompt))
        self.assertNotIn("[", prompt)
        self.assertTrue(
            "MEMORY HIGH CONFIDENCE" in prompt or "MEMORY LOW CONFIDENCE" in prompt
        )

    def test_pipeline_can_render_completion_style_prompt(self):
        prompt, _, selected = self.memory.prepare_prompt(
            "Debug the memory retrieval gating bug in the current project.",
            prompt_style="completion",
        )

        self.assertTrue(selected)
        self.assertIn("Instructions:", prompt)
        self.assertTrue(
            "Relevant memory:" in prompt or "Possible but lower confidence memory:" in prompt
        )
        self.assertIn("Assistant response:", prompt)
        self.assertNotIn("SYSTEM:", prompt)


if __name__ == "__main__":
    unittest.main()
