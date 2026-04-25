"""Tests for the generic `Final answer:` extraction helper."""
from __future__ import annotations

import unittest

from memory.postprocess import extract_final_answer_marker
from memory.prompting import final_answer_instruction, terse_answer_hint
from memory.query import QueryPlan


class ExtractFinalAnswerMarkerTest(unittest.TestCase):
    def test_returns_empty_when_no_marker(self):
        self.assertEqual(extract_final_answer_marker("Just a sentence."), "")

    def test_returns_empty_for_empty_input(self):
        self.assertEqual(extract_final_answer_marker(""), "")
        self.assertEqual(extract_final_answer_marker(None), "")

    def test_extracts_value_after_final_answer_line(self):
        self.assertEqual(
            extract_final_answer_marker(
                "Some reasoning.\nFinal answer: Business Administration"
            ),
            "Business Administration",
        )

    def test_strips_double_quotes(self):
        self.assertEqual(
            extract_final_answer_marker("Final answer: \"Summer Vibes\""),
            "Summer Vibes",
        )

    def test_strips_single_quotes(self):
        self.assertEqual(extract_final_answer_marker("Final answer: 'Tokyo'"), "Tokyo")

    def test_strips_smart_quotes(self):
        self.assertEqual(
            extract_final_answer_marker("Final answer: \u201cParis\u201d"), "Paris"
        )

    def test_strips_trailing_period_for_words(self):
        self.assertEqual(extract_final_answer_marker("Final answer: $800."), "$800")

    def test_keeps_decimal_in_numbers(self):
        self.assertEqual(extract_final_answer_marker("Final answer: 3.14"), "3.14")

    def test_takes_last_marker_when_multiple_lines(self):
        text = (
            "Initial guess.\nFinal answer: $700\n"
            "Wait, re-reading.\nFinal answer: $800"
        )
        self.assertEqual(extract_final_answer_marker(text), "$800")

    def test_accepts_short_alias_answer(self):
        self.assertEqual(extract_final_answer_marker("Answer: 7 days"), "7 days")

    def test_is_case_insensitive(self):
        self.assertEqual(extract_final_answer_marker("FINAL ANSWER: hello"), "hello")

    def test_ignores_inline_chain_of_thought(self):
        # When the model muses about "the final answer might be 5" mid-sentence
        # but ends with the canonical line, we should grab the trailing value.
        text = "Let me think. The final answer might be 5.\nFinal answer: 7"
        self.assertEqual(extract_final_answer_marker(text), "7")


class PromptingTest(unittest.TestCase):
    def test_instruction_mentions_final_answer_format(self):
        text = final_answer_instruction()
        self.assertIn("Final answer:", text)
        self.assertIn("terse", text.lower())

    def test_terse_hint_for_ordering(self):
        plan = QueryPlan(reasoning_kind="ordering")
        self.assertIn("ordering", terse_answer_hint(plan).lower())

    def test_terse_hint_for_difference_uses_unit(self):
        plan = QueryPlan(reasoning_kind="difference", unit_hint="weeks")
        self.assertIn("weeks", terse_answer_hint(plan))

    def test_terse_hint_default_factual_is_empty(self):
        plan = QueryPlan(reasoning_kind="factual")
        self.assertEqual(terse_answer_hint(plan), "")

    def test_terse_hint_handles_none(self):
        self.assertEqual(terse_answer_hint(None), "")


if __name__ == "__main__":
    unittest.main()
