import unittest

from benchmarks.question_type_filters import normalize_question_types, question_type_slug


class QuestionTypeFilterTest(unittest.TestCase):
    def test_normalize_question_types_aliases(self):
        self.assertEqual(
            normalize_question_types("temporal,multisession,single_session_assistant,knowledge"),
            "temporal-reasoning,multi-session,single-session-assistant,knowledge-update",
        )

    def test_question_type_slug_is_stable(self):
        self.assertEqual(
            question_type_slug("temporal,multi-session"),
            "temporal_reasoning__multi_session",
        )


if __name__ == "__main__":
    unittest.main()
