import unittest

from benchmarks.openai_responses import (
    build_responses_payload,
    extract_output_text,
    extract_usage,
)


class OpenAIResponsesHelpersTest(unittest.TestCase):
    def test_build_responses_payload_sets_optional_fields(self):
        payload = build_responses_payload(
            model="gpt-4.1-mini",
            instructions="You are helpful.",
            user_input="Hello",
            max_output_tokens=32,
            temperature=0.0,
            top_p=1.0,
            metadata={"question_id": "q1"},
            reasoning_effort="low",
            verbosity="low",
        )
        self.assertEqual(payload["model"], "gpt-4.1-mini")
        self.assertEqual(payload["input"], "Hello")
        self.assertEqual(payload["reasoning"]["effort"], "low")
        self.assertEqual(payload["text"]["verbosity"], "low")
        self.assertEqual(payload["metadata"]["question_id"], "q1")
        self.assertNotIn("temperature", payload)
        self.assertNotIn("top_p", payload)

    def test_build_responses_payload_keeps_non_default_sampling_fields(self):
        payload = build_responses_payload(
            model="gpt-4.1-mini",
            instructions="You are helpful.",
            user_input="Hello",
            max_output_tokens=32,
            temperature=0.7,
            top_p=0.9,
        )
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.9)

    def test_extract_output_text_prefers_output_text(self):
        response_json = {
            "output_text": "Final answer",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ignored"}],
                }
            ],
        }
        self.assertEqual(extract_output_text(response_json), "Final answer")

    def test_extract_output_text_falls_back_to_output_items(self):
        response_json = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Hello"},
                        {"type": "output_text", "text": " world"},
                    ],
                }
            ]
        }
        self.assertEqual(extract_output_text(response_json), "Hello world")

    def test_extract_usage(self):
        usage = extract_usage({"usage": {"input_tokens": 10, "output_tokens": 4}})
        self.assertEqual(usage["input_tokens"], 10)
        self.assertEqual(usage["output_tokens"], 4)
        self.assertEqual(usage["total_tokens"], 14)


if __name__ == "__main__":
    unittest.main()
