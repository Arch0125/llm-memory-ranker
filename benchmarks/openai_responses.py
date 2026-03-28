from __future__ import annotations

import time

import httpx


def build_responses_payload(
    model,
    instructions,
    user_input,
    max_output_tokens,
    temperature=0.0,
    top_p=1.0,
    metadata=None,
    reasoning_effort="",
    verbosity="",
):
    payload = {
        "model": model,
        "instructions": instructions,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            }
        ],
        "max_output_tokens": max_output_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "store": False,
    }
    if metadata:
        payload["metadata"] = {str(key): str(value) for key, value in metadata.items()}
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    if verbosity:
        payload["text"] = {"verbosity": verbosity}
    return payload


def extract_output_text(response_json):
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    parts = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                parts.append(content.get("text", ""))
    return "".join(parts).strip()


def extract_usage(response_json):
    usage = response_json.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def create_response(
    api_key,
    payload,
    base_url="https://api.openai.com/v1",
    timeout_seconds=120.0,
    max_retries=5,
):
    url = f"{base_url.rstrip('/')}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    with httpx.Client(timeout=timeout_seconds) as client:
        for attempt in range(max_retries + 1):
            try:
                response = client.post(url, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(min(8.0, 0.75 * (2 ** attempt)))
                    continue
                response.raise_for_status()
                data = response.json()
                return {
                    "response_id": data.get("id"),
                    "output_text": extract_output_text(data),
                    "usage": extract_usage(data),
                    "raw": data,
                }
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= max_retries:
                    break
                time.sleep(min(8.0, 0.75 * (2 ** attempt)))

    raise RuntimeError(f"OpenAI Responses API call failed: {last_error}")
