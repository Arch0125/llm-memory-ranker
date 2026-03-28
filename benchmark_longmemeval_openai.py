"""
Run LongMemEval with the local memory-aware inference layer and OpenAI's Responses API.

This keeps retrieval, gating, and prompt assembly local, but swaps the final
generator from the local MLX model to an OpenAI-hosted model.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from benchmarks.longmemeval import (
    abstention_score,
    build_query_text,
    contains_match_score,
    exact_match_score,
    iter_history_memories,
    load_longmemeval_instances,
    selected_session_ids,
    selected_session_recall,
    summarize_records,
    token_f1_score,
)
from benchmarks.openai_responses import build_responses_payload, create_response
from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder
from memory.explain import format_trace
from prompt.template import DEFAULT_SYSTEM_PROMPT, render_memory_line

# -----------------------------------------------------------------------------
dataset_path = "data/longmemeval_oracle.json"
output_path = "reports/longmemeval_openai_predictions.jsonl"
details_path = "reports/longmemeval_openai_details.jsonl"
summary_path = "reports/longmemeval_openai_summary.json"
start_index = 0
max_examples = 0
question_types = ""
history_granularity = "turn"
include_assistant_turns = True
include_question_date = True
max_new_tokens = 64
temperature = 0.0
top_p = 1.0
memory_enabled = True
memory_user_id = "benchmark"
memory_embedder = "hash-384"
memory_top_k = 24
memory_max_items = 8
memory_similarity_threshold = 0.12
memory_critic_threshold = 0.52
memory_maybe_threshold = 0.42
memory_max_age_days = -1
memory_token_budget = 512
memory_type_allowlist = ""
memory_recent_context = ""
memory_explain = False
openai_api_key = ""
openai_base_url = "https://api.openai.com/v1"
openai_model = "gpt-5-mini"
openai_timeout_seconds = 120.0
openai_max_retries = 5
openai_reasoning_effort = ""
openai_verbosity = ""
exec(open("configurator.py").read())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def _build_memory_block(selected_hits):
    if not selected_hits:
        return ""
    lines = [
        "Retrieved memory:",
        *[f"- {render_memory_line(hit, plain_text=True)}" for hit in selected_hits],
        "Use retrieved memory only if it is directly relevant to the user's request.",
    ]
    return "\n".join(lines)


def _build_instructions(selected_hits, recent_context="", base_system_prompt=None):
    parts = [base_system_prompt or DEFAULT_SYSTEM_PROMPT]
    memory_block = _build_memory_block(selected_hits)
    if memory_block:
        parts.append(memory_block)
    if recent_context:
        parts.append(f"Recent context:\n{recent_context.strip()}")
    parts.append(
        "Answer the user's request directly. If the retrieved memory is insufficient, say you do not know."
    )
    return "\n\n".join(parts)


def _make_memory_system(store, embedder):
    return MemoryAwareInference(
        store=store,
        embedder=embedder,
        config=MemoryAwareConfig(
            user_id=memory_user_id,
            top_k=memory_top_k,
            max_items=memory_max_items,
            similarity_threshold=memory_similarity_threshold,
            critic_threshold=memory_critic_threshold,
            maybe_threshold=memory_maybe_threshold,
            max_age_days=None if memory_max_age_days < 0 else memory_max_age_days,
            memory_token_budget=memory_token_budget,
            type_allowlist=memory_type_allowlist,
        ),
    )


def _filter_instances(instances):
    filtered = instances[start_index:]
    if question_types:
        allowed = {item.strip() for item in question_types.split(",") if item.strip()}
        filtered = [item for item in filtered if item.get("question_type") in allowed]
    if max_examples > 0:
        filtered = filtered[:max_examples]
    return filtered


def _ingest_history(store, embedder, instance):
    for memory_item in iter_history_memories(
        instance,
        granularity=history_granularity,
        include_assistant_turns=include_assistant_turns,
    ):
        store.add_memory(
            user_id=memory_user_id,
            text=memory_item["text"],
            memory_type=memory_item["memory_type"],
            importance=memory_item["importance"],
            embedding=embedder.embed(memory_item["text"]),
            embedding_model=embedder.model_name,
            metadata=memory_item["metadata"],
        )


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


if history_granularity not in {"turn", "session"}:
    raise ValueError("history_granularity must be 'turn' or 'session'")

resolved_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
if not resolved_api_key:
    raise ValueError("Set OPENAI_API_KEY or pass --openai_api_key=...")

instances = _filter_instances(load_longmemeval_instances(dataset_path))
if not instances:
    raise ValueError("No LongMemEval instances selected")

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(details_path).parent.mkdir(parents=True, exist_ok=True)
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

embedder = build_embedder(memory_embedder)
prediction_rows = []
detail_rows = []

for index, instance in enumerate(instances, start=1):
    question_id = instance["question_id"]
    query_text = build_query_text(instance, include_question_date=include_question_date)

    with tempfile.TemporaryDirectory(prefix="longmemeval-openai-") as tempdir:
        store = SQLiteMemoryStore(str(Path(tempdir) / "memory.sqlite"))
        selected_hits = []
        trace = None

        if memory_enabled:
            _ingest_history(store, embedder, instance)
            memory_system = _make_memory_system(store, embedder)
            _, trace, selected_hits = memory_system.prepare_prompt(
                query_text=query_text,
                recent_context=memory_recent_context,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                encode=None,
                prompt_style="completion",
            )

        instructions = _build_instructions(
            selected_hits=selected_hits,
            recent_context=memory_recent_context,
            base_system_prompt=DEFAULT_SYSTEM_PROMPT,
        )
        response = create_response(
            api_key=resolved_api_key,
            payload=build_responses_payload(
                model=openai_model,
                instructions=instructions,
                user_input=query_text,
                max_output_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                metadata={
                    "benchmark": "longmemeval",
                    "question_id": question_id,
                    "memory_enabled": memory_enabled,
                },
                reasoning_effort=openai_reasoning_effort,
                verbosity=openai_verbosity,
            ),
            base_url=openai_base_url,
            timeout_seconds=openai_timeout_seconds,
            max_retries=openai_max_retries,
        )
        hypothesis = response["output_text"]
        store.close()

    if memory_explain and trace is not None:
        print(f"[{index}/{len(instances)}] {question_id}")
        print(format_trace(trace))

    usage = response["usage"]
    prediction_rows.append(
        {
            "question_id": question_id,
            "hypothesis": hypothesis,
        }
    )
    detail_rows.append(
        {
            "question_id": question_id,
            "question_type": instance.get("question_type"),
            "question": instance.get("question"),
            "answer": instance.get("answer"),
            "hypothesis": hypothesis,
            "openai_model": openai_model,
            "response_id": response.get("response_id"),
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "exact_match": exact_match_score(hypothesis, instance.get("answer", "")),
            "contains_match": contains_match_score(hypothesis, instance.get("answer", "")),
            "token_f1": token_f1_score(hypothesis, instance.get("answer", "")),
            "abstention_accuracy": abstention_score(
                hypothesis,
                question_id.endswith("_abs"),
            ),
            "selected_memory_count": len(selected_hits),
            "selected_session_ids": selected_session_ids(selected_hits),
            "answer_session_ids": instance.get("answer_session_ids", []),
            "selected_session_recall": selected_session_recall(
                selected_hits,
                instance.get("answer_session_ids", []),
            ),
        }
    )

    if index % 10 == 0 or index == len(instances):
        print(f"Processed {index}/{len(instances)} LongMemEval examples with OpenAI")

summary = summarize_records(detail_rows)
summary.update(
    {
        "dataset_path": dataset_path,
        "memory_enabled": memory_enabled,
        "history_granularity": history_granularity,
        "include_assistant_turns": include_assistant_turns,
        "openai_model": openai_model,
        "output_path": output_path,
        "details_path": details_path,
        "total_input_tokens": sum(row["input_tokens"] for row in detail_rows),
        "total_output_tokens": sum(row["output_tokens"] for row in detail_rows),
        "total_tokens": sum(row["total_tokens"] for row in detail_rows),
    }
)

_write_jsonl(output_path, prediction_rows)
_write_jsonl(details_path, detail_rows)
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

print(f"Wrote predictions to {output_path}")
print(f"Wrote detailed local metrics to {details_path}")
print(f"Wrote summary to {summary_path}")
print(json.dumps(summary, indent=2, sort_keys=True))
