"""
Generate an official-style LongMemEval retrieval log for the local memory retriever.

The output format matches the official retrieval log schema used by
src/evaluation/print_retrieval_metrics.py in the LongMemEval repository.
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.argv_overrides import apply_argv_overrides
from benchmarks.question_type_filters import normalize_question_types
from benchmarks.longmemeval import (
    build_official_retrieval_log_entry,
    load_longmemeval_instances,
    summarize_official_retrieval_logs,
)
from memory import build_embedder


dataset_path = "data/longmemeval_s_cleaned.json"
output_path = "reports/longmemeval_retrievallog_turn_memory.jsonl"
summary_path = "reports/longmemeval_retrievallog_turn_memory_summary.json"
max_examples = 0
start_index = 0
question_types = ""
granularity = "turn"
memory_embedder = "benchmark-auto"
apply_argv_overrides(globals())
question_types = normalize_question_types(question_types)


if granularity not in {"session", "turn"}:
    raise ValueError("granularity must be 'session' or 'turn'")


def _filter_instances(instances):
    filtered = instances[start_index:]
    if question_types:
        allowed = {item.strip() for item in question_types.split(",") if item.strip()}
        filtered = [item for item in filtered if item.get("question_type") in allowed]
    if max_examples > 0:
        filtered = filtered[:max_examples]
    return filtered


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


instances = _filter_instances(load_longmemeval_instances(dataset_path))
if not instances:
    raise ValueError("No LongMemEval instances selected")

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

embedder = build_embedder(memory_embedder)
rows = []
for index, instance in enumerate(instances, start=1):
    rows.append(build_official_retrieval_log_entry(instance, granularity=granularity, embedder=embedder))
    if index % 10 == 0 or index == len(instances):
        print(f"Processed {index}/{len(instances)} LongMemEval retrieval examples")

summary = summarize_official_retrieval_logs(rows)
summary.update(
    {
        "dataset_path": dataset_path,
        "question_types": question_types,
        "granularity": granularity,
        "memory_embedder": memory_embedder,
        "output_path": output_path,
    }
)

_write_jsonl(output_path, rows)
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)

print(f"Wrote retrieval log to {output_path}")
print(f"Wrote retrieval summary to {summary_path}")
print(json.dumps(summary, indent=2, sort_keys=True))
