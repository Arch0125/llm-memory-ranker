"""
Run an official-style LongMemEval experiment matrix for a fixed reader model.

This script runs three comparable conditions:

1. S full-history baseline: the reader sees the full chat history directly.
2. S + memory layer: the reader sees retrieved evidence from the local memory system.
3. Oracle upper bound: the reader sees only the oracle evidence sessions.

If you point --official_repo_path at a local checkout of the official LongMemEval
repository, this script will also invoke evaluate_qa.py and print_qa_metrics.py
for each condition.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from benchmarks.argv_overrides import apply_argv_overrides
from benchmarks.question_type_filters import normalize_question_types, question_type_slug


data_dir = "data"
reports_dir = "reports/longmemeval_protocol"
s_dataset = "longmemeval_s_cleaned.json"
oracle_dataset = "longmemeval_oracle.json"
max_examples = 0
openai_model = "gpt-4.1-mini"
include_assistant_turns = False
history_format = "nl"
history_granularity = "hybrid"
official_repo_path = ""
official_eval_model = "gpt-4o"
run_conditions = "s_full_history,s_memory,oracle_upper_bound"
question_types = ""
# Set to False to skip the post-hoc retrieval-log passes (session + turn) — they
# add ~30-60s and are diagnostic-only.
run_retrieval_logs = True
# Forwarded to benchmark_longmemeval_openai.py for the s_memory condition.
memory_fusion_strategy = "weighted"
memory_use_bm25 = False
memory_use_query_expansion = False
memory_diversity = 0.0
memory_rerank_top_k = 0
apply_argv_overrides(globals())
question_types = normalize_question_types(question_types)


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "benchmark_longmemeval_openai.py"
RETRIEVAL_RUNNER = ROOT / "benchmark_longmemeval_retrieval.py"
QUESTION_TYPE_SUFFIX = question_type_slug(question_types)


def _dataset_path(name):
    return str((ROOT / data_dir / name).resolve())


def _reports_path(name):
    if not name:
        return (ROOT / reports_dir).resolve()
    if QUESTION_TYPE_SUFFIX:
        path = Path(name)
        suffixes = "".join(path.suffixes)
        stem = path.name[: -len(suffixes)] if suffixes else path.name
        name = f"{stem}__{QUESTION_TYPE_SUFFIX}{suffixes}"
    return (ROOT / reports_dir / name).resolve()


def _parse_conditions(value):
    allowed = {"s_full_history", "s_memory", "oracle_upper_bound"}
    picked = []
    for item in (value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if item not in allowed:
            raise ValueError(f"Unknown run condition: {item}")
        picked.append(item)
    return picked or ["s_full_history", "s_memory", "oracle_upper_bound"]


def _run_python(args, cwd=None):
    cmd = [sys.executable] + args
    subprocess.run(cmd, check=True, cwd=cwd)
    return cmd


def _run_condition(label, dataset_path, memory_enabled, reader_context_mode):
    output_path = str(_reports_path(f"{label}_predictions.jsonl"))
    details_path = str(_reports_path(f"{label}_details.jsonl"))
    summary_path = str(_reports_path(f"{label}_summary.json"))

    args = [
        str(RUNNER),
        f"--dataset_path={dataset_path}",
        f"--openai_model={openai_model}",
        f"--output_path={output_path}",
        f"--details_path={details_path}",
        f"--summary_path={summary_path}",
        f"--memory_enabled={memory_enabled}",
        f"--reader_context_mode={reader_context_mode}",
        f"--include_assistant_turns={include_assistant_turns}",
        f"--history_format={history_format}",
        f"--history_granularity={history_granularity}",
    ]
    if max_examples > 0:
        args.append(f"--max_examples={max_examples}")
    if question_types:
        args.append(f"--question_types={question_types}")
    if memory_enabled:
        args.extend([
            f"--memory_fusion_strategy={memory_fusion_strategy}",
            f"--memory_use_bm25={memory_use_bm25}",
            f"--memory_use_query_expansion={memory_use_query_expansion}",
            f"--memory_diversity={memory_diversity}",
            f"--memory_rerank_top_k={memory_rerank_top_k}",
        ])

    cmd = _run_python(args, cwd=str(ROOT))
    return {
        "label": label,
        "dataset_path": dataset_path,
        "memory_enabled": memory_enabled,
        "reader_context_mode": reader_context_mode,
        "output_path": output_path,
        "details_path": details_path,
        "summary_path": summary_path,
        "command": cmd,
    }


def _run_retrieval_condition(label, dataset_path, granularity):
    output_path = str(_reports_path(f"{label}_retrievallog_{granularity}_memory.jsonl"))
    summary_path = str(_reports_path(f"{label}_retrievallog_{granularity}_memory_summary.json"))
    args = [
        str(RETRIEVAL_RUNNER),
        f"--dataset_path={dataset_path}",
        f"--granularity={granularity}",
        f"--output_path={output_path}",
        f"--summary_path={summary_path}",
    ]
    if max_examples > 0:
        args.append(f"--max_examples={max_examples}")
    if question_types:
        args.append(f"--question_types={question_types}")
    cmd = _run_python(args, cwd=str(ROOT))
    return {
        "label": label,
        "dataset_path": dataset_path,
        "granularity": granularity,
        "output_path": output_path,
        "summary_path": summary_path,
        "summary": _load_summary(summary_path),
        "command": cmd,
    }


def _find_script(repo_root, script_name):
    matches = list(Path(repo_root).resolve().rglob(script_name))
    if not matches:
        raise FileNotFoundError(f"Could not find {script_name} under {repo_root}")
    return matches[0]


def _run_official_eval(condition):
    repo_root = Path(official_repo_path).resolve()
    eval_script = _find_script(repo_root, "evaluate_qa.py")
    metrics_script = _find_script(repo_root, "print_qa_metrics.py")
    eval_dir = eval_script.parent

    prediction_path = str(Path(condition["output_path"]).resolve())
    dataset_path = str(Path(condition["dataset_path"]).resolve())
    log_path = prediction_path + ".log"
    eval_stdout_path = str(_reports_path(f"{condition['label']}_official_eval.txt"))
    metrics_stdout_path = str(_reports_path(f"{condition['label']}_official_metrics.txt"))

    eval_cmd = [sys.executable, str(eval_script), official_eval_model, prediction_path, dataset_path]
    eval_proc = subprocess.run(
        eval_cmd,
        check=False,
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
    )
    if eval_proc.returncode != 0:
        raise RuntimeError(
            "Official LongMemEval QA evaluation failed.\n"
            f"Command: {' '.join(eval_cmd)}\n"
            f"stdout:\n{eval_proc.stdout}\n"
            f"stderr:\n{eval_proc.stderr}"
        )
    Path(eval_stdout_path).write_text(eval_proc.stdout + eval_proc.stderr, encoding="utf-8")
    if not Path(log_path).exists():
        alt_log_path = eval_dir / (Path(prediction_path).name + ".log")
        if alt_log_path.exists():
            log_path = str(alt_log_path)

    metrics_cmd = [sys.executable, str(metrics_script), official_eval_model, log_path, dataset_path]
    metrics_proc = subprocess.run(
        metrics_cmd,
        check=False,
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
    )
    if metrics_proc.returncode != 0:
        raise RuntimeError(
            "Official LongMemEval QA metric aggregation failed.\n"
            f"Command: {' '.join(metrics_cmd)}\n"
            f"stdout:\n{metrics_proc.stdout}\n"
            f"stderr:\n{metrics_proc.stderr}"
        )
    Path(metrics_stdout_path).write_text(metrics_proc.stdout + metrics_proc.stderr, encoding="utf-8")

    return {
        "official_eval_stdout_path": eval_stdout_path,
        "official_metrics_stdout_path": metrics_stdout_path,
        "official_log_path": log_path,
    }


def _run_official_retrieval_eval(retrieval_condition):
    repo_root = Path(official_repo_path).resolve()
    metrics_script = _find_script(repo_root, "print_retrieval_metrics.py")
    eval_dir = metrics_script.parent
    retrieval_log_path = str(Path(retrieval_condition["output_path"]).resolve())
    metrics_stdout_path = str(
        _reports_path(
            f"{retrieval_condition['label']}_retrievallog_{retrieval_condition['granularity']}_official_metrics.txt"
        )
    )
    metrics_cmd = [sys.executable, str(metrics_script), retrieval_log_path]
    metrics_proc = subprocess.run(
        metrics_cmd,
        check=False,
        cwd=str(eval_dir),
        capture_output=True,
        text=True,
    )
    if metrics_proc.returncode != 0:
        raise RuntimeError(
            "Official LongMemEval retrieval metric evaluation failed.\n"
            f"Command: {' '.join(metrics_cmd)}\n"
            f"stdout:\n{metrics_proc.stdout}\n"
            f"stderr:\n{metrics_proc.stderr}"
        )
    Path(metrics_stdout_path).write_text(metrics_proc.stdout + metrics_proc.stderr, encoding="utf-8")
    return {
        "official_metrics_stdout_path": metrics_stdout_path,
    }


def _load_summary(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_delta(memory_summary, baseline_summary):
    print("Comparison: S + memory vs S full-history baseline")
    for key in ("exact_match", "token_f1", "contains_match"):
        mem_value = memory_summary.get(key, 0.0) or 0.0
        base_value = baseline_summary.get(key, 0.0) or 0.0
        delta = mem_value - base_value
        print(f"- {key}: memory={mem_value:.4f} baseline={base_value:.4f} delta={delta:+.4f}")


reports_root = _reports_path("")
reports_root.mkdir(parents=True, exist_ok=True)

conditions = _parse_conditions(run_conditions)
condition_specs = {
    "s_full_history": (_dataset_path(s_dataset), False, "full-history"),
    "s_memory": (_dataset_path(s_dataset), True, "memory"),
    "oracle_upper_bound": (_dataset_path(oracle_dataset), False, "full-history"),
}

manifest = {
    "openai_model": openai_model,
    "official_eval_model": official_eval_model,
    "question_types": question_types,
    "data_dir": str((ROOT / data_dir).resolve()),
    "reports_dir": str(reports_root),
    "conditions": [],
    "retrieval_conditions": [],
}

for name in conditions:
    dataset_path, memory_enabled_value, context_mode = condition_specs[name]
    condition = _run_condition(
        label=name,
        dataset_path=dataset_path,
        memory_enabled=memory_enabled_value,
        reader_context_mode=context_mode,
    )
    condition["summary"] = _load_summary(condition["summary_path"])
    if official_repo_path:
        condition.update(_run_official_eval(condition))
    manifest["conditions"].append(condition)

if "s_memory" in conditions and run_retrieval_logs:
    for granularity in ("session", "turn"):
        retrieval_condition = _run_retrieval_condition(
            label="s_memory",
            dataset_path=_dataset_path(s_dataset),
            granularity=granularity,
        )
        if official_repo_path:
            retrieval_condition.update(_run_official_retrieval_eval(retrieval_condition))
        manifest["retrieval_conditions"].append(retrieval_condition)

manifest_path = _reports_path("manifest.json")
with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)

print(f"Wrote protocol manifest to {manifest_path}")

by_label = {item["label"]: item for item in manifest["conditions"]}
if "s_memory" in by_label and "s_full_history" in by_label:
    _print_delta(by_label["s_memory"]["summary"], by_label["s_full_history"]["summary"])
