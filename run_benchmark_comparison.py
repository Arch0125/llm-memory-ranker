#!/usr/bin/env python3
"""
Run a head-to-head comparison of the full-history baseline vs the memory layer
on the first N examples from the LongMemEval-S dataset, then print a comparison table.

Usage:
    ./venv/bin/python run_benchmark_comparison.py
    ./venv/bin/python run_benchmark_comparison.py --max_examples=50
    ./venv/bin/python run_benchmark_comparison.py --max_examples=100 --official_repo_path=/tmp/LongMemEval-official
    ./venv/bin/python run_benchmark_comparison.py --max_examples=500 --official_repo_path=/tmp/LongMemEval-official --reuse_baseline=True
    ./venv/bin/python run_benchmark_comparison.py --max_examples=500 --question_types=temporal,multisession

Set OPENAI_API_KEY in your environment before running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.argv_overrides import apply_argv_overrides
from benchmarks.question_type_filters import normalize_question_types, question_type_slug


# ── defaults (all overridable via --flag=value) ─────────────────────────────
max_examples = 100
openai_model = "gpt-4.1-mini"
dataset_path = "data/longmemeval_s_cleaned.json"
reports_dir = "reports/longmemeval_protocol"
official_repo_path = ""          # set to run GPT-4o autoeval
official_eval_model = "gpt-4o"
include_assistant_turns = False
history_format = "nl"
history_granularity = "hybrid"
question_types = ""
reuse_baseline = False
reuse_memory = False
parallel_workers = 8
apply_argv_overrides(globals())
question_types = normalize_question_types(question_types)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "benchmark_longmemeval_openai.py"
REPORTS = ROOT / reports_dir
REPORTS.mkdir(parents=True, exist_ok=True)
QUESTION_TYPE_SUFFIX = question_type_slug(question_types)

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("ERROR: Set OPENAI_API_KEY before running.", file=sys.stderr)
    sys.exit(1)


def _report_name(name):
    if not QUESTION_TYPE_SUFFIX:
        return name
    path = Path(name)
    suffixes = "".join(path.suffixes)
    stem = path.name[: -len(suffixes)] if suffixes else path.name
    return f"{stem}__{QUESTION_TYPE_SUFFIX}{suffixes}"


def _artifacts(label):
    output = REPORTS / _report_name(f"{label}_predictions.jsonl")
    details = REPORTS / _report_name(f"{label}_details.jsonl")
    summary = REPORTS / _report_name(f"{label}_summary.json")
    eval_results = REPORTS / _report_name(f"{label}_predictions.jsonl.eval-results-{official_eval_model}")
    return output, details, summary, eval_results


def _load_summary(summary_path):
    with open(summary_path, "r") as f:
        return json.load(f)


def _run(label, memory_enabled, reader_context_mode, reuse_existing=False):
    """Run benchmark_longmemeval_openai.py for one condition, or reuse prior outputs."""
    output, details, summary, eval_results = _artifacts(label)

    if reuse_existing and output.exists() and summary.exists():
        print(f"\n{'='*70}")
        print(f"  Reusing: {label}  ({max_examples} examples, model={openai_model})")
        if question_types:
            print(f"  Question types: {question_types}")
        print(f"{'='*70}\n")
        return _load_summary(summary), str(output), str(details), str(summary), eval_results.exists()

    cmd = [
        sys.executable, str(RUNNER),
        f"--dataset_path={dataset_path}",
        f"--max_examples={max_examples}",
        f"--openai_model={openai_model}",
        f"--output_path={output}",
        f"--details_path={details}",
        f"--summary_path={summary}",
        f"--memory_enabled={memory_enabled}",
        f"--reader_context_mode={reader_context_mode}",
        f"--include_assistant_turns={include_assistant_turns}",
        f"--history_format={history_format}",
        f"--history_granularity={history_granularity}",
        f"--parallel_workers={parallel_workers}",
    ]
    if question_types:
        cmd.append(f"--question_types={question_types}")
    print(f"\n{'='*70}")
    print(f"  Running: {label}  ({max_examples} examples, model={openai_model})")
    if question_types:
        print(f"  Question types: {question_types}")
    print(f"{'='*70}\n")
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    return _load_summary(summary), str(output), str(details), str(summary), False


def _run_official_eval(prediction_path, reuse_existing=False):
    """Run or reuse the official LongMemEval evaluate_qa.py + print_qa_metrics.py."""
    repo = Path(official_repo_path).resolve()
    eval_script = list(repo.rglob("evaluate_qa.py"))[0]
    metrics_script = list(repo.rglob("print_qa_metrics.py"))[0]
    ds = str(Path(dataset_path).resolve())
    pred = str(Path(prediction_path).resolve())
    eval_results = pred + f".eval-results-{official_eval_model}"

    if reuse_existing and Path(eval_results).exists():
        print(f"Reusing official eval results for {Path(prediction_path).name}")
    else:
        subprocess.run(
            [sys.executable, str(eval_script), official_eval_model, pred, ds],
            check=True, cwd=str(eval_script.parent),
        )

    # print_qa_metrics.py  →  stdout
    result = subprocess.run(
        [sys.executable, str(metrics_script), eval_results, ds],
        check=True, cwd=str(metrics_script.parent),
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def _parse_official_metrics(output_text):
    """Parse the stdout of print_qa_metrics.py into a dict."""
    metrics = {}
    for line in output_text.splitlines():
        line = line.strip()
        if line.startswith("Overall Accuracy:"):
            metrics["overall"] = float(line.split(":")[1].strip())
        elif line.startswith("Abstention Accuracy:"):
            parts = line.split(":")[1].strip().split()
            metrics["abstention"] = float(parts[0])
        elif line.startswith("Task-averaged Accuracy:"):
            val = line.split(":")[1].strip()
            metrics["task_avg"] = float(val) if val != "nan" else None
        elif ":" in line and "(" in line:
            # e.g.  "single-session-user: 0.9143 (70)"
            task, rest = line.split(":", 1)
            task = task.strip()
            parts = rest.strip().split()
            if parts and parts[0] != "nan":
                n = int(parts[1].strip("()")) if len(parts) > 1 else 0
                metrics[task] = {"accuracy": float(parts[0]), "n": n}
            else:
                n = int(parts[1].strip("()")) if len(parts) > 1 else 0
                metrics[task] = {"accuracy": None, "n": n}
    return metrics


def _print_table(baseline_metrics, memory_metrics, baseline_summary, memory_summary):
    """Print a markdown-style comparison table."""
    divider = "-" * 80
    example_count = max(baseline_summary.get("examples", 0), memory_summary.get("examples", 0))

    print(f"\n\n{'='*80}")
    print(f"  BENCHMARK COMPARISON  —  {example_count} examples  —  model: {openai_model}")
    if question_types:
        print(f"  Question types: {question_types}")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Metric':<40} {'Baseline':>12} {'Memory':>12} {'Delta':>12}")
    print(divider)

    def _row(label, base_val, mem_val):
        b = f"{base_val:.4f}" if base_val is not None else "n/a"
        m = f"{mem_val:.4f}" if mem_val is not None else "n/a"
        if base_val is not None and mem_val is not None:
            d = mem_val - base_val
            delta = f"{d:+.4f}"
        else:
            delta = "—"
        print(f"{label:<40} {b:>12} {m:>12} {delta:>12}")

    # Overall from official eval
    _row("Overall Accuracy (official)", baseline_metrics.get("overall"), memory_metrics.get("overall"))
    _row("Abstention Accuracy", baseline_metrics.get("abstention"), memory_metrics.get("abstention"))
    _row("Task-averaged Accuracy", baseline_metrics.get("task_avg"), memory_metrics.get("task_avg"))

    print(divider)

    # Per-task
    all_tasks = sorted(set(
        k for k in list(baseline_metrics) + list(memory_metrics)
        if isinstance(baseline_metrics.get(k), dict) or isinstance(memory_metrics.get(k), dict)
    ))
    for task in all_tasks:
        b = baseline_metrics.get(task, {})
        m = memory_metrics.get(task, {})
        b_acc = b.get("accuracy") if isinstance(b, dict) else None
        m_acc = m.get("accuracy") if isinstance(m, dict) else None
        b_n = b.get("n", 0) if isinstance(b, dict) else 0
        m_n = m.get("n", 0) if isinstance(m, dict) else 0
        n_str = f"(n={max(b_n, m_n)})"
        _row(f"  {task} {n_str}", b_acc, m_acc)

    print(divider)

    # Local metrics from summary.json
    _row("Local: contains_match", baseline_summary.get("contains_match"), memory_summary.get("contains_match"))
    _row("Local: exact_match", baseline_summary.get("exact_match"), memory_summary.get("exact_match"))
    _row("Local: token_f1", baseline_summary.get("token_f1"), memory_summary.get("token_f1"))

    print(divider)

    # Token usage
    b_tokens = baseline_summary.get("total_input_tokens", 0)
    m_tokens = memory_summary.get("total_input_tokens", 0)
    savings = (1 - m_tokens / b_tokens) * 100 if b_tokens else 0
    print(f"{'Input tokens (total)':<40} {b_tokens:>12,} {m_tokens:>12,} {savings:>+11.1f}%")
    print(f"{'Output tokens (total)':<40} {baseline_summary.get('total_output_tokens', 0):>12,} {memory_summary.get('total_output_tokens', 0):>12,}")
    print()


# ── Run both conditions ─────────────────────────────────────────────────────
baseline_summary, baseline_pred, _, _, baseline_eval_exists = _run(
    "cmp_baseline", memory_enabled=False, reader_context_mode="full-history", reuse_existing=reuse_baseline,
)
memory_summary, memory_pred, _, _, memory_eval_exists = _run(
    "cmp_memory", memory_enabled=True, reader_context_mode="memory", reuse_existing=reuse_memory,
)

# ── Official eval (if repo provided) ────────────────────────────────────────
if official_repo_path:
    print(f"\n{'='*70}")
    print(f"  Running official GPT-4o evaluation on both conditions...")
    print(f"{'='*70}\n")

    baseline_eval_output = _run_official_eval(baseline_pred, reuse_existing=reuse_baseline and baseline_eval_exists)
    memory_eval_output = _run_official_eval(memory_pred, reuse_existing=reuse_memory and memory_eval_exists)

    print("\n── Baseline official eval ──")
    print(baseline_eval_output)
    print("\n── Memory official eval ──")
    print(memory_eval_output)

    baseline_metrics = _parse_official_metrics(baseline_eval_output)
    memory_metrics = _parse_official_metrics(memory_eval_output)
else:
    # Without official eval, build metrics from local summary
    baseline_metrics = {"overall": baseline_summary.get("contains_match")}
    memory_metrics = {"overall": memory_summary.get("contains_match")}
    print("\nNote: No --official_repo_path set. Using local contains_match as proxy.")
    print("For GPT-4o autoeval, rerun with --official_repo_path=/tmp/LongMemEval-official\n")

# ── Print comparison ────────────────────────────────────────────────────────
_print_table(baseline_metrics, memory_metrics, baseline_summary, memory_summary)
