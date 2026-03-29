#!/usr/bin/env python3
"""
Run a head-to-head comparison of the full-history baseline vs the memory layer
on the first N examples from the LongMemEval-S dataset, then print a comparison table.

Usage:
    ./venv/bin/python run_benchmark_comparison.py
    ./venv/bin/python run_benchmark_comparison.py --max_examples=50
    ./venv/bin/python run_benchmark_comparison.py --max_examples=100 --official_repo_path=/tmp/LongMemEval-official

Set OPENAI_API_KEY in your environment before running.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


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
exec(open("configurator.py").read())  # overrides from CLI
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "benchmark_longmemeval_openai.py"
REPORTS = ROOT / reports_dir
REPORTS.mkdir(parents=True, exist_ok=True)

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("ERROR: Set OPENAI_API_KEY before running.", file=sys.stderr)
    sys.exit(1)


def _run(label, memory_enabled, reader_context_mode):
    """Run benchmark_longmemeval_openai.py for one condition."""
    output   = str(REPORTS / f"{label}_predictions.jsonl")
    details  = str(REPORTS / f"{label}_details.jsonl")
    summary  = str(REPORTS / f"{label}_summary.json")

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
    ]
    print(f"\n{'='*70}")
    print(f"  Running: {label}  ({max_examples} examples, model={openai_model})")
    print(f"{'='*70}\n")
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    with open(summary, "r") as f:
        return json.load(f), output, details, summary


def _run_official_eval(prediction_path):
    """Run the official LongMemEval evaluate_qa.py + print_qa_metrics.py."""
    repo = Path(official_repo_path).resolve()
    eval_script = list(repo.rglob("evaluate_qa.py"))[0]
    metrics_script = list(repo.rglob("print_qa_metrics.py"))[0]
    ds = str(Path(dataset_path).resolve())
    pred = str(Path(prediction_path).resolve())

    # evaluate_qa.py  →  creates .eval-results-{model}
    subprocess.run(
        [sys.executable, str(eval_script), official_eval_model, pred, ds],
        check=True, cwd=str(eval_script.parent),
    )
    eval_results = pred + f".eval-results-{official_eval_model}"

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

    print(f"\n\n{'='*80}")
    print(f"  BENCHMARK COMPARISON  —  {max_examples} examples  —  model: {openai_model}")
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
baseline_summary, baseline_pred, _, _ = _run(
    "cmp_baseline", memory_enabled=False, reader_context_mode="full-history",
)
memory_summary, memory_pred, _, _ = _run(
    "cmp_memory", memory_enabled=True, reader_context_mode="memory",
)

# ── Official eval (if repo provided) ────────────────────────────────────────
if official_repo_path:
    print(f"\n{'='*70}")
    print(f"  Running official GPT-4o evaluation on both conditions...")
    print(f"{'='*70}\n")

    baseline_eval_output = _run_official_eval(baseline_pred)
    memory_eval_output = _run_official_eval(memory_pred)

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
