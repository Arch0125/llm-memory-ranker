"""Compare two LongMemEval protocol report dirs side-by-side.

Usage::

    python compare_protocol_runs.py BASELINE_DIR CANDIDATE_DIR

Prints overall + per-question-type metrics for ``s_full_history``,
``s_memory`` and ``oracle_upper_bound``. When the candidate dir is missing
a condition's summary, this script automatically falls back to the
baseline dir's value for that condition — so the typical workflow is:

    1. Run a full baseline once into ``reports/longmemeval_protocol/``.
    2. For each new memory-engine config, run **only** ``s_memory`` into
       a fresh dir with ``--run_conditions=s_memory --reports_dir=...``.
    3. Compare; the unchanged conditions (``s_full_history``,
       ``oracle_upper_bound``) are reused from the baseline automatically.

If a condition has no summary in *either* dir it is skipped silently.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


CONDITIONS = ("s_full_history", "s_memory", "oracle_upper_bound")
METRICS = ("exact_match", "contains_match", "token_f1", "abstention_accuracy")


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def _row(label, baseline, candidate, metric):
    a = baseline.get(metric) if baseline else None
    b = candidate.get(metric) if candidate else None
    a_str = f"{a:.3f}" if isinstance(a, (int, float)) else "  -  "
    b_str = f"{b:.3f}" if isinstance(b, (int, float)) else "  -  "
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        delta = b - a
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.3f}"
    else:
        delta_str = "  -  "
    print(f"  {label:<30s} {metric:<22s} {a_str}  ->  {b_str}   ({delta_str})")


def _print_block(title, baseline, candidate, *, candidate_is_fallback=False):
    suffix = "  (candidate reused from baseline)" if candidate_is_fallback else ""
    print(f"\n=== {title}{suffix} ===")
    for metric in METRICS:
        _row("overall", baseline, candidate, metric)
    by_a = (baseline or {}).get("by_question_type") or {}
    by_b = (candidate or {}).get("by_question_type") or {}
    qtypes = sorted(set(by_a) | set(by_b))
    for qt in qtypes:
        a = by_a.get(qt) or {}
        b = by_b.get(qt) or {}
        for metric in ("exact_match", "contains_match", "token_f1"):
            _row(f"  {qt}", a, b, metric)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    base_dir = Path(sys.argv[1]).resolve()
    cand_dir = Path(sys.argv[2]).resolve()
    print(f"baseline:  {base_dir}")
    print(f"candidate: {cand_dir}")
    for cond in CONDITIONS:
        a = _load(base_dir / f"{cond}_summary.json")
        b = _load(cand_dir / f"{cond}_summary.json")
        candidate_is_fallback = False
        if b is None and a is not None:
            # Reuse the baseline's value for unchanged conditions so the
            # caller doesn't have to re-run them.
            b = a
            candidate_is_fallback = True
        if a is None and b is None:
            continue
        _print_block(cond, a, b, candidate_is_fallback=candidate_is_fallback)


if __name__ == "__main__":
    main()
