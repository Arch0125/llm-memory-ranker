"""Print a Supermemory-style leaderboard table from one or more LongMemEval
summary JSONs.

Usage::

    python print_results_table.py \\
        "Full-history (gpt-4o-mini)=reports/longmemeval_protocol/s_full_history_summary.json" \\
        "Memory layer (gpt-4o-mini)=reports/longmemeval_full/s_memory_summary.json" \\
        "Oracle (gpt-4o-mini)=reports/longmemeval_full/oracle_upper_bound_summary.json"

Each positional argument is ``LABEL=PATH``. The columns are the standard
LongMemEval question-type abbreviations (SSU/SSA/SSP/KU/TR/MS) plus Overall.
The default metric is ``contains_match`` (closest local proxy for LLM-as-Judge
accuracy); switch with ``--metric={exact_match,contains_match,token_f1}``.

A trailing "Delta vs ..." row is appended automatically, comparing the last
row to the first row (override with ``--baseline-index`` or pass
``--no-delta``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CATEGORY_ORDER = [
    ("single-session-user",        "SSU"),
    ("single-session-assistant",   "SSA"),
    ("single-session-preference",  "SSP"),
    ("knowledge-update",           "KU"),
    ("temporal-reasoning",         "TR"),
    ("multi-session",              "MS"),
]
CATEGORY_LABELS = [abbr for _, abbr in CATEGORY_ORDER]
COLUMNS = CATEGORY_LABELS + ["Overall"]


def load_summary(path):
    """Load a LongMemEval summary JSON; return None if missing."""
    p = Path(path)
    if not p.exists():
        print(f"warning: {path} does not exist; skipping", file=sys.stderr)
        return None
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def row_values(summary, metric: str):
    """Return [SSU, SSA, SSP, KU, TR, MS, Overall] floats (or None)."""
    if not summary:
        return [None] * len(COLUMNS)
    by_type = summary.get("by_question_type") or {}
    values = []
    for canonical, _ in CATEGORY_ORDER:
        cell = by_type.get(canonical) or {}
        v = cell.get(metric)
        values.append(float(v) if isinstance(v, (int, float)) else None)
    overall = summary.get(metric)
    values.append(float(overall) if isinstance(overall, (int, float)) else None)
    return values


def _fmt_pct(value, *, sign=False):
    if value is None:
        return "  -  "
    pct = value * 100.0
    if sign:
        arrow = "\u2191" if pct >= 0 else "\u2193"
        return f"{arrow}{abs(pct):.2f}%"
    return f"{pct:.2f}%"


def render_table(rows, *, baseline_index=None, label_width=None, col_width=None):
    """Render a leaderboard-style table.

    Parameters
    ----------
    rows : list[(label, values)]
        ``values`` must be a list aligned to :data:`COLUMNS`.
    baseline_index : int | None
        If given (and at least 2 rows exist), appends a trailing "Delta"
        row comparing the *last* row to ``rows[baseline_index]``.
    """
    if label_width is None:
        label_width = max(len(label) for label, _ in rows) if rows else 0
        if baseline_index is not None and len(rows) >= 2:
            label_width = max(
                label_width,
                len(f"Delta ({rows[-1][0]} vs {rows[baseline_index][0]})"),
            )
        label_width = max(label_width, len("Categories"))
    if col_width is None:
        col_width = max(8, max(len(c) for c in COLUMNS))

    header = "| " + "Categories".ljust(label_width) + " |"
    for col in COLUMNS:
        header += " " + col.center(col_width) + " |"
    sep = "|" + "-" * (label_width + 2) + "|"
    for _ in COLUMNS:
        sep += "-" * (col_width + 2) + "|"

    lines = [header, sep]
    for label, values in rows:
        line = "| " + label.ljust(label_width) + " |"
        for v in values:
            line += " " + _fmt_pct(v).center(col_width) + " |"
        lines.append(line)

    if baseline_index is not None and len(rows) >= 2:
        baseline_label, baseline_vals = rows[baseline_index]
        last_label, last_vals = rows[-1]
        deltas = []
        for a, b in zip(baseline_vals, last_vals):
            deltas.append(None if (a is None or b is None) else b - a)
        delta_label = f"Delta ({last_label} vs {baseline_label})"
        line = "| " + delta_label.ljust(label_width) + " |"
        for d in deltas:
            cell = _fmt_pct(d, sign=True) if d is not None else "  -  "
            line += " " + cell.center(col_width) + " |"
        lines.append(line)

    return "\n".join(lines)


def build_rows(label_to_path, metric: str):
    """Convenience: given an iterable of (label, summary_path) pairs, return
    [(label, row_values(...))] suitable for :func:`render_table`."""
    rows = []
    for label, path in label_to_path:
        summary = load_summary(path) if path else None
        rows.append((label, row_values(summary, metric)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "rows",
        nargs="+",
        help="One or more LABEL=summary.json pairs (one row per method).",
    )
    parser.add_argument(
        "--metric",
        default="contains_match",
        choices=["exact_match", "contains_match", "token_f1"],
        help="Which metric to report in the table cells. Default: contains_match.",
    )
    parser.add_argument(
        "--baseline-index",
        type=int,
        default=0,
        help="Row index used as the baseline for the trailing Delta row (default 0).",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Suppress the trailing Delta row.",
    )
    args = parser.parse_args()

    pairs = []
    for raw in args.rows:
        if "=" not in raw:
            print(f"error: expected LABEL=PATH, got {raw!r}", file=sys.stderr)
            sys.exit(2)
        label, _, path = raw.partition("=")
        pairs.append((label.strip(), path))

    if not pairs:
        sys.exit(1)

    rows = build_rows(pairs, args.metric)
    print(f"Metric: {args.metric}\n")
    print(render_table(rows, baseline_index=None if args.no_delta else args.baseline_index))


if __name__ == "__main__":
    main()
