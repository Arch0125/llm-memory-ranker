"""Benchmark-agnostic runner for the generic memory engine.

This script exercises the memory layer end-to-end against any registered
benchmark adapter (`longmemeval`, `locomo`, `memorybench`, ...). It does NOT
call any external LLM; it reports retrieval + deterministic-solver metrics so
you can iterate on the memory layer without spending API credits.

Usage:
    python run_benchmark.py --benchmark=longmemeval --data=data/longmemeval_oracle.json --max=20

Modes:
    retrieval  -> score top-k retrieval against gold session ids when available.
    solver     -> only evaluate the deterministic solver (no LLM).
    full       -> retrieval + solver (default).

Outputs:
    A JSONL of per-example results and a small summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from memory import (
    MemoryAwareConfig,
    MemoryAwareInference,
    SQLiteMemoryStore,
    build_embedder,
)
from memory.adapters import get_adapter, list_adapters
from memory.evidence import build_evidence_table
from memory.solver import solve


def _ingest(memory: MemoryAwareInference, items):
    for item in items:
        memory.remember(
            text=item.get("text", ""),
            memory_type=item.get("memory_type", "ephemeral"),
            importance=float(item.get("importance", 0.5)),
            metadata=item.get("metadata") or {},
        )


def _retrieval_metrics(plan, hits, instance):
    """Heuristic retrieval metric: did we retrieve any memory tied to a gold session?"""
    gold_session_ids = set(instance.extras.get("answer_session_ids") or [])
    if not gold_session_ids:
        return None
    retrieved_session_ids = {
        (hit.record.metadata or {}).get("session_id")
        for hit in hits
        if (hit.record.metadata or {}).get("session_id")
    }
    if not retrieved_session_ids:
        return {"recall": 0.0}
    overlap = len(gold_session_ids & retrieved_session_ids)
    return {
        "recall": overlap / len(gold_session_ids),
        "retrieved_sessions": list(retrieved_session_ids)[:8],
        "gold_sessions": list(gold_session_ids),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help=f"One of: {list_adapters() or '<auto-discovered>'}")
    parser.add_argument("--data", required=True, help="Path to dataset file")
    parser.add_argument("--out", default="reports/run_benchmark.jsonl")
    parser.add_argument("--mode", choices=["retrieval", "solver", "full"], default="full")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--granularity", default="hybrid")
    parser.add_argument("--user-id", default="benchmark")
    parser.add_argument("--embedder", default="benchmark-auto")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-items", type=int, default=12)
    parser.add_argument("--similarity-threshold", type=float, default=0.08)
    parser.add_argument("--critic-threshold", type=float, default=0.44)
    parser.add_argument("--maybe-threshold", type=float, default=0.34)
    parser.add_argument("--token-budget", type=int, default=1200)
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--bm25", action="store_true", default=True)
    parser.add_argument("--no-bm25", dest="bm25", action="store_false")
    parser.add_argument("--query-expansion", action="store_true", default=True)
    parser.add_argument("--no-query-expansion", dest="query_expansion", action="store_false")
    parser.add_argument("--diversity", type=float, default=0.0)
    parser.add_argument("--rerank", action="store_true", default=False)
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = parser.parse_args()

    adapter = get_adapter(args.benchmark, granularity=args.granularity)
    instances = adapter.filter(
        adapter.load(args.data),
        question_types=args.question_types,
        start_index=args.start,
        max_examples=args.max,
    )

    embedder = build_embedder(args.embedder)
    reranker = None
    if args.rerank:
        from memory.rerank import CrossEncoderReranker

        reranker = CrossEncoderReranker.try_load(args.rerank_model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": args.benchmark,
        "instances": 0,
        "metric_sums": {},
        "metric_counts": {},
        "solver_resolved": 0,
        "retrieval_recalls": [],
    }

    started = time.time()
    with out_path.open("w", encoding="utf-8") as out_handle:
        for instance in instances:
            with tempfile.TemporaryDirectory() as tmp:
                store = SQLiteMemoryStore(os.path.join(tmp, "memory.sqlite"))
                config = MemoryAwareConfig(
                    user_id=args.user_id,
                    top_k=args.top_k,
                    max_items=args.max_items,
                    similarity_threshold=args.similarity_threshold,
                    critic_threshold=args.critic_threshold,
                    maybe_threshold=args.maybe_threshold,
                    memory_token_budget=args.token_budget,
                    fusion_strategy=args.fusion,
                    use_bm25=args.bm25,
                    use_query_expansion=args.query_expansion,
                    diversity=args.diversity,
                    rerank_top_k=args.top_k if args.rerank else 0,
                )
                memory = MemoryAwareInference(
                    store=store,
                    embedder=embedder,
                    config=config,
                    reranker=reranker,
                )

                _ingest(memory, list(adapter.ingest(instance)))
                plan = adapter.analyze(instance)

                ranked = memory.rank_hits(plan.query_text, plan=plan)
                chosen = memory.choose_hits(ranked)
                selected, _ = memory.budget_hits(chosen)

                solver_result = None
                if args.mode in {"solver", "full"} and selected:
                    solver_result = solve(plan, selected)

                retrieval = None
                if args.mode in {"retrieval", "full"}:
                    retrieval = _retrieval_metrics(plan, selected, instance)

                record = {
                    "instance_id": instance.instance_id,
                    "question": instance.question,
                    "question_type": instance.question_type,
                    "plan_kind": plan.reasoning_kind,
                    "selected_count": len(selected),
                    "retrieval": retrieval,
                    "solver": (
                        {
                            "resolved": solver_result.resolved,
                            "answer": solver_result.answer,
                            "confidence": solver_result.confidence,
                            "mode": solver_result.mode,
                            "rationale": solver_result.rationale,
                        } if solver_result else None
                    ),
                    "evidence_preview": build_evidence_table(selected[:6])[:500],
                }

                if solver_result and solver_result.resolved:
                    summary["solver_resolved"] += 1
                    metrics = adapter.score(plan, solver_result.answer, instance)
                    record["scores"] = metrics
                    for key, value in metrics.items():
                        if value is None:
                            continue
                        summary["metric_sums"][key] = summary["metric_sums"].get(key, 0.0) + value
                        summary["metric_counts"][key] = summary["metric_counts"].get(key, 0) + 1

                if retrieval and "recall" in retrieval:
                    summary["retrieval_recalls"].append(retrieval["recall"])

                out_handle.write(json.dumps(record) + "\n")
                summary["instances"] += 1
                store.close()

    elapsed = time.time() - started
    summary["elapsed_seconds"] = round(elapsed, 2)
    if summary["metric_counts"]:
        summary["averages"] = {
            key: round(summary["metric_sums"][key] / summary["metric_counts"][key], 4)
            for key in summary["metric_counts"]
        }
    if summary["retrieval_recalls"]:
        recalls = summary["retrieval_recalls"]
        summary["retrieval_recall_mean"] = round(sum(recalls) / len(recalls), 4)
    summary.pop("metric_sums", None)
    summary.pop("metric_counts", None)
    summary.pop("retrieval_recalls", None)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
