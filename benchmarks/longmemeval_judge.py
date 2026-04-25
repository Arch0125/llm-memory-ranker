"""LongMemEval LLM-as-judge grader.

Mirrors the official scoring methodology from
https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py:

- Question-type-specific evaluation prompts (5 variants for SSU/SSA/MS,
  TR, KU, SSP, plus an abstention template for ``*_abs`` question_ids).
- Default judge model: ``gpt-4o`` (configurable). Same pricing tier as
  Supermemory and Mastra OM publications.
- Each prediction is graded as a binary "yes"/"no" by the judge. Overall
  accuracy is the unweighted mean across the 6 LongMemEval categories.

This module is deliberately small and dependency-light so it can be invoked
from ``run_longmemeval_protocol.py`` after a normal benchmark run, without
requiring a local checkout of the upstream LongMemEval repo.

CLI:

    python -m benchmarks.longmemeval_judge \\
        --predictions=reports/<run>/<cond>_predictions.jsonl \\
        --dataset=data/longmemeval_oracle.json \\
        --output=reports/<run>/<cond>_judge.jsonl \\
        --summary=reports/<run>/<cond>_judge_summary.json \\
        --judge_model=gpt-4o \\
        --judge_workers=8
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from benchmarks.argv_overrides import apply_argv_overrides
from benchmarks.openai_responses import build_responses_payload, create_response


# -----------------------------------------------------------------------------
# Prompt templates copied verbatim from
# https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py
# Keeping them verbatim is important: the reported leaderboard scores
# (Supermemory, Mastra OM, Hindsight, etc.) all use these exact prompts.
# -----------------------------------------------------------------------------
_DEFAULT_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_TEMPORAL_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_KNOWLEDGE_UPDATE_TEMPLATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer.\n\n"
    "Question: {question}\n\n"
    "Correct Answer: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_PREFERENCE_TEMPLATE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {question}\n\n"
    "Rubric: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_ABSTENTION_TEMPLATE = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\n"
    "Question: {question}\n\n"
    "Explanation: {answer}\n\n"
    "Model Response: {response}\n\n"
    "Does the model correctly identify the question as unanswerable? Answer "
    "yes or no only."
)


_TEMPLATE_BY_QTYPE = {
    "single-session-user": _DEFAULT_TEMPLATE,
    "single-session-assistant": _DEFAULT_TEMPLATE,
    "multi-session": _DEFAULT_TEMPLATE,
    "temporal-reasoning": _TEMPORAL_TEMPLATE,
    "knowledge-update": _KNOWLEDGE_UPDATE_TEMPLATE,
    "single-session-preference": _PREFERENCE_TEMPLATE,
}


# Question types in the canonical LongMemEval reporting order.
QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
)


def is_abstention_question(question_id: str) -> bool:
    """Return True if ``question_id`` is an abstention variant.

    LongMemEval marks abstention questions by appending ``_abs`` to the
    base question_id.
    """
    return bool(question_id) and question_id.endswith("_abs")


def build_judge_prompt(
    question_type: str,
    question: str,
    answer: str,
    response: str,
    *,
    abstention: bool,
) -> str:
    """Assemble the LongMemEval judge prompt for one question."""
    if abstention:
        template = _ABSTENTION_TEMPLATE
    else:
        try:
            template = _TEMPLATE_BY_QTYPE[question_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown LongMemEval question_type: {question_type!r}"
            ) from exc
    return template.format(question=question, answer=answer, response=response)


# -----------------------------------------------------------------------------
# Judge runner
# -----------------------------------------------------------------------------


def _parse_yes_no(text: str) -> bool:
    """Parse the judge's free-form reply into a binary label.

    Mirrors the upstream check ('yes' in response.lower()), which is
    intentionally permissive since the judge is instructed to answer
    "yes" or "no" only.
    """
    if not text:
        return False
    return "yes" in text.strip().lower()


def _load_predictions(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_references(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            data = json.load(handle)
        else:
            data = [json.loads(line) for line in handle if line.strip()]
    return {entry["question_id"]: entry for entry in data}


def _grade_one(
    *,
    judge_model: str,
    api_key: str,
    base_url: str,
    timeout_seconds: float,
    max_retries: int,
    question_type: str,
    question: str,
    answer: str,
    response: str,
    abstention: bool,
) -> tuple[bool, str, dict]:
    prompt = build_judge_prompt(
        question_type, question, answer, response, abstention=abstention
    )
    payload = build_responses_payload(
        model=judge_model,
        instructions="You are a strict evaluator. Answer 'yes' or 'no' only.",
        user_input=prompt,
        # OpenAI Responses API requires max_output_tokens >= 16; the judge
        # only ever needs to emit "yes"/"no" so 16 is plenty of slack.
        max_output_tokens=16,
        temperature=0.0,
        top_p=1.0,
    )
    result = create_response(
        api_key=api_key,
        payload=payload,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    raw = result.get("output_text", "") or ""
    label = _parse_yes_no(raw)
    return label, raw, result.get("usage") or {}


def judge_predictions(
    *,
    predictions: list[dict],
    references: dict[str, dict],
    judge_model: str = "gpt-4o",
    api_key: str | None = None,
    base_url: str = "https://api.openai.com/v1",
    workers: int = 8,
    timeout_seconds: float = 120.0,
    max_retries: int = 5,
    progress_every: int = 25,
) -> list[dict]:
    """Grade ``predictions`` with a LongMemEval-style LLM judge.

    Returns one record per prediction with fields:

    - question_id, question_type, question, answer, hypothesis
    - judge_label: bool
    - judge_raw: str (raw judge reply)
    - judge_input_tokens / judge_output_tokens / judge_total_tokens

    Predictions whose ``question_id`` is missing from ``references`` are
    skipped (with a warning printed to stderr).
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set OPENAI_API_KEY in the environment or pass --judge_api_key=..."
        )

    print_lock = threading.Lock()

    def _process(idx: int, entry: dict) -> dict | None:
        qid = entry.get("question_id")
        ref = references.get(qid)
        if ref is None:
            with print_lock:
                print(
                    f"[longmemeval_judge] skipping {qid!r}: not in references",
                    file=sys.stderr,
                )
            return None
        qtype = ref["question_type"]
        question = ref["question"]
        answer = ref["answer"]
        hypothesis = entry.get("hypothesis") or entry.get("prediction") or ""
        abstention = is_abstention_question(qid)
        label, raw, usage = _grade_one(
            judge_model=judge_model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            question_type=qtype,
            question=question,
            answer=answer,
            response=hypothesis,
            abstention=abstention,
        )
        return {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "answer": answer,
            "hypothesis": hypothesis,
            "abstention": abstention,
            "judge_model": judge_model,
            "judge_label": bool(label),
            "judge_raw": raw,
            "judge_input_tokens": usage.get("input_tokens", 0),
            "judge_output_tokens": usage.get("output_tokens", 0),
            "judge_total_tokens": usage.get("total_tokens", 0),
        }

    results: list[dict] = []
    if workers <= 1:
        for idx, entry in enumerate(predictions):
            row = _process(idx, entry)
            if row is not None:
                results.append(row)
            if progress_every and (idx + 1) % progress_every == 0:
                print(
                    f"[longmemeval_judge] graded {idx + 1}/{len(predictions)}",
                    flush=True,
                )
        return results

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process, idx, entry): idx
            for idx, entry in enumerate(predictions)
        }
        for fut in as_completed(futures):
            row = fut.result()
            completed += 1
            if row is not None:
                results.append(row)
            if progress_every and completed % progress_every == 0:
                print(
                    f"[longmemeval_judge] graded {completed}/{len(predictions)}",
                    flush=True,
                )
    # Preserve input order so the output is deterministic.
    order = {entry.get("question_id"): idx for idx, entry in enumerate(predictions)}
    results.sort(key=lambda row: order.get(row["question_id"], 0))
    return results


def summarize_results(results: Iterable[dict]) -> dict:
    """Compute per-category and overall LongMemEval-style accuracy.

    Overall accuracy follows Supermemory/Mastra reporting: unweighted
    mean across the 6 categories (categories with no examples are
    excluded). Also returns a micro-averaged accuracy for completeness.
    """
    by_type: dict[str, list[bool]] = {}
    abstention_labels: list[bool] = []
    total_input = 0
    total_output = 0
    total = 0
    micro = 0
    for row in results:
        labels = by_type.setdefault(row["question_type"], [])
        labels.append(bool(row["judge_label"]))
        if row.get("abstention"):
            abstention_labels.append(bool(row["judge_label"]))
        total_input += int(row.get("judge_input_tokens") or 0)
        total_output += int(row.get("judge_output_tokens") or 0)
        total += 1
        if row["judge_label"]:
            micro += 1

    by_question_type = {}
    for qtype in QUESTION_TYPES:
        labels = by_type.get(qtype, [])
        if not labels:
            continue
        by_question_type[qtype] = {
            "examples": len(labels),
            "judge_accuracy": sum(labels) / len(labels),
        }
    # Surface any other types (e.g. abstention slices) without breaking
    # the canonical 6-category overall.
    for qtype, labels in by_type.items():
        if qtype in by_question_type:
            continue
        by_question_type[qtype] = {
            "examples": len(labels),
            "judge_accuracy": sum(labels) / len(labels),
        }

    canonical_accuracies = [
        by_question_type[qtype]["judge_accuracy"]
        for qtype in QUESTION_TYPES
        if qtype in by_question_type
    ]
    overall_unweighted = (
        statistics.fmean(canonical_accuracies) if canonical_accuracies else 0.0
    )
    overall_micro = (micro / total) if total else 0.0

    return {
        "examples": total,
        "judge_accuracy": overall_unweighted,
        "judge_accuracy_micro": overall_micro,
        "abstention_judge_accuracy": (
            (sum(abstention_labels) / len(abstention_labels))
            if abstention_labels
            else None
        ),
        "judge_input_tokens": total_input,
        "judge_output_tokens": total_output,
        "judge_total_tokens": total_input + total_output,
        "by_question_type": by_question_type,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
predictions = ""
dataset = ""
output = ""
summary = ""
judge_model = "gpt-4o"
judge_api_key = ""
judge_base_url = "https://api.openai.com/v1"
judge_workers = 8
judge_timeout_seconds = 120.0
judge_max_retries = 5


def _print_summary(summary_data: dict) -> None:
    print(f"\nLongMemEval LLM-judge accuracy ({summary_data.get('examples', 0)} examples):")
    print(
        f"  overall (unweighted mean over categories): "
        f"{summary_data.get('judge_accuracy', 0.0):.4f}"
    )
    print(
        f"  overall (micro): {summary_data.get('judge_accuracy_micro', 0.0):.4f}"
    )
    by_type = summary_data.get("by_question_type", {})
    for qtype in QUESTION_TYPES:
        if qtype not in by_type:
            continue
        bucket = by_type[qtype]
        print(
            f"  {qtype}: {bucket['judge_accuracy']:.4f} "
            f"({int(round(bucket['judge_accuracy'] * bucket['examples']))}/{bucket['examples']})"
        )
    abst = summary_data.get("abstention_judge_accuracy")
    if abst is not None:
        print(f"  abstention: {abst:.4f}")


def main() -> None:
    apply_argv_overrides(globals())

    if not predictions:
        raise SystemExit("--predictions=<path> is required")
    if not dataset:
        raise SystemExit("--dataset=<path> is required")

    pred_rows = _load_predictions(predictions)
    refs = _load_references(dataset)

    api_key = judge_api_key or os.environ.get("OPENAI_API_KEY", "")

    rows = judge_predictions(
        predictions=pred_rows,
        references=refs,
        judge_model=judge_model,
        api_key=api_key,
        base_url=judge_base_url,
        workers=judge_workers,
        timeout_seconds=judge_timeout_seconds,
        max_retries=judge_max_retries,
    )
    summary_data = summarize_results(rows)

    out_path = output or (predictions + f".judge-{judge_model}.jsonl")
    summary_path = summary or (predictions + f".judge-{judge_model}.summary.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_data, handle, indent=2, sort_keys=True)
    print(f"Wrote per-question judge labels to {out_path}")
    print(f"Wrote judge summary to {summary_path}")
    _print_summary(summary_data)


if __name__ == "__main__":
    main()
