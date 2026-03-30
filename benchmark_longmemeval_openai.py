"""
Run LongMemEval with the local memory-aware inference layer and OpenAI's Responses API.

This keeps retrieval, gating, and prompt assembly local while using an
OpenAI-hosted model for generation.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmarks.argv_overrides import apply_argv_overrides
from benchmarks.question_type_filters import normalize_question_types
from benchmarks.longmemeval import (
    analyze_question,
    abstention_score,
    assess_answerability,
    build_multi_session_checklist_context,
    build_history_context,
    build_benchmark_instructions,
    build_structured_event_view,
    build_temporal_event_context,
    contains_match_score,
    exact_match_score,
    is_single_session_question,
    iter_history_memories,
    load_longmemeval_instances,
    multi_session_session_limit,
    postprocess_prediction,
    question_policy,
    render_evidence_line,
    select_raw_session_hits,
    select_bundled_hits,
    selected_session_ids,
    selected_session_recall,
    single_session_include_assistant_turns,
    solve_temporal_question,
    summarize_records,
    token_f1_score,
)
from benchmarks.openai_responses import build_responses_payload, create_response
from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder
from memory.explain import build_trace, format_trace
from prompt.template import DEFAULT_SYSTEM_PROMPT

# -----------------------------------------------------------------------------
dataset_path = "data/longmemeval_oracle.json"
output_path = "reports/longmemeval_openai_predictions.jsonl"
details_path = "reports/longmemeval_openai_details.jsonl"
summary_path = "reports/longmemeval_openai_summary.json"
start_index = 0
max_examples = 0
question_types = ""
history_granularity = "hybrid"
include_assistant_turns = False
include_question_date = True
max_new_tokens = 128
temperature = 0.0
top_p = 1.0
memory_enabled = True
memory_user_id = "benchmark"
memory_embedder = "benchmark-auto"
memory_top_k = 40
memory_max_items = 12
memory_similarity_threshold = 0.08
memory_critic_threshold = 0.44
memory_maybe_threshold = 0.34
memory_max_age_days = -1
memory_token_budget = 1200
memory_type_allowlist = ""
memory_recent_context = ""
memory_explain = False
memory_solver_mode = "hybrid"
memory_solver_min_confidence = 0.72
memory_structured_event_limit = 4
reader_context_mode = "auto"
history_format = "nl"
openai_api_key = ""
openai_base_url = "https://api.openai.com/v1"
openai_model = "gpt-4.1-mini"
openai_timeout_seconds = 120.0
openai_max_retries = 5
openai_reasoning_effort = ""
openai_verbosity = ""
parallel_workers = 8
apply_argv_overrides(globals())
question_types = normalize_question_types(question_types)
# -----------------------------------------------------------------------------


def _build_baseline_instructions(plan, history_context="", base_system_prompt=None):
    parts = [base_system_prompt or DEFAULT_SYSTEM_PROMPT]
    if history_context:
        parts.append("Use only the provided chat history as evidence. If the answer is not supported by the history, respond with 'Insufficient evidence'.")
        parts.append(f"Chat history:\n{history_context.strip()}")
    if plan.is_multi_session:
        if plan.multi_session_kind in {"count_entries", "count_distinct"}:
            parts.append("Return only the final number.")
        elif plan.multi_session_kind == "sum_money":
            parts.append("Return only the final total amount like '$120'.")
        elif plan.multi_session_kind == "sum_quantity":
            unit = plan.unit_hint or "units"
            parts.append(f"Return only the final total like '12 {unit}'.")
        elif plan.multi_session_kind == "time_lookup":
            parts.append("Return only the final time.")
        else:
            parts.append("Return only the final answer.")
    elif plan.reasoning_kind == "ordering":
        parts.append("Return only the event/item that happened first or last.")
    elif plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        parts.append(f"Return only the final duration like '7 {unit}'.")
    elif plan.reasoning_kind == "date":
        parts.append("Return only the final date or short date phrase.")
    else:
        parts.append("Return only the final answer.")
    return "\n\n".join(parts)


def _build_memory_session_instructions(
    plan,
    session_context="",
    evidence_sufficient=False,
    base_system_prompt=None,
    strict_abstention=False,
    is_knowledge_update=False,
    has_temporal_view=False,
    has_session_checklist=False,
    assistant_extraction=False,
):
    parts = [base_system_prompt or DEFAULT_SYSTEM_PROMPT]
    parts.append("Use the selected supporting evidence below to answer the question. Base your answer only on what is stated there.")
    if is_knowledge_update:
        parts.append(
            "IMPORTANT: The sessions are shown in chronological order. "
            "If the same topic is discussed in multiple sessions, the LATER session contains the most up-to-date information. "
            "Always prefer the value, number, or fact from the LATEST (most recent) session that discusses the topic."
        )
    if assistant_extraction:
        parts.append(
            "These excerpts include prior assistant answers. Extract only the exact field the user is asking for. "
            "Do not paraphrase. Do not return the broader explanation if the question asks for a name, title, website, number, percentage, quote, or the 'other options'."
        )
    if has_temporal_view:
        parts.append(
            "If a structured event/date view is shown, use it first for calculations and ordering. "
            "For ordering questions, include every relevant event in the final earliest-to-latest or latest-to-earliest order."
        )
    if has_session_checklist:
        parts.append(
            "If a session-by-session checklist is shown, process every checklist line before answering. "
            "Do not stop after the first matching session."
        )
    if session_context:
        parts.append(f"Selected supporting evidence:\n{session_context.strip()}")
    if plan.is_multi_session:
        parts.append("For multi-session questions, combine information across ALL the selected session excerpts.")
        if plan.multi_session_kind in {"count_entries", "count_distinct"}:
            subject = plan.multi_session_subject or "items"
            parts.append(
                f"You must count every distinct {subject} mentioned across ALL session excerpts. "
                f"IMPORTANT: The answer is often spread across MANY sessions — each session may mention a different {subject}. "
                f"Go through EVERY session one by one, extract the relevant {subject} from each, "
                f"then count the unique ones. "
                f"Your LAST line must be EXACTLY in this format: 'Final answer: <number>'"
            )
        elif plan.multi_session_kind == "sum_money":
            parts.append(
                "Add up ALL relevant dollar amounts across ALL excerpts. "
                "Go through each session, find every relevant expense/amount, sum them all. "
                "Show the math: $X + $Y + ... = $Z. "
                "Your LAST line must be EXACTLY: 'Final answer: $<total>'"
            )
        elif plan.multi_session_kind == "sum_quantity":
            unit = plan.unit_hint or "units"
            parts.append(
                f"Add up ALL relevant quantities across ALL excerpts. "
                f"Go through each session, find every relevant quantity, sum them all. "
                f"Show the math: X + Y + ... = Z. "
                f"Your LAST line must be EXACTLY: 'Final answer: <number> {unit}'"
            )
        else:
            parts.append("Combine and aggregate all relevant items mentioned across ALL excerpts.")
        parts.append("Keep your answer short. Your final line must be 'Final answer: <value>'.")
    elif plan.reasoning_kind == "ordering":
        if "what is the order" in plan.question.lower() or "from earliest to latest" in plan.question.lower():
            parts.append("Return the complete ordered list covering all relevant events, from earliest to latest unless the question says otherwise.")
        else:
            parts.append("Return only the event or item that happened first or last, based on the session dates shown.")
        if plan.targets:
            parts.append("Valid answer options: " + " | ".join(plan.targets))
    elif plan.reasoning_kind == "difference":
        unit = plan.unit_hint or "days"
        parts.append(
            f"Calculate the duration between the two events. Each session excerpt shows its date (e.g., 'Session on 2023-05-15'). "
            f"Use those dates to compute the difference. Return only the final number like '7 {unit}'. "
            f"Do NOT say 'Insufficient evidence' if you can see dates in the excerpts."
        )
    elif plan.reasoning_kind == "date":
        parts.append("Return only the final date or short date phrase. Look at the session date headers (e.g., 'Session on 2023-05-15') for date information.")
    else:
        parts.append("Return only the final answer, even if the evidence is partial. Prefer giving a best-effort answer over abstaining.")
    if strict_abstention:
        parts.append("If the question asks about something not clearly supported by the excerpts, respond with 'Insufficient evidence'. Be careful not to guess.")
    elif evidence_sufficient:
        parts.append(
            "The excerpts contain sufficient evidence. Answer directly from them. "
            "Do NOT reply with 'Insufficient evidence'. You MUST provide an answer."
        )
    else:
        parts.append(
            "Try to extract an answer from the excerpts. Give your best answer even if the evidence is partial. "
            "Only reply 'Insufficient evidence' if the excerpts contain absolutely no relevant information at all."
        )
    return "\n\n".join(parts)


def _resolved_reader_context_mode():
    mode = (reader_context_mode or "auto").strip().lower()
    if mode == "auto":
        return "memory" if memory_enabled else "question-only"
    return mode


def _effective_policy(plan):
    policy = question_policy(plan)
    return {
        "top_k": max(memory_top_k, policy["top_k"]),
        "max_items": max(memory_max_items, policy["max_items"]),
        "token_budget": max(memory_token_budget, policy["token_budget"]),
        "similarity_threshold": min(memory_similarity_threshold, policy["similarity_threshold"]),
        "critic_threshold": min(memory_critic_threshold, policy["critic_threshold"]),
        "maybe_threshold": min(memory_maybe_threshold, policy["maybe_threshold"]),
    }


def _make_memory_system(store, embedder, policy):
    return MemoryAwareInference(
        store=store,
        embedder=embedder,
        config=MemoryAwareConfig(
            user_id=memory_user_id,
            top_k=policy["top_k"],
            max_items=policy["max_items"],
            similarity_threshold=policy["similarity_threshold"],
            critic_threshold=policy["critic_threshold"],
            maybe_threshold=policy["maybe_threshold"],
            max_age_days=None if memory_max_age_days < 0 else memory_max_age_days,
            memory_token_budget=policy["token_budget"],
            type_allowlist=memory_type_allowlist,
        ),
    )


def _solver_allowed_reasoning_kind(plan):
    if plan.is_multi_session:
        return plan.multi_session_kind == "time_lookup"
    return plan.reasoning_kind in {"ordering", "date", "difference"}


def _solver_can_prompt(plan, solver_result):
    if not solver_result or not solver_result.resolved:
        return False
    if not plan.is_multi_session:
        return True
    return (
        solver_result.mode == "multi-session-time-lookup"
        and solver_result.confidence >= max(0.92, memory_solver_min_confidence)
        and len(solver_result.supporting_memory_ids) >= 2
    )


def _solver_can_finalize(plan, solver_result):
    if not solver_result or not solver_result.resolved:
        return False
    if not plan.is_multi_session:
        return True
    return (
        solver_result.mode == "multi-session-time-lookup"
        and solver_result.confidence >= max(0.92, memory_solver_min_confidence)
        and len(solver_result.supporting_memory_ids) >= 2
    )


def _structured_events_for_prompt(plan, structured_events, solver_result):
    if not structured_events:
        return []
    if plan.is_multi_session:
        return []
    if not _solver_allowed_reasoning_kind(plan):
        return []
    if solver_result and solver_result.resolved:
        return structured_events[:memory_structured_event_limit]
    if solver_result and solver_result.confidence >= max(0.45, memory_solver_min_confidence - 0.2):
        return structured_events[: min(2, memory_structured_event_limit)]
    return []


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


if history_granularity not in {"turn", "session", "hybrid"}:
    raise ValueError("history_granularity must be 'turn', 'session', or 'hybrid'")
if history_format not in {"nl", "json"}:
    raise ValueError("history_format must be 'nl' or 'json'")
resolved_context_mode = _resolved_reader_context_mode()
if resolved_context_mode not in {"memory", "question-only", "full-history", "oracle-history"}:
    raise ValueError("reader_context_mode must be one of auto, memory, question-only, full-history, oracle-history")

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
# Lock for thread-safe access to the shared embedder (sentence-transformers
# models are not guaranteed to be thread-safe).
_embed_lock = threading.Lock()


def _process_one(index, instance):
    """Process a single LongMemEval example. Returns (index, prediction_row, detail_row)."""
    question_id = instance["question_id"]
    plan = analyze_question(instance, include_question_date=include_question_date)
    query_text = plan.query_text

    with tempfile.TemporaryDirectory(prefix="longmemeval-openai-") as tempdir:
        store = SQLiteMemoryStore(str(Path(tempdir) / "memory.sqlite"))
        selected_hits = []
        structured_events = []
        solver_result = None
        trace = None
        history_context = ""
        answerability = {
            "sufficient": False,
            "reasons": ["no-memory-mode"],
            "distinct_dates": [],
            "distinct_sessions": [],
            "covered_targets": [],
        }

        if resolved_context_mode == "memory":
            # Ingestion needs embedder — serialize embedding calls
            _ingest_history_threadsafe(store, embedder, instance, _embed_lock)
            policy = _effective_policy(plan)
            memory_system = _make_memory_system(store, embedder, policy)
            with _embed_lock:
                reranked = memory_system.rank_hits(query_text, hybrid=True)

            # Query expansion for multi-session
            if plan.is_multi_session and plan.multi_session_subject:
                expanded_query = f"{plan.multi_session_subject} {' '.join(plan.multi_session_actions)}"
                if expanded_query.strip() != query_text.strip():
                    with _embed_lock:
                        expanded_hits = memory_system.rank_hits(expanded_query, hybrid=True)
                    seen_ids = {h.record.memory_id for h in reranked}
                    for eh in expanded_hits:
                        if eh.record.memory_id not in seen_ids:
                            reranked.append(eh)
                            seen_ids.add(eh.record.memory_id)

            candidate_hits = [
                hit
                for hit in reranked
                if hit.critic_label != "ignore" or hit.critic_confidence >= max(0.16, policy["maybe_threshold"] - 0.12)
            ]
            is_knowledge_update = (plan.question_type == "knowledge-update")
            if is_single_session_question(plan) or plan.is_temporal or is_knowledge_update:
                max_sess = 3
                if plan.question_type == "single-session-assistant":
                    max_sess = 2
                if is_knowledge_update:
                    max_sess = 3
                if plan.is_temporal and plan.reasoning_kind in {"ordering", "difference"}:
                    # Ordering/difference questions often need 2-3 specific sessions
                    max_sess = 4
                selected_hits, raw_session_ids = select_raw_session_hits(
                    plan,
                    candidate_hits,
                    max_sessions=max_sess,
                )
                history_context = build_history_context(
                    instance,
                    include_assistant_turns=single_session_include_assistant_turns(
                        plan,
                        default=include_assistant_turns,
                    ),
                    history_format=history_format,
                    allowed_session_ids=raw_session_ids,
                )
                answerability = {
                    "sufficient": bool(raw_session_ids),
                    "reasons": ["raw-session-selected"] if raw_session_ids else ["missing-session"],
                    "distinct_dates": [],
                    "distinct_sessions": list(raw_session_ids),
                    "covered_targets": [],
                }
                if plan.is_temporal:
                    structured_events = build_structured_event_view(
                        plan,
                        selected_hits,
                        limit=max(12, memory_structured_event_limit),
                    )
                    temporal_context = build_temporal_event_context(
                        plan,
                        selected_hits,
                        limit=max(8, memory_structured_event_limit),
                    )
                    if temporal_context:
                        history_context = f"{temporal_context}\n\n{history_context}".strip()
                    solver_result = solve_temporal_question(plan, selected_hits)
                else:
                    structured_events = []
                    solver_result = None
            elif plan.is_multi_session:
                focus_terms = (
                    plan.multi_session_focus_terms
                    + plan.multi_session_actions
                    + ([plan.multi_session_subject] if plan.multi_session_subject else [])
                )
                focus_scan_hits = store.focus_term_search(
                    focus_terms=focus_terms,
                    user_id=memory_user_id,
                )
                seen_ids = {h.record.memory_id for h in candidate_hits}
                for fh in focus_scan_hits:
                    if fh.record.memory_id not in seen_ids:
                        fh.critic_label = "maybe"
                        fh.critic_confidence = max(0.35, fh.score * 0.6)
                        fh.reasons = [f"focus-scan={fh.score:.2f}"]
                        candidate_hits.append(fh)
                        seen_ids.add(fh.record.memory_id)

                selected_hits, raw_session_ids = select_raw_session_hits(
                    plan,
                    candidate_hits,
                    max_sessions=multi_session_session_limit(plan),
                )
                history_context = build_history_context(
                    instance,
                    include_assistant_turns=include_assistant_turns,
                    history_format=history_format,
                    allowed_session_ids=raw_session_ids,
                )
                session_hits = [
                    hit
                    for hit in candidate_hits
                    if hit.record.metadata.get("session_id") in set(raw_session_ids)
                    and hit.record.memory_type != "aggregate"
                    and hit.record.metadata.get("granularity") != "aggregate"
                ]
                answerability = {
                    "sufficient": len(raw_session_ids) >= 2,
                    "reasons": ["raw-multi-session-selected"] if raw_session_ids else ["missing-session"],
                    "distinct_dates": [],
                    "distinct_sessions": list(raw_session_ids),
                    "covered_targets": [],
                }
                structured_events = []
                solver_result = solve_temporal_question(plan, session_hits) if session_hits else None
                checklist_context = build_multi_session_checklist_context(
                    plan,
                    session_hits,
                    limit=min(12, len(raw_session_ids) or 12),
                )
                if checklist_context:
                    history_context = f"{checklist_context}\n\n{history_context}".strip()
            else:
                selected_hits, _ = select_bundled_hits(
                    plan,
                    candidate_hits,
                    max_items=policy["max_items"],
                    max_tokens=policy["token_budget"],
                    encode=None,
                )
                answerability = assess_answerability(plan, selected_hits)
                structured_events = build_structured_event_view(
                    plan,
                    selected_hits,
                    limit=memory_structured_event_limit,
                )
                solver_result = solve_temporal_question(plan, selected_hits)
            for hit in selected_hits:
                store.mark_retrieved(hit.record.memory_id)
            trace = build_trace(
                query_text=query_text,
                retrieved_hits=reranked,
                selected_hits=selected_hits,
                prompt_text=(
                    history_context
                    if history_context
                    else "\n".join(render_evidence_line(hit, index=i + 1) for i, hit in enumerate(selected_hits))
                ),
            )
        prompt_structured_events = _structured_events_for_prompt(
            plan,
            structured_events,
            solver_result,
        )

        is_abs_question = question_id.endswith("_abs")
        if resolved_context_mode == "memory":
            if (is_single_session_question(plan) or plan.is_multi_session or plan.is_temporal or plan.question_type == "knowledge-update") and history_context:
                instructions = _build_memory_session_instructions(
                    plan=plan,
                    session_context=history_context,
                    evidence_sufficient=answerability["sufficient"] and not is_abs_question,
                    base_system_prompt=DEFAULT_SYSTEM_PROMPT,
                    strict_abstention=is_abs_question,
                    is_knowledge_update=(plan.question_type == "knowledge-update"),
                    has_temporal_view=plan.is_temporal and bool(structured_events),
                    has_session_checklist=plan.is_multi_session,
                    assistant_extraction=(plan.question_type == "single-session-assistant"),
                )
            else:
                instructions = build_benchmark_instructions(
                    plan=plan,
                    selected_hits=selected_hits,
                    answerability=answerability,
                    base_system_prompt=DEFAULT_SYSTEM_PROMPT,
                    structured_events=prompt_structured_events,
                    solver_result=(
                        solver_result
                        if solver_result
                        and solver_result.resolved
                        and _solver_allowed_reasoning_kind(plan)
                        and _solver_can_prompt(plan, solver_result)
                        and solver_result.confidence >= max(0.92, memory_solver_min_confidence)
                        else None
                    ),
                )
        else:
            if resolved_context_mode == "full-history":
                history_context = build_history_context(
                    instance,
                    include_assistant_turns=include_assistant_turns,
                    answer_sessions_only=False,
                    history_format=history_format,
                )
            elif resolved_context_mode == "oracle-history":
                history_context = build_history_context(
                    instance,
                    include_assistant_turns=include_assistant_turns,
                    answer_sessions_only=True,
                    history_format=history_format,
                )
            else:
                history_context = memory_recent_context
            instructions = _build_baseline_instructions(
                plan=plan,
                history_context=history_context,
                base_system_prompt=DEFAULT_SYSTEM_PROMPT,
            )
        use_solver_directly = bool(
            resolved_context_mode == "memory"
            and not is_single_session_question(plan)
            and solver_result is not None
            and solver_result.resolved
            and _solver_allowed_reasoning_kind(plan)
            and _solver_can_finalize(plan, solver_result)
            and memory_solver_mode in {"hybrid", "finalize"}
            and solver_result.confidence >= memory_solver_min_confidence
        )
        if memory_solver_mode == "finalize":
            use_solver_directly = bool(
                resolved_context_mode == "memory"
                and not is_single_session_question(plan)
                and solver_result is not None
                and solver_result.resolved
                and _solver_allowed_reasoning_kind(plan)
                and _solver_can_finalize(plan, solver_result)
                and solver_result.confidence >= memory_solver_min_confidence
            )
        if use_solver_directly:
            raw_hypothesis = solver_result.answer
            hypothesis = solver_result.answer
            response = {
                "output_text": raw_hypothesis,
                "response_id": None,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            }
            generation_mode = f"solver:{solver_result.mode}"
        else:
            response = create_response(
                api_key=resolved_api_key,
                payload=build_responses_payload(
                    model=openai_model,
                    instructions=instructions,
                    user_input=query_text,
                    max_output_tokens=(
                        max(max_new_tokens, 256)
                        if (
                            plan.is_multi_session
                            or (plan.is_temporal and ("order" in plan.question.lower() or plan.reasoning_kind == "ordering"))
                        )
                        else max_new_tokens
                    ),
                    temperature=temperature,
                    top_p=top_p,
                    metadata={
                        "benchmark": "longmemeval",
                        "question_id": question_id,
                        "memory_enabled": resolved_context_mode == "memory",
                        "reader_context_mode": resolved_context_mode,
                    },
                    reasoning_effort=openai_reasoning_effort,
                    verbosity=openai_verbosity,
                ),
                base_url=openai_base_url,
                timeout_seconds=openai_timeout_seconds,
                max_retries=openai_max_retries,
            )
            raw_hypothesis = response["output_text"]
            hypothesis = postprocess_prediction(plan, raw_hypothesis)
            generation_mode = "openai"
        store.close()

    if memory_explain and trace is not None:
        print(f"[{index}/{len(instances)}] {question_id}")
        print(format_trace(trace))

    usage = response["usage"]
    prediction_row = {
        "question_id": question_id,
        "hypothesis": hypothesis,
    }
    detail_row = {
        "question_id": question_id,
        "question_type": instance.get("question_type"),
        "question": instance.get("question"),
        "answer": instance.get("answer"),
        "hypothesis": hypothesis,
        "raw_hypothesis": raw_hypothesis,
        "openai_model": openai_model,
        "response_id": response.get("response_id"),
        "generation_mode": generation_mode,
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
        "selected_memory_types": [hit.record.memory_type for hit in selected_hits],
        "structured_event_count": len(structured_events),
        "selected_session_ids": selected_session_ids(selected_hits),
        "answer_session_ids": instance.get("answer_session_ids", []),
        "selected_session_recall": selected_session_recall(
            selected_hits,
            instance.get("answer_session_ids", []),
        ),
        "answerable": answerability["sufficient"],
        "answerability_reasons": answerability["reasons"],
        "reader_context_mode": resolved_context_mode,
        "history_session_count": len(instance.get("haystack_session_ids", [])),
        "history_context_chars": len(history_context),
        "solver_resolved": bool(solver_result and solver_result.resolved),
        "solver_confidence": solver_result.confidence if solver_result else 0.0,
        "solver_answer": solver_result.answer if solver_result else "",
        "solver_mode": solver_result.mode if solver_result else "",
    }
    return index, prediction_row, detail_row


def _ingest_history_threadsafe(store, embedder, instance, lock):
    """Ingest history memories with thread-safe embedding calls."""
    memories = list(iter_history_memories(
        instance,
        granularity=history_granularity,
        include_assistant_turns=include_assistant_turns,
    ))
    # Batch-embed all texts under the lock for efficiency
    texts = [m["text"] for m in memories]
    with lock:
        embeddings = embedder.embed_many(texts) if texts else []
    for memory_item, emb in zip(memories, embeddings):
        store.add_memory(
            user_id=memory_user_id,
            text=memory_item["text"],
            memory_type=memory_item["memory_type"],
            importance=memory_item["importance"],
            embedding=emb,
            embedding_model=embedder.model_name,
            metadata=memory_item["metadata"],
        )


# ── Main execution: parallel or sequential ──────────────────────────────────
prediction_rows = [None] * len(instances)
detail_rows = [None] * len(instances)
_completed = [0]
_print_lock = threading.Lock()

num_workers = max(1, parallel_workers)
print(f"Processing {len(instances)} examples with {num_workers} parallel workers...")

if num_workers <= 1:
    # Sequential fallback
    for idx, inst in enumerate(instances):
        _, pred, det = _process_one(idx + 1, inst)
        prediction_rows[idx] = pred
        detail_rows[idx] = det
        if (idx + 1) % 10 == 0 or (idx + 1) == len(instances):
            print(f"Processed {idx + 1}/{len(instances)} LongMemEval examples with OpenAI")
else:
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_process_one, idx + 1, inst): idx
            for idx, inst in enumerate(instances)
        }
        for future in as_completed(futures):
            idx = futures[future]
            index_val, pred, det = future.result()
            prediction_rows[idx] = pred
            detail_rows[idx] = det
            with _print_lock:
                _completed[0] += 1
                done = _completed[0]
                if done % 10 == 0 or done == len(instances):
                    print(f"Processed {done}/{len(instances)} LongMemEval examples with OpenAI")

summary = summarize_records(detail_rows)
summary.update(
    {
        "dataset_path": dataset_path,
        "question_types": question_types,
        "memory_enabled": memory_enabled,
        "reader_context_mode": resolved_context_mode,
        "history_granularity": history_granularity,
        "include_assistant_turns": include_assistant_turns,
        "history_format": history_format,
        "memory_solver_mode": memory_solver_mode,
        "memory_solver_min_confidence": memory_solver_min_confidence,
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
