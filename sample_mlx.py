"""
Sample from an MLX model on Apple Silicon, optionally with the local memory layer.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from memory import MemoryAwareConfig, MemoryAwareInference, SQLiteMemoryStore, build_embedder
from memory.explain import format_trace
from prompt.template import DEFAULT_SYSTEM_PROMPT, render_memory_line

# -----------------------------------------------------------------------------
model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
revision = None
trust_remote_code = False
system_prompt = None
start = "Hello"
num_samples = 1
max_new_tokens = 256
temperature = 0.7
top_p = 0.95
top_k = 40
seed = None
verbose = False
memory_enabled = False
memory_db_path = "memory.sqlite"
memory_user_id = "default"
memory_embedder = "hash-384"
memory_top_k = 12
memory_max_items = 4
memory_similarity_threshold = 0.18
memory_critic_threshold = 0.58
memory_maybe_threshold = 0.48
memory_max_age_days = -1
memory_token_budget = 192
memory_type_allowlist = ""
memory_recent_context = ""
memory_capture_input = False
memory_capture_type = "auto"
memory_capture_importance = 0.5
memory_version_group_id = ""
memory_explain = False
ignore_chat_template = False
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


def _build_prompt(tokenizer, query_text, selected_hits, recent_context="", base_system_prompt=None):
    base_system_prompt = base_system_prompt or DEFAULT_SYSTEM_PROMPT
    memory_block = _build_memory_block(selected_hits)

    if not ignore_chat_template and getattr(tokenizer, "has_chat_template", False):
        system_parts = [base_system_prompt]
        if memory_block:
            system_parts.append(memory_block)
        if recent_context:
            system_parts.append(f"Recent context:\n{recent_context.strip()}")
        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            {"role": "user", "content": query_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    fallback_parts = [base_system_prompt]
    if memory_block:
        fallback_parts.append(memory_block)
    if recent_context:
        fallback_parts.append(f"Recent context:\n{recent_context.strip()}")
    fallback_parts.append(f"User request:\n{query_text}\n\nAssistant response:\n")
    return "\n\n".join(fallback_parts)


tokenizer_config = {"trust_remote_code": trust_remote_code}
model, tokenizer = load(
    model_path,
    tokenizer_config=tokenizer_config,
    revision=revision,
)

memory_system = None
memory_store = None
trace = None
selected_hits = []
if memory_enabled:
    memory_store = SQLiteMemoryStore(memory_db_path)
    embedder = build_embedder(memory_embedder)
    memory_system = MemoryAwareInference(
        store=memory_store,
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
    _, trace, selected_hits = memory_system.prepare_prompt(
        query_text=start,
        recent_context=memory_recent_context,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        encode=tokenizer.encode,
        prompt_style="completion",
    )

prompt = _build_prompt(
    tokenizer=tokenizer,
    query_text=start,
    selected_hits=selected_hits,
    recent_context=memory_recent_context,
    base_system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
)

if memory_explain and trace is not None:
    print(format_trace(trace))

sampler = make_sampler(
    temp=temperature,
    top_p=top_p,
    top_k=top_k,
)
if seed is not None:
    mx.random.seed(seed)

for _ in range(num_samples):
    text = generate(
        model,
        tokenizer,
        prompt,
        verbose=verbose,
        max_tokens=max_new_tokens,
        sampler=sampler,
    )
    print(text)
    print("---------------")

if memory_enabled and memory_capture_input:
    memory_system.remember(
        text=start,
        memory_type=memory_capture_type,
        importance=memory_capture_importance,
        version_group_id=memory_version_group_id or None,
    )
if memory_store is not None:
    memory_store.close()
