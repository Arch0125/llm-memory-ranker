from dataclasses import dataclass, field
from typing import Any

from memory.cache import CachedEmbedder, InMemoryEmbeddingCache
from memory.critic import HeuristicCritic, rerank_with_critic
from memory.expansion import expand_query
from memory.explain import build_trace
from memory.recency import apply_recency_bias
from memory.rerank import null_reranker
from memory.retrieve import (
    gate_hits,
    infer_capture_type,
    parse_type_allowlist,
    retrieve_candidates,
    retrieve_for_query,
    retrieve_hybrid_candidates,
)
from prompt.budget import select_memories
from prompt.template import DEFAULT_SYSTEM_PROMPT, assemble_prompt


@dataclass
class MemoryAwareConfig:
    user_id: str = "default"
    top_k: int = 12
    max_items: int = 4
    similarity_threshold: float = 0.18
    critic_threshold: float = 0.58
    maybe_threshold: float = 0.48
    max_age_days: int | None = None
    stable_importance_threshold: float = 0.8
    memory_token_budget: int = 192
    type_allowlist: list[str] = field(default_factory=list)
    inject_system_prompt_without_memory: bool = False

    # ---- retrieval upgrades (all default-off, opt-in) ----------------------
    fusion_strategy: str = "weighted"      # 'rrf' | 'weighted'
    use_bm25: bool = False                 # use rank_bm25 BM25Okapi vs handcrafted
    use_query_expansion: bool = False      # fan out queries via memory.expansion
    keyword_weight: float = 0.35
    diversity: float = 0.0                 # 0..1, MMR strength in budget select
    use_embedding_cache: bool = True
    rerank_top_k: int = 0                  # 0 disables cross-encoder rerank
    rerank_blend: float = 0.7

    # Recency bias: post-fusion multiplicative bonus to the most-recent dated
    # memories. 0 disables. Triggered only for plans whose reasoning_kind /
    # question_type matches `recency_bias_kinds`. Generic across benchmarks.
    recency_bias: float = 0.0
    recency_bias_kinds: list[str] = field(
        default_factory=lambda: ["knowledge-update"]
    )

    # Optional benchmark-level state, not used by the core but threaded by
    # adapters wanting to pass per-query plans through to a reranker etc.
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryAwareInference:
    def __init__(
        self,
        store,
        embedder,
        critic=None,
        config=None,
        *,
        reranker=None,
        embedding_cache=None,
    ):
        self.store = store
        self.config = config or MemoryAwareConfig()
        self.critic = critic or HeuristicCritic()
        self.reranker = reranker or null_reranker()

        if self.config.use_embedding_cache and not isinstance(embedder, CachedEmbedder):
            self.embedder = CachedEmbedder(embedder, embedding_cache or InMemoryEmbeddingCache())
        else:
            self.embedder = embedder

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def rank_hits(self, query_text, hybrid=False, *, plan=None):
        type_allowlist = parse_type_allowlist(self.config.type_allowlist)

        expansions = None
        if self.config.use_query_expansion and plan is not None:
            expansions = expand_query(
                query_text,
                entities=getattr(plan, "entities", None),
                anchor_date=getattr(plan, "anchor_date", ""),
                targets=getattr(plan, "targets", None),
                focus_terms=getattr(plan, "focus_terms", None),
            )
            if expansions and expansions[0] == query_text:
                expansions = expansions[1:]

        if expansions or self.config.fusion_strategy == "rrf":
            retrieved = retrieve_for_query(
                query_text,
                self.store,
                self.embedder,
                self.config.user_id,
                top_k=self.config.top_k,
                type_allowlist=type_allowlist,
                expansions=expansions,
                fusion=self.config.fusion_strategy if self.config.fusion_strategy != "weighted" else "rrf",
                use_bm25=self.config.use_bm25,
                keyword_weight=self.config.keyword_weight,
            )
        elif hybrid:
            retrieved = retrieve_hybrid_candidates(
                query_text=query_text,
                store=self.store,
                embedder=self.embedder,
                user_id=self.config.user_id,
                top_k=self.config.top_k,
                type_allowlist=type_allowlist,
                fusion=self.config.fusion_strategy,
                use_bm25=self.config.use_bm25,
                keyword_weight=self.config.keyword_weight,
            )
        else:
            retrieved = retrieve_candidates(
                query_text=query_text,
                store=self.store,
                embedder=self.embedder,
                user_id=self.config.user_id,
                top_k=self.config.top_k,
                type_allowlist=type_allowlist,
            )

        gated = gate_hits(
            retrieved,
            sim_threshold=self.config.similarity_threshold,
            max_age_days=self.config.max_age_days,
            stable_importance_threshold=self.config.stable_importance_threshold,
        )

        if self.config.recency_bias > 0 and plan is not None:
            apply_recency_bias(
                gated,
                strength=self.config.recency_bias,
                plan=plan,
                trigger_kinds=self.config.recency_bias_kinds,
            )

        if self.config.rerank_top_k and getattr(self.reranker, "available", False):
            gated = self.reranker.rerank(
                query_text,
                gated,
                top_k=self.config.rerank_top_k,
                blend=self.config.rerank_blend,
            )

        return rerank_with_critic(query_text, gated, self.critic)

    def choose_hits(self, reranked):
        chosen = []
        low_confidence_slots = 1
        for hit in reranked:
            if hit.critic_label == "use" and hit.critic_confidence >= self.config.critic_threshold:
                chosen.append(hit)
            elif (
                hit.critic_label == "maybe"
                and hit.critic_confidence >= self.config.maybe_threshold
                and low_confidence_slots > 0
            ):
                chosen.append(hit)
                low_confidence_slots -= 1
        return chosen

    def budget_hits(
        self,
        hits,
        encode=None,
        plain_text_prompt=False,
        allowed_chars=None,
        prompt_style="chat",
        max_items=None,
        max_tokens=None,
    ):
        return select_memories(
            hits,
            max_items=max_items or self.config.max_items,
            max_tokens=max_tokens or self.config.memory_token_budget,
            encode=encode,
            plain_text=plain_text_prompt,
            allowed_chars=allowed_chars,
            style=prompt_style,
            diversity=self.config.diversity,
        )

    def prepare_prompt(
        self,
        query_text,
        recent_context="",
        system_prompt=None,
        encode=None,
        plain_text_prompt=False,
        allowed_chars=None,
        prompt_style="chat",
        plan=None,
    ):
        reranked = self.rank_hits(query_text, plan=plan)
        chosen = self.choose_hits(reranked)

        selected, _ = self.budget_hits(
            chosen,
            encode=encode,
            plain_text_prompt=plain_text_prompt,
            allowed_chars=allowed_chars,
            prompt_style=prompt_style,
        )
        if not selected and not recent_context and not self.config.inject_system_prompt_without_memory:
            from prompt.template import sanitize_for_allowed_chars

            prompt = sanitize_for_allowed_chars(query_text, allowed_chars=allowed_chars)
        else:
            prompt = assemble_prompt(
                query_text=query_text,
                memories=selected,
                recent_context=recent_context,
                system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
                plain_text=plain_text_prompt,
                allowed_chars=allowed_chars,
                style=prompt_style,
            )

        for hit in selected:
            self.store.mark_retrieved(hit.record.memory_id)

        trace = build_trace(
            query_text=query_text,
            retrieved_hits=reranked,
            selected_hits=selected,
            prompt_text=prompt,
        )
        return prompt, trace, selected

    def remember(
        self,
        text,
        memory_type="auto",
        importance=0.5,
        metadata=None,
        version_group_id=None,
    ):
        resolved_type = infer_capture_type(text) if memory_type == "auto" else memory_type
        embedding = self.embedder.embed(text)
        return self.store.add_memory(
            user_id=self.config.user_id,
            text=text,
            memory_type=resolved_type,
            importance=importance,
            embedding=embedding,
            embedding_model=self.embedder.model_name,
            metadata=metadata,
            version_group_id=version_group_id,
        )

    def forget(self, memory_id):
        self.store.delete_memory(memory_id, user_id=self.config.user_id)

    def list_memories(self, status="active", limit=100):
        return self.store.list_memories(self.config.user_id, status=status, limit=limit)
