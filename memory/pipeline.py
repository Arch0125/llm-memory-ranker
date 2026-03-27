from dataclasses import dataclass, field

from memory.critic import HeuristicCritic, rerank_with_critic
from memory.explain import build_trace
from memory.retrieve import gate_hits, infer_capture_type, parse_type_allowlist, retrieve_candidates
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


class MemoryAwareInference:
    def __init__(self, store, embedder, critic=None, config=None):
        self.store = store
        self.embedder = embedder
        self.critic = critic or HeuristicCritic()
        self.config = config or MemoryAwareConfig()

    def prepare_prompt(
        self,
        query_text,
        recent_context="",
        system_prompt=None,
        encode=None,
        plain_text_prompt=False,
        allowed_chars=None,
        prompt_style="chat",
    ):
        type_allowlist = parse_type_allowlist(self.config.type_allowlist)
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
        reranked = rerank_with_critic(query_text, gated, self.critic)

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

        selected, _ = select_memories(
            chosen,
            max_items=self.config.max_items,
            max_tokens=self.config.memory_token_budget,
            encode=encode,
            plain_text=plain_text_prompt,
            allowed_chars=allowed_chars,
            style=prompt_style,
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
