from memory.selection import mmr_select

from .template import render_memory_line, sanitize_for_allowed_chars


def estimate_token_count(text, encode=None):
    if encode is not None:
        try:
            return len(encode(text))
        except Exception:
            pass
    words = max(1, len(text.split()))
    return words + max(1, len(text) // 24)


def _budget_score(hit):
    label_bonus = {"use": 2.0, "maybe": 1.0, "ignore": 0.0}.get(hit.critic_label, 0.0)
    freshness_bonus = max(0.0, 1.0 - (hit.age_days / 180.0))
    return (
        (2.5 * label_bonus)
        + (1.6 * hit.critic_confidence)
        + (1.2 * hit.score)
        + (0.8 * hit.record.importance)
        + (0.2 * freshness_bonus)
    )


def select_memories(
    hits,
    max_items,
    max_tokens,
    encode=None,
    plain_text=False,
    allowed_chars=None,
    style="chat",
    diversity=0.0,
):
    """Select up to `max_items` memory hits within `max_tokens`.

    `diversity` in (0, 1] enables MMR-based selection: items are first
    re-ordered to balance relevance against redundancy with what's already
    been chosen. `diversity=0.0` keeps the original score-based ordering.
    """
    selected = []
    used_tokens = 0

    if diversity and diversity > 0.0:
        scored = sorted(hits, key=_budget_score, reverse=True)
        for index, hit in enumerate(scored):
            hit.score = max(hit.score, _budget_score(hit) / 10.0)
        ranked = mmr_select(scored, k=max(max_items * 2, max_items), lambda_=1.0 - min(diversity, 0.95))
    else:
        ranked = sorted(hits, key=_budget_score, reverse=True)

    for hit in ranked:
        render_plain = plain_text or style == "completion"
        rendered = sanitize_for_allowed_chars(
            render_memory_line(hit, plain_text=render_plain),
            allowed_chars=allowed_chars,
        )
        hit.token_cost = estimate_token_count(rendered, encode=encode)
        if len(selected) >= max_items:
            break
        if hit.token_cost > max_tokens:
            continue
        if used_tokens + hit.token_cost > max_tokens:
            continue
        selected.append(hit)
        used_tokens += hit.token_cost
    return selected, used_tokens
