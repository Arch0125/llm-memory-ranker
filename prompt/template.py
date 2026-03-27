import re


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use retrieved memories only if they are directly relevant. "
    "If a memory is marked LOW CONFIDENCE or OUTDATED, do not mention it unless the user asks."
)


def _priority_from_importance(importance):
    if importance >= 0.85:
        return "high"
    if importance >= 0.6:
        return "medium"
    return "low"


def _confidence_bucket(hit):
    if hit.critic_label == "use" and hit.critic_confidence >= 0.75:
        return "HIGH"
    if hit.critic_label == "use" or hit.critic_confidence >= 0.6:
        return "MEDIUM"
    return "LOW"


def _staleness_bucket(hit):
    if hit.age_days > 45:
        return "OUTDATED"
    return "STABLE"


def sanitize_for_allowed_chars(text, allowed_chars=None):
    if not allowed_chars:
        return text
    sanitized_lines = []
    for raw_line in text.splitlines():
        sanitized = "".join(char if char in allowed_chars else " " for char in raw_line)
        sanitized = re.sub(r" +", " ", sanitized).strip()
        sanitized_lines.append(sanitized)
    return "\n".join(sanitized_lines)


def render_memory_line(hit, plain_text=False):
    if plain_text:
        return (
            f"type: {hit.record.memory_type}. "
            f"priority: {_priority_from_importance(hit.record.importance)}. "
            f"confidence: {_confidence_bucket(hit).lower()}. "
            f"status: {_staleness_bucket(hit).lower()}. "
            f"memory: {hit.record.text}"
        )
    return (
        f"[TYPE={hit.record.memory_type}]"
        f"[PRIORITY={_priority_from_importance(hit.record.importance)}]"
        f"[CONFIDENCE={_confidence_bucket(hit)}]"
        f"[STATUS={_staleness_bucket(hit)}]"
        f"[SIM={hit.score:.2f}] "
        f"{hit.record.text}"
    )


def assemble_prompt(
    query_text,
    memories,
    recent_context="",
    system_prompt=None,
    plain_text=False,
    allowed_chars=None,
    style="chat",
):
    if not memories and not recent_context and not system_prompt:
        return sanitize_for_allowed_chars(query_text, allowed_chars=allowed_chars)

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    render_plain = plain_text or style == "completion"
    high_confidence = []
    low_confidence = []
    for hit in memories:
        line = sanitize_for_allowed_chars(
            render_memory_line(hit, plain_text=render_plain),
            allowed_chars=allowed_chars,
        )
        if _confidence_bucket(hit) == "LOW":
            low_confidence.append(f"- {line}")
        else:
            high_confidence.append(f"- {line}")

    if style == "completion":
        sections = [f"Instructions:\n{sanitize_for_allowed_chars(system_prompt, allowed_chars=allowed_chars)}"]
        if high_confidence:
            sections.append("Relevant memory:\n" + "\n".join(high_confidence))
        if low_confidence:
            sections.append(
                "Possible but lower confidence memory:\n"
                + "\n".join(low_confidence)
            )
        if recent_context:
            sections.append(
                "Recent context:\n"
                + sanitize_for_allowed_chars(recent_context.strip(), allowed_chars=allowed_chars)
            )
        sections.append(
            "User request:\n"
            + sanitize_for_allowed_chars(query_text.strip(), allowed_chars=allowed_chars)
            + "\n\nAssistant response:\n"
        )
        return sanitize_for_allowed_chars("\n\n".join(sections), allowed_chars=allowed_chars)

    sections = [f"SYSTEM:\n{system_prompt}"]
    if high_confidence:
        if plain_text:
            sections.append("MEMORY HIGH CONFIDENCE:\n" + "\n".join(high_confidence))
        else:
            sections.append("MEMORY (HIGH CONFIDENCE / STABLE):\n" + "\n".join(high_confidence))
    if low_confidence:
        if plain_text:
            sections.append("MEMORY LOW CONFIDENCE:\n" + "\n".join(low_confidence))
        else:
            sections.append(
                "MEMORY (LOW CONFIDENCE / MAY BE IRRELEVANT):\n"
                + "\n".join(low_confidence)
            )
    if recent_context:
        sections.append(f"RECENT CONTEXT:\n{recent_context.strip()}")
    sections.append(f"USER:\n{query_text.strip()}\n\nASSISTANT:\n")
    return sanitize_for_allowed_chars("\n\n".join(sections), allowed_chars=allowed_chars)
