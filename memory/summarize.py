from collections import defaultdict


def summarize_memories(mem_items, per_type_limit=3):
    grouped = defaultdict(list)
    for item in mem_items:
        grouped[item.memory_type].append(item)

    lines = []
    for memory_type in sorted(grouped):
        ranked = sorted(
            grouped[memory_type],
            key=lambda item: (item.importance, item.times_retrieved, item.last_accessed_at),
            reverse=True,
        )
        snippets = "; ".join(item.text for item in ranked[:per_type_limit])
        lines.append(f"{memory_type}: {snippets}")
    return "\n".join(lines)
