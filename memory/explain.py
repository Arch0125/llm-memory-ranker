from .utils import preview


def build_trace(query_text, retrieved_hits, selected_hits, prompt_text):
    return {
        "query": query_text,
        "retrieved": [
            {
                "memory_id": hit.record.memory_id,
                "type": hit.record.memory_type,
                "score": round(hit.score, 4),
                "label": hit.critic_label,
                "confidence": round(hit.critic_confidence, 4),
                "age_days": hit.age_days,
                "reasons": hit.reasons,
                "text_preview": preview(hit.record.text),
            }
            for hit in retrieved_hits
        ],
        "selected": [
            {
                "memory_id": hit.record.memory_id,
                "type": hit.record.memory_type,
                "score": round(hit.score, 4),
                "label": hit.critic_label,
                "confidence": round(hit.critic_confidence, 4),
                "token_cost": hit.token_cost,
                "text_preview": preview(hit.record.text),
            }
            for hit in selected_hits
        ],
        "prompt_preview": preview(prompt_text, limit=180),
    }


def format_trace(trace):
    lines = [
        "Memory trace:",
        f"- query: {preview(trace['query'], 120)}",
        f"- retrieved: {len(trace['retrieved'])}",
        f"- selected: {len(trace['selected'])}",
    ]
    for item in trace["selected"]:
        lines.append(
            "- selected "
            f"{item['memory_id']} "
            f"type={item['type']} "
            f"label={item['label']} "
            f"conf={item['confidence']:.2f} "
            f"score={item['score']:.2f} "
            f"tokens={item['token_cost']} "
            f"text={item['text_preview']}"
        )
    return "\n".join(lines)
