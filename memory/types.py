from dataclasses import dataclass, field


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    memory_type: str
    text: str
    created_at: str
    last_accessed_at: str
    times_retrieved: int = 0
    importance: float = 0.5
    decay_score: float = 0.0
    status: str = "active"
    source_turn_id: str | None = None
    version_group_id: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryHit:
    record: MemoryRecord
    score: float
    embedding_model: str
    age_days: int
    reasons: list[str] = field(default_factory=list)
    critic_label: str = "candidate"
    critic_confidence: float = 0.0
    token_cost: int = 0


@dataclass
class CriticDecision:
    label: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
