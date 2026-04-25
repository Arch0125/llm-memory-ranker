"""BenchmarkAdapter protocol + registry.

A `BenchmarkAdapter` is everything the generic engine needs to know about a
specific dataset:

- how to load and filter instances from disk
- how to ingest an instance into the memory store as typed memories
- how to analyze a question into a `QueryPlan`
- how to postprocess a model output into the dataset's expected answer format
- how to score a (prediction, gold) pair

Adapters are registered by name and selected at runtime. New adapters subclass
`BenchmarkAdapter` and decorate themselves with `@register_adapter("name")`.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..query import QueryPlan


@dataclass
class BenchmarkInstance:
    """A single benchmark example wrapped uniformly.

    `raw` is the original dict the dataset stored — adapters keep it so they can
    pull benchmark-specific fields without bloating this dataclass.
    """

    instance_id: str
    question: str
    answer: Any
    raw: dict
    question_type: str = ""
    extras: dict = field(default_factory=dict)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol every benchmark adapter must satisfy."""

    name: str

    def load(self, path: str) -> list[BenchmarkInstance]: ...

    def filter(
        self,
        instances: Sequence[BenchmarkInstance],
        *,
        question_types: str = "",
        start_index: int = 0,
        max_examples: int = 0,
    ) -> list[BenchmarkInstance]: ...

    def ingest(self, instance: BenchmarkInstance) -> Iterable[dict]:
        """Yield memory items {text, memory_type, importance, metadata} for a single instance."""
        ...

    def analyze(self, instance: BenchmarkInstance, *, include_anchor_date: bool = True) -> QueryPlan: ...

    def postprocess(self, plan: QueryPlan, raw_text: str) -> str: ...

    def score(self, plan: QueryPlan, prediction: str, instance: BenchmarkInstance) -> dict[str, float | None]:
        """Return a dict of metric_name -> value (None for not-applicable metrics)."""
        ...


_REGISTRY: dict[str, type] = {}


def register_adapter(name: str):
    """Decorator: register an adapter class under `name`."""

    def decorator(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_adapter(name: str, **kwargs) -> BenchmarkAdapter:
    """Instantiate the adapter registered under `name`."""
    if name not in _REGISTRY:
        # Try lazy-import the bundled adapter modules so that adapters are only
        # imported when needed.
        _ensure_loaded(name)
    if name not in _REGISTRY:
        raise KeyError(f"Unknown benchmark adapter: {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _ensure_loaded(name: str) -> None:
    # Import known modules on demand to keep startup cheap and avoid hard deps
    # on benchmark-specific code at import time.
    if name == "longmemeval":
        from . import longmemeval  # noqa: F401
    elif name == "locomo":
        from . import locomo  # noqa: F401
    elif name == "memorybench":
        from . import memorybench  # noqa: F401
