"""Benchmark adapters plug benchmark-specific data, analysis, and scoring into
the generic memory engine.

A registry lookup at `get_adapter("name")` returns the class so scripts can
take a `--benchmark=longmemeval` flag and stay generic.
"""
from .base import BenchmarkAdapter, BenchmarkInstance, get_adapter, list_adapters, register_adapter

__all__ = [
    "BenchmarkAdapter",
    "BenchmarkInstance",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
