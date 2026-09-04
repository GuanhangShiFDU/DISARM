"""Shared accounting helpers for model-based defenses and benchmark metrics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_CALL_SCOPE_KEY = "_llm_call_scope"
_BENCHMARK_INJECTIONS_KEY = "_benchmark_injections"


def overhead_metrics(extra_args: dict) -> dict:
    """Return the per-task overhead counters stored in the trace metrics."""

    all_metrics = extra_args.setdefault("defense_metrics", {})
    metrics = all_metrics.setdefault("overhead", {})
    metrics.setdefault("total_llm_calls", 0)
    metrics.setdefault("aux_llm_calls", 0)
    metrics.setdefault("total_embedding_calls", 0)
    metrics.setdefault("aux_embedding_calls", 0)
    metrics.setdefault("llm_calls_by_component", {})
    metrics.setdefault("embedding_calls_by_component", {})
    return metrics


def record_llm_call(
    extra_args: dict,
    *,
    component: str | None = None,
    auxiliary: bool | None = None,
) -> None:
    """Record one logical chat-completion invocation.

    A surrounding :func:`llm_call_scope` overrides the default component and
    auxiliary flag. This lets an ordinary backend wrapper account for guard or
    shadow calls without counting them twice.
    """

    scope = extra_args.get(_CALL_SCOPE_KEY, {})
    component = scope.get("component", component or "agent")
    auxiliary = scope.get("auxiliary", False if auxiliary is None else auxiliary)

    metrics = overhead_metrics(extra_args)
    metrics["total_llm_calls"] += 1
    if auxiliary:
        metrics["aux_llm_calls"] += 1
    components = metrics["llm_calls_by_component"]
    components[component] = components.get(component, 0) + 1


def record_embedding_call(extra_args: dict, *, component: str, auxiliary: bool = True) -> None:
    """Record one embedding API invocation."""

    metrics = overhead_metrics(extra_args)
    metrics["total_embedding_calls"] += 1
    if auxiliary:
        metrics["aux_embedding_calls"] += 1
    components = metrics["embedding_calls_by_component"]
    components[component] = components.get(component, 0) + 1


@contextmanager
def llm_call_scope(extra_args: dict, *, component: str, auxiliary: bool) -> Iterator[None]:
    """Temporarily label calls made through a regular backend LLM wrapper."""

    sentinel = object()
    previous: Any = extra_args.get(_CALL_SCOPE_KEY, sentinel)
    extra_args[_CALL_SCOPE_KEY] = {"component": component, "auxiliary": auxiliary}
    try:
        yield
    finally:
        if previous is sentinel:
            extra_args.pop(_CALL_SCOPE_KEY, None)
        else:
            extra_args[_CALL_SCOPE_KEY] = previous


def benchmark_injection_payloads(extra_args: dict) -> tuple[str, ...]:
    """Return exact payloads inserted into the environment for this task."""

    payloads = extra_args.get(_BENCHMARK_INJECTIONS_KEY, ())
    return tuple(payload for payload in payloads if isinstance(payload, str) and payload)


def contains_benchmark_injection(text: str, extra_args: dict) -> bool:
    """Whether a retrieved data sample contains a benchmark-inserted payload."""

    return any(payload in text for payload in benchmark_injection_payloads(extra_args))


def record_detector_outcome(metrics: dict, *, predicted_injection: bool, contains_injection: bool) -> None:
    """Update a detector's tool-output/action-level confusion matrix."""

    for key in ("detector_evaluations", "true_positives", "true_negatives", "false_positives", "false_negatives"):
        metrics.setdefault(key, 0)
    metrics["detector_evaluations"] += 1
    if predicted_injection and contains_injection:
        metrics["true_positives"] += 1
    elif predicted_injection:
        metrics["false_positives"] += 1
    elif contains_injection:
        metrics["false_negatives"] += 1
    else:
        metrics["true_negatives"] += 1


def benchmark_extra_args(injections: dict[str, str]) -> dict:
    """Construct internal task metadata used only for evaluation accounting."""

    return {_BENCHMARK_INJECTIONS_KEY: tuple(injections.values())}
