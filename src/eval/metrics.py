"""
Evaluation metrics for the neural debugger.

  pass@1        Fraction of CruxEval problems solved at the top-1 sample.
  field_accuracy  Per-field exact-match accuracy (evt, src, locals, args).
"""

from __future__ import annotations

from typing import Any


def pass_at_1(results: list[bool]) -> float:
    """Fraction of True values in `results`."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def field_accuracy(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute per-field exact-match accuracy.

    Each prediction/reference dict should have keys: evt, src, args, locals.
    Returns a dict mapping field name -> accuracy in [0, 1].
    """
    fields = ["evt", "src", "args", "locals"]
    counts: dict[str, int] = {f: 0 for f in fields}
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        for f in fields:
            if pred.get(f) == ref.get(f):
                counts[f] += 1

    if total == 0:
        return {f: 0.0 for f in fields}
    return {f: counts[f] / total for f in fields}
