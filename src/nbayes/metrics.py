"""Classification metrics implemented without external libraries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


def _check(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("metrics are undefined for empty inputs")


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    _check(y_true, y_pred)
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    """Rows are true labels, columns are predicted labels, in ``labels`` order."""
    _check(y_true, y_pred)
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return matrix


@dataclass(frozen=True)
class BinaryScores:
    precision: float
    recall: float
    f1: float


def binary_scores(y_true: Sequence[str], y_pred: Sequence[str], positive: str) -> BinaryScores:
    """Precision, recall and F1 for the ``positive`` class.

    A ratio whose denominator is zero (for example, precision when nothing was
    predicted positive) is defined as 0.0 rather than raising.
    """
    _check(y_true, y_pred)
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryScores(precision, recall, f1)
