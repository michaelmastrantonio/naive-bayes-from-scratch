"""Dataset loading and splitting helpers."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path


def load_csv(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a CSV with ``text`` and ``label`` columns."""
    texts, labels = [], []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def stratified_k_fold(
    labels: Sequence[str], k: int = 5, seed: int = 0
) -> Iterator[tuple[list[int], list[int]]]:
    """Yield ``(train_idx, test_idx)`` pairs that keep the class balance per fold.

    Indices of each class are shuffled with a fixed seed and dealt round-robin
    into ``k`` folds, so every fold has roughly the same proportion of each
    class as the full dataset.
    """
    if k < 2:
        raise ValueError("k must be at least 2")
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    position = 0
    for label in sorted(by_class):
        indices = by_class[label]
        rng.shuffle(indices)
        for i in indices:
            folds[position % k].append(i)
            position += 1

    for f in range(k):
        test = sorted(folds[f])
        train = sorted(i for g in range(k) if g != f for i in folds[g])
        yield train, test
