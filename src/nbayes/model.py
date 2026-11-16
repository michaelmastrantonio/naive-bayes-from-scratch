"""Multinomial Naive Bayes text classifier implemented from scratch.

For a document d made of tokens w_1..w_n and a class c, Naive Bayes picks

    argmax_c  log P(c) + sum_i log P(w_i | c)

assuming tokens are conditionally independent given the class. Word
likelihoods use additive (Laplace) smoothing:

    P(w | c) = (count(w, c) + alpha) / (total_words(c) + alpha * |V|)

Working in log-space avoids floating-point underflow when multiplying many
small probabilities.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from .text import tokenize


class NotFittedError(RuntimeError):
    """Raised when predicting with a model that has not been trained."""


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0, tokenizer: Callable[[str], list[str]] = tokenize):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.classes_: list[str] = []
        self._log_prior: dict[str, float] = {}
        self._word_counts: dict[str, Counter[str]] = {}
        self._total_words: dict[str, int] = {}
        self._vocabulary: set[str] = set()

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> MultinomialNaiveBayes:
        """Learn class priors and per-class word counts from labelled texts."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("cannot fit on an empty dataset")

        class_counts = Counter(labels)
        self.classes_ = sorted(class_counts)
        self._word_counts = {c: Counter() for c in self.classes_}
        self._vocabulary = set()

        for text, label in zip(texts, labels):
            tokens = self.tokenizer(text)
            self._word_counts[label].update(tokens)
            self._vocabulary.update(tokens)

        n = len(labels)
        self._log_prior = {c: math.log(class_counts[c] / n) for c in self.classes_}
        self._total_words = {c: sum(self._word_counts[c].values()) for c in self.classes_}
        return self

    def _log_likelihood(self, token: str, cls: str) -> float:
        numerator = self._word_counts[cls][token] + self.alpha
        denominator = self._total_words[cls] + self.alpha * len(self._vocabulary)
        return math.log(numerator / denominator)

    def log_scores(self, text: str) -> dict[str, float]:
        """Unnormalised log-posterior of each class for ``text``."""
        if not self.classes_:
            raise NotFittedError("call fit() before predicting")
        # Words never seen in training carry no evidence for any class.
        tokens = [t for t in self.tokenizer(text) if t in self._vocabulary]
        return {
            c: self._log_prior[c] + sum(self._log_likelihood(t, c) for t in tokens)
            for c in self.classes_
        }

    def predict_proba(self, text: str) -> dict[str, float]:
        """Posterior probability of each class, normalised with log-sum-exp."""
        scores = self.log_scores(text)
        top = max(scores.values())
        exp = {c: math.exp(s - top) for c, s in scores.items()}
        total = sum(exp.values())
        return {c: v / total for c, v in exp.items()}

    def predict(self, text: str) -> str:
        scores = self.log_scores(text)
        return max(scores, key=scores.__getitem__)

    def top_words(self, cls: str, k: int = 10) -> list[str]:
        """Words that most increase the odds of ``cls`` versus the other classes."""
        others = [c for c in self.classes_ if c != cls]

        def log_ratio(word: str) -> float:
            rest = max(self._log_likelihood(word, o) for o in others)
            return self._log_likelihood(word, cls) - rest

        return sorted(self._vocabulary, key=log_ratio, reverse=True)[:k]
