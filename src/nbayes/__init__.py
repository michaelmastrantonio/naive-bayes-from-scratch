"""Naive Bayes text classification from scratch, with evaluation metrics."""

from .metrics import accuracy, binary_scores, confusion_matrix
from .model import MultinomialNaiveBayes, NotFittedError
from .text import tokenize

__all__ = [
    "MultinomialNaiveBayes",
    "NotFittedError",
    "accuracy",
    "binary_scores",
    "confusion_matrix",
    "tokenize",
]
__version__ = "0.1.0"
