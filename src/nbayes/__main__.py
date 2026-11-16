"""Command-line interface.

    python -m nbayes evaluate data/sms_es.csv --folds 5
    python -m nbayes predict data/sms_es.csv "Has ganado un premio, llama ya"
"""

from __future__ import annotations

import argparse
import statistics

from .data import load_csv, stratified_k_fold
from .metrics import accuracy, binary_scores, confusion_matrix
from .model import MultinomialNaiveBayes


def evaluate(path: str, folds: int, alpha: float, positive: str) -> None:
    texts, labels = load_csv(path)
    classes = sorted(set(labels))
    totals = [[0] * len(classes) for _ in classes]
    accs, f1s = [], []

    for train, test in stratified_k_fold(labels, k=folds):
        model = MultinomialNaiveBayes(alpha=alpha).fit(
            [texts[i] for i in train], [labels[i] for i in train]
        )
        y_true = [labels[i] for i in test]
        y_pred = [model.predict(texts[i]) for i in test]
        accs.append(accuracy(y_true, y_pred))
        f1s.append(binary_scores(y_true, y_pred, positive).f1)
        for r, row in enumerate(confusion_matrix(y_true, y_pred, classes)):
            for c, value in enumerate(row):
                totals[r][c] += value

    print(f"Dataset: {path} ({len(texts)} messages, classes: {', '.join(classes)})")
    print(f"{folds}-fold stratified cross-validation, alpha={alpha}\n")
    print(f"Accuracy : {statistics.mean(accs):.3f} +/- {statistics.stdev(accs):.3f}")
    print(f"F1 ({positive}): {statistics.mean(f1s):.3f} +/- {statistics.stdev(f1s):.3f}\n")

    width = max(len(c) for c in classes) + 2
    print("Confusion matrix (rows = true, columns = predicted)")
    print(" " * width + "".join(c.rjust(width) for c in classes))
    for cls, row in zip(classes, totals):
        print(cls.ljust(width) + "".join(str(v).rjust(width) for v in row))

    full = MultinomialNaiveBayes(alpha=alpha).fit(texts, labels)
    print(f"\nMost indicative words for '{positive}': {', '.join(full.top_words(positive, 8))}")


def predict(path: str, text: str, alpha: float) -> None:
    texts, labels = load_csv(path)
    model = MultinomialNaiveBayes(alpha=alpha).fit(texts, labels)
    probabilities = model.predict_proba(text)
    print(f"Prediction: {model.predict(text)}")
    for cls, p in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  P({cls}) = {p:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nbayes", description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing (default 1.0)")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="cross-validate on a labelled CSV")
    ev.add_argument("csv")
    ev.add_argument("--folds", type=int, default=5)
    ev.add_argument("--positive", default="spam", help="class used for F1")

    pr = sub.add_parser("predict", help="train on a CSV and classify one text")
    pr.add_argument("csv")
    pr.add_argument("text")

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        evaluate(args.csv, args.folds, args.alpha, args.positive)
    else:
        predict(args.csv, args.text, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
