import math
from pathlib import Path

import pytest

from nbayes import (
    MultinomialNaiveBayes,
    NotFittedError,
    accuracy,
    binary_scores,
    confusion_matrix,
    tokenize,
)
from nbayes.data import load_csv, stratified_k_fold

DATA = Path(__file__).resolve().parents[1] / "data" / "sms_es.csv"


# --- tokenizer -------------------------------------------------------------

def test_tokenize_lowercases_strips_accents_and_keeps_enye():
    assert tokenize("Él CANTÓ en España") == ["canto", "españa"]


def test_tokenize_collapses_numbers_and_drops_punctuation():
    assert tokenize("¡Gana 500€ ya!") == ["gana", "<num>"]


# --- model -----------------------------------------------------------------

@pytest.fixture
def toy_model():
    texts = ["premio gratis", "gratis dinero", "clase mañana", "apuntes clase"]
    labels = ["spam", "spam", "ham", "ham"]
    return MultinomialNaiveBayes().fit(texts, labels)


def test_predicts_obvious_examples(toy_model):
    assert toy_model.predict("dinero gratis") == "spam"
    assert toy_model.predict("apuntes de clase") == "ham"


def test_probabilities_sum_to_one(toy_model):
    probs = toy_model.predict_proba("premio clase")
    assert math.isclose(sum(probs.values()), 1.0)


def test_unknown_words_fall_back_to_prior():
    model = MultinomialNaiveBayes().fit(["a b", "c d", "e f"], ["x", "x", "y"])
    probs = model.predict_proba("palabra desconocida")
    assert math.isclose(probs["x"], 2 / 3)


def test_laplace_smoothing_matches_hand_computation(toy_model):
    # spam has 4 tokens; vocabulary has 6 words -> P(premio|spam) = (1+1)/(4+6)
    assert math.isclose(toy_model._log_likelihood("premio", "spam"), math.log(2 / 10))


def test_long_documents_do_not_underflow(toy_model):
    probs = toy_model.predict_proba("gratis " * 5000)
    assert probs["spam"] > 0.99


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        MultinomialNaiveBayes().predict("hola")


@pytest.mark.parametrize("alpha", [0, -1])
def test_alpha_must_be_positive(alpha):
    with pytest.raises(ValueError):
        MultinomialNaiveBayes(alpha=alpha)


def test_fit_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        MultinomialNaiveBayes().fit(["a", "b"], ["x"])


# --- metrics ---------------------------------------------------------------

def test_metrics_on_known_example():
    y_true = ["spam", "spam", "ham", "ham", "ham"]
    y_pred = ["spam", "ham", "spam", "ham", "ham"]
    assert accuracy(y_true, y_pred) == 3 / 5
    assert confusion_matrix(y_true, y_pred, ["ham", "spam"]) == [[2, 1], [1, 1]]
    scores = binary_scores(y_true, y_pred, positive="spam")
    assert scores.precision == 0.5
    assert scores.recall == 0.5
    assert scores.f1 == 0.5


def test_precision_is_zero_when_nothing_predicted_positive():
    scores = binary_scores(["spam", "ham"], ["ham", "ham"], positive="spam")
    assert scores.precision == 0.0
    assert scores.f1 == 0.0


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        accuracy([], [])


# --- data ------------------------------------------------------------------

def test_k_fold_covers_every_index_exactly_once():
    labels = ["a"] * 7 + ["b"] * 5
    seen = []
    for train, test in stratified_k_fold(labels, k=3):
        assert not set(train) & set(test)
        seen.extend(test)
    assert sorted(seen) == list(range(12))


def test_k_fold_is_stratified():
    labels = ["spam"] * 10 + ["ham"] * 20
    for _, test in stratified_k_fold(labels, k=5):
        assert sum(labels[i] == "spam" for i in test) == 2


def test_end_to_end_on_bundled_dataset():
    texts, labels = load_csv(DATA)
    accs = []
    for train, test in stratified_k_fold(labels, k=5):
        model = MultinomialNaiveBayes().fit([texts[i] for i in train], [labels[i] for i in train])
        accs.append(accuracy([labels[i] for i in test], [model.predict(texts[i]) for i in test]))
    assert sum(accs) / len(accs) > 0.8
