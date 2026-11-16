"""Text normalisation and tokenisation (Spanish-aware, standard library only)."""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-zñ0-9]+")

# Very common Spanish words that carry little information for classification.
STOPWORDS_ES = frozenset(
    "a al algo con de del el en es esta este la las lo los me mi no o para por que se "
    "su sus te tu un una uno y ya".split()
)


def strip_accents(text: str) -> str:
    """Remove accents (á -> a) but keep ñ, which is a different letter in Spanish."""
    text = text.replace("ñ", "\0").replace("Ñ", "\1")
    decomposed = unicodedata.normalize("NFD", text)
    without = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return without.replace("\0", "ñ").replace("\1", "Ñ")


def tokenize(text: str, stopwords: frozenset[str] = STOPWORDS_ES) -> list[str]:
    """Lower-case, strip accents and split ``text`` into word tokens.

    Numbers are collapsed into a single ``<num>`` token, because the exact
    value ("50" vs "500") rarely matters but the presence of a number does.

    >>> tokenize("¡GANA 500€ ahora!")
    ['gana', '<num>', 'ahora']
    """
    normalised = strip_accents(text.lower())
    tokens = []
    for token in _TOKEN_RE.findall(normalised):
        if token.isdigit():
            tokens.append("<num>")
        elif token not in stopwords:
            tokens.append(token)
    return tokens
