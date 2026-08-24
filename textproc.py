"""
Text preprocessing utilities.

Bigrams are defined the way Hassan, Hollander, van Lent & Tahoun (2019, QJE)
define them: adjacent two-word combinations after stripping punctuation.
Tokens keep only alphabetic words (numbers and symbols are dropped, matching
how most replications of this measure treat conference-call transcripts).
"""

import re
from dataclasses import dataclass
from typing import List, Tuple

_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation/numbers, return a flat list of word tokens."""
    if not text:
        return []
    return [w.lower() for w in _WORD_RE.findall(text)]


def bigrams(tokens: List[str]) -> List[Tuple[int, str]]:
    """
    Return (start_index, bigram_string) for every adjacent word pair.
    start_index is the position of the bigram's first token in `tokens`,
    so callers can look up a +/- word window around it.
    """
    return [(i, f"{tokens[i]} {tokens[i + 1]}") for i in range(len(tokens) - 1)]


@dataclass
class DocStats:
    n_words: int
    n_bigrams: int


def doc_stats(tokens: List[str]) -> DocStats:
    return DocStats(n_words=len(tokens), n_bigrams=max(len(tokens) - 1, 0))


def normalize_bigram_string(raw: str) -> str:
    """
    Normalize a bigram however it arrives in a library file (extra spaces,
    underscores, mixed case, punctuation) to the same "word word" format
    produced by bigrams() on document text, so lookups match.
    Returns "" if it doesn't reduce to exactly two word tokens.
    """
    if not isinstance(raw, str):
        return ""
    toks = tokenize(raw)
    if len(toks) != 2:
        return ""
    return f"{toks[0]} {toks[1]}"
