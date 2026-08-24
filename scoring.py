"""
Scoring engine implementing the PRisk / PSentiment / topic-PRiskT logic
described in Hassan, Hollander, van Lent & Tahoun (2019, QJE) and in NL
Analytics' maintained-vintage documentation:

  PRisk:
    1. Break the document into bigrams (adjacent word pairs).
    2. For every bigram that sits in the political-bigram library, look
       at the +/- WINDOW words around it (the bigram's own two tokens
       excluded).
    3. If a risk/uncertainty synonym appears in that window, add the
       bigram's library weight to a running total.
    4. Divide by the total number of bigrams (or words) in the document
       and rescale.

  PSentiment:
    Same walk over political-bigram occurrences, but instead of gating
    on a risk synonym, count net positive-minus-negative tone words in
    the window around every political-bigram occurrence.

  Topic PRiskT:
    Identical to PRisk, restricted to the subset of the political-bigram
    library tagged with a given topic, conventionally with a wider
    (+/- 20 word) window.

This module is a research-prototype approximation: the exact tf-idf
weighting formula and topic dictionaries in the original paper are not
publicly specified in enough detail to byte-reproduce here. Where a
choice had to be made, it is called out in README.md and in the
Methodology tab of the app.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .textproc import bigrams, tokenize

DEFAULT_WEIGHT_DIVISOR = 100_000.0  # HHLT bigram-weight files are pre-scaled by 1e5


@dataclass
class ScoreResult:
    score: float
    weighted_sum: float
    n_matches: int
    denominator: int
    matches: List[Tuple[str, int]] = field(default_factory=list)  # (bigram, position)


def _political_occurrences(
    tokens: List[str], bigram_lib: Dict[str, float]
) -> List[Tuple[int, str, float]]:
    """All (position, bigram, weight) triples where the bigram is in the library."""
    hits = []
    for pos, bg in bigrams(tokens):
        w = bigram_lib.get(bg)
        if w is not None:
            hits.append((pos, bg, w))
    return hits


def _window_has_any(tokens: List[str], center_pos: int, window: int, vocab: Set[str]) -> bool:
    """
    True if any token in the +/- window around the bigram at center_pos
    (tokens[center_pos], tokens[center_pos+1]) is in `vocab`. The bigram's
    own two tokens are excluded, matching the paper's convention.
    """
    lo = max(0, center_pos - window)
    hi = min(len(tokens), center_pos + 2 + window)
    for i in range(lo, hi):
        if i == center_pos or i == center_pos + 1:
            continue
        if tokens[i] in vocab:
            return True
    return False


def _window_net_sentiment(
    tokens: List[str], center_pos: int, window: int, pos_words: Set[str], neg_words: Set[str]
) -> int:
    lo = max(0, center_pos - window)
    hi = min(len(tokens), center_pos + 2 + window)
    net = 0
    for i in range(lo, hi):
        if i == center_pos or i == center_pos + 1:
            continue
        tok = tokens[i]
        if tok in pos_words:
            net += 1
        elif tok in neg_words:
            net -= 1
    return net


def compute_prisk(
    tokens: List[str],
    bigram_lib: Dict[str, float],
    risk_synonyms: Set[str],
    window: int = 10,
    denom_mode: str = "bigram",
    weight_divisor: float = DEFAULT_WEIGHT_DIVISOR,
) -> ScoreResult:
    occ = _political_occurrences(tokens, bigram_lib)
    weighted_sum = 0.0
    matches = []
    for pos, bg, w in occ:
        if _window_has_any(tokens, pos, window, risk_synonyms):
            weighted_sum += w
            matches.append((bg, pos))

    denom = max(len(tokens) - 1, 1) if denom_mode == "bigram" else max(len(tokens), 1)
    score = (weighted_sum / weight_divisor) / denom if denom else 0.0
    return ScoreResult(score=score, weighted_sum=weighted_sum, n_matches=len(matches),
                        denominator=denom, matches=matches)


def compute_psentiment(
    tokens: List[str],
    bigram_lib: Dict[str, float],
    positive_words: Set[str],
    negative_words: Set[str],
    window: int = 10,
    denom_mode: str = "bigram",
    weight_divisor: float = DEFAULT_WEIGHT_DIVISOR,
) -> ScoreResult:
    occ = _political_occurrences(tokens, bigram_lib)
    weighted_sum = 0.0
    matches = []
    for pos, bg, w in occ:
        net = _window_net_sentiment(tokens, pos, window, positive_words, negative_words)
        if net != 0:
            weighted_sum += w * net
            matches.append((bg, pos))

    denom = max(len(tokens) - 1, 1) if denom_mode == "bigram" else max(len(tokens), 1)
    score = (weighted_sum / weight_divisor) / denom if denom else 0.0
    return ScoreResult(score=score, weighted_sum=weighted_sum, n_matches=len(matches),
                        denominator=denom, matches=matches)


def compute_topic_prisk(
    tokens: List[str],
    bigram_lib: Dict[str, float],
    topic_bigrams: Dict[str, Set[str]],
    risk_synonyms: Set[str],
    window: int = 20,
    denom_mode: str = "bigram",
    weight_divisor: float = DEFAULT_WEIGHT_DIVISOR,
) -> Dict[str, ScoreResult]:
    """
    topic_bigrams: {topic_name: {bigram, bigram, ...}} -- which bigrams in
    bigram_lib belong to which topic. A bigram may belong to more than one
    topic.
    """
    results = {}
    for topic, bset in topic_bigrams.items():
        sub_lib = {bg: w for bg, w in bigram_lib.items() if bg in bset}
        results[topic] = compute_prisk(
            tokens, sub_lib, risk_synonyms, window=window,
            denom_mode=denom_mode, weight_divisor=weight_divisor,
        )
    return results


def score_document(
    text: str,
    bigram_lib: Dict[str, float],
    risk_synonyms: Set[str],
    positive_words: Optional[Set[str]] = None,
    negative_words: Optional[Set[str]] = None,
    topic_bigrams: Optional[Dict[str, Set[str]]] = None,
    prisk_window: int = 10,
    topic_window: int = 20,
    denom_mode: str = "bigram",
    weight_divisor: float = DEFAULT_WEIGHT_DIVISOR,
) -> dict:
    """One-shot convenience wrapper used by the Streamlit UI."""
    tokens = tokenize(text)
    n_words = len(tokens)
    n_bigrams = max(n_words - 1, 0)

    out = {
        "n_words": n_words,
        "n_bigrams": n_bigrams,
        "PRisk": None,
        "PSentiment": None,
        "n_prisk_matches": 0,
        "n_psentiment_matches": 0,
    }

    prisk_res = compute_prisk(tokens, bigram_lib, risk_synonyms, window=prisk_window,
                               denom_mode=denom_mode, weight_divisor=weight_divisor)
    out["PRisk"] = prisk_res.score
    out["n_prisk_matches"] = prisk_res.n_matches
    out["_prisk_matches"] = prisk_res.matches

    if positive_words is not None and negative_words is not None:
        psent_res = compute_psentiment(tokens, bigram_lib, positive_words, negative_words,
                                        window=prisk_window, denom_mode=denom_mode,
                                        weight_divisor=weight_divisor)
        out["PSentiment"] = psent_res.score
        out["n_psentiment_matches"] = psent_res.n_matches

    if topic_bigrams:
        topic_res = compute_topic_prisk(tokens, bigram_lib, topic_bigrams, risk_synonyms,
                                         window=topic_window, denom_mode=denom_mode,
                                         weight_divisor=weight_divisor)
        for topic, res in topic_res.items():
            out[f"PRiskT_{topic}"] = res.score

    return out
