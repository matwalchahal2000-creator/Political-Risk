"""
Build a political-bigram library from scratch, the way HHLT (2019) build
theirs: compare a "political" training corpus against a "non-political"
one, and weight each bigram by how disproportionately it shows up in the
political side.

This is a documented approximation of the paper's approach, not a
byte-exact replica -- the paper's exact tf-idf formula and its newspaper/
textbook training corpora aren't public in enough detail to reproduce
here. Swap in your own weighting rule if you need publication-grade
fidelity; the rest of the app only needs a bigram -> weight table, in
whatever form you produce it.

Weighting rule used here:
    tf_P(b)  = freq of b in the political corpus / total bigrams in P
    tf_N(b)  = freq of b in the non-political corpus / total bigrams in N
    weight(b) = tf_P(b) / (tf_P(b) + tf_N(b))   for b appearing in P
              (a specificity ratio in [0, 1]: 1.0 = political-only bigram)

Bigrams are then optionally filtered by a minimum raw frequency in P
(to drop one-off noise) and a minimum specificity ratio (to drop bigrams
that are common in both corpora, e.g. generic business language).
"""

from collections import Counter
from typing import Dict, List

import pandas as pd

from .textproc import bigrams, tokenize


def _bigram_counts(text: str) -> Counter:
    tokens = tokenize(text)
    return Counter(bg for _, bg in bigrams(tokens))


def build_library(
    political_texts: List[str],
    nonpolitical_texts: List[str],
    min_freq_political: int = 2,
    min_specificity: float = 0.6,
) -> pd.DataFrame:
    counts_p = Counter()
    for t in political_texts:
        counts_p.update(_bigram_counts(t))

    counts_n = Counter()
    for t in nonpolitical_texts:
        counts_n.update(_bigram_counts(t))

    total_p = sum(counts_p.values()) or 1
    total_n = sum(counts_n.values()) or 1

    rows = []
    for bg, freq_p in counts_p.items():
        if freq_p < min_freq_political:
            continue
        freq_n = counts_n.get(bg, 0)
        tf_p = freq_p / total_p
        tf_n = freq_n / total_n
        specificity = tf_p / (tf_p + tf_n) if (tf_p + tf_n) > 0 else 0.0
        if specificity < min_specificity:
            continue
        rows.append({
            "bigram": bg,
            "weight": round(specificity * 100_000, 4),  # match HHLT's x100,000 convention
            "freq_political": freq_p,
            "freq_nonpolitical": freq_n,
            "specificity_ratio": round(specificity, 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df
