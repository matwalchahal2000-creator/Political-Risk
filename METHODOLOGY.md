## What this app computes

**PRisk** — for each occurrence of a political bigram in a document, check whether a
risk/uncertainty synonym appears within a +/- word window (default 10, editable in the
sidebar). If so, add the bigram's library weight to a running total. Divide by the total
number of bigrams (or words) in the document, undo the library's x100,000 pre-scaling, done.
This mirrors the counting rule in Hassan, Hollander, van Lent & Tahoun (2019, *QJE*) — political
bigrams are the ones already flagged as political in whichever library is loaded; the app does
not re-derive "political-ness" at scoring time.

**PSentiment** — same walk over political-bigram occurrences, but instead of gating on a risk
synonym, count net positive-minus-negative tone words in the window around each occurrence.
Needs a sentiment dictionary loaded (Lexicons tab).

**PRiskT (topic PRisk)** — identical to PRisk, restricted to the subset of the library tagged
with a given topic (Economy, Trade, Tax, Security, Institutions, Health, Environment,
Technology), conventionally with a wider window (default 20 words, matching NL Analytics'
published convention for the topic series).

## Where this is a faithful replication, and where it's an approximation

**Faithful to the published methodology:**
- Bigram definition (adjacent word pairs, punctuation stripped).
- The counting rule: political bigram + risk synonym within a word window, weighted sum
  divided by a document-level denominator.
- The 10-word window for PRisk/PSentiment and the ~20-word window for topic measures.
- If you load the actual Hassan et al. bigram/weight file, the weights themselves are exact —
  this app doesn't alter them.

**Documented approximations (research-prototype level, not byte-exact):**
- **The India bigram library builder** uses a specificity-ratio weight
  (`tf_political / (tf_political + tf_nonpolitical)`), which is *in the spirit of* the tf-idf
  weighting HHLT describe but is not their exact formula — the paper's precise algorithm and
  training corpora (US newspaper political sections + a US politics textbook, vs. non-political
  newspaper sections + a different textbook) are not public in enough detail to reproduce here.
  Bigrams appearing in the political corpus below the frequency/specificity thresholds you set
  are dropped.
- **The risk/uncertainty synonym list** (Lexicons tab) is composed for this app, not copied from
  the paper's underlying Oxford dictionary entries. It follows the paper's documented exclusions
  (dropping "question(s)" and "venture", which the original authors found were false positives
  in transcripts) — extend it with the same caution.
- **The topic-tagging rules** are an illustrative starter set for an Indian context (RBI/SEBI/GST/
  Lok Sabha-flavored bigrams), not the paper's original newspaper-derived topic dictionaries.
  You'll likely want to expand this against whatever bigram library you're actually using.

## Applying a US-trained library to Indian documents

The published Hassan et al. library was trained on US political text, so bigrams like "the
constitution" or "president obama" will rarely fire on Indian earnings calls or MD&A sections —
expect PRisk to run low and noisy if you use it as-is on Indian text. It's still a reasonable
baseline for bigrams that travel (e.g. "interest rates," "trade war," "monetary policy"), and
useful as a sanity check while you build out the India-specific library (Build India Library tab).
Before relying on scores from either library in a paper, do what HHLT do in the original paper:
spot-check a handful of high- and low-scoring firm-quarters by hand and confirm the score tracks
what's actually being discussed (face validity), and compare your distribution's mean/median
against a published benchmark (HHLT's own US sample averages roughly 0.12 for PRisk, right-skewed)
as a rough sanity check on scale — not as a target Indian firms should match.

## Citations

If you use the published US bigram library or benchmark against HHLT's published series, cite:

Hassan, T. A., Hollander, S., van Lent, L., & Tahoun, A. (2019). Firm-Level Political Risk:
Measurement and Effects. *The Quarterly Journal of Economics*, 134(4), 2135-2202.
https://doi.org/10.1093/qje/qjz021

If you reference the maintained/updated vintage or its topic series:

NL Analytics. (2026). *Political Risk and Sentiment* [Data set].
https://apps.nlanalytics.tech/curated-measures/political-risk/

If you use the Loughran-McDonald sentiment dictionary:

Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis,
Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
