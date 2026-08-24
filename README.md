# PRisk-India

A Streamlit app that scores Indian-firm annual reports/MD&A text or earnings-call
transcripts for firm-level political risk (**PRisk**), political sentiment
(**PSentiment**), and topic-level political risk (**PRiskT**) — following the
bigram + risk-synonym-window methodology behind Hassan, Hollander, van Lent &
Tahoun (2019, *QJE*), as maintained by NL Analytics.

See `METHODOLOGY.md` (also shown inside the app) for exactly what's replicated
faithfully versus approximated, and for citations.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Nothing
is sent to a server beyond your own machine — all scoring runs locally.

## Getting a political-bigram library

You need at least one before you can score anything. Two options, and you can
switch between them any time from the sidebar:

1. **US baseline (Hassan et al.)** — download the CSV from
   https://www.firmlevelrisk.com/download (cite their papers if you use it —
   see METHODOLOGY.md), then upload it in the "Score Documents" tab and map
   its bigram/weight columns.
2. **India library** — build one yourself in the "Build India Library" tab
   from your own political vs. non-political training text. Expect to iterate
   on this: the quality of your training corpora drives everything downstream.

## Typical workflow

1. **Score Documents** tab → load a bigram library → upload one or more
   annual-report/MD&A or transcript files → fix up the firm/ticker/period/doc
   type table → Score → download the results CSV.
2. Optionally load a sentiment dictionary and a topic map (**Manage Lexicons**
   tab) first, to also get PSentiment and PRiskT_* columns.
3. Repeat scoring runs across sessions — the results table accumulates until
   you clear it, so you can build up a panel across firms/quarters over time.

## Project layout

```
app.py                  Streamlit UI (4 tabs)
engine/
  textproc.py            tokenization, bigram extraction
  extract.py              PDF/DOCX/TXT -> raw text
  scoring.py               PRisk / PSentiment / PRiskT engine
  corpus_builder.py        India bigram-library builder (tf-idf-style weighting)
data/
  risk_synonyms.csv        starter risk/uncertainty word list (editable in-app)
  topic_keywords.csv       starter topic-tagging rules (editable in-app)
METHODOLOGY.md            formulas, what's exact vs. approximated, citations
```

## Notes

- All scoring settings (word windows, denominator, weight scaling) live in the
  sidebar and apply to every document you score in that session.
- The India library builder is a simplified, documented approximation of
  HHLT's tf-idf weighting — treat scores from it as a research prototype, not
  a publication-ready replication, until you've validated it (see
  METHODOLOGY.md's face-validity suggestion).
- Nothing here talks to the internet at runtime — you supply all text and
  lexicons locally, so it's safe to use with unpublished/embargoed transcripts.
