"""
PRisk-India: firm-level political risk scoring for Indian firms,
following the bigram + risk-synonym-window methodology of
Hassan, Hollander, van Lent & Tahoun (2019, QJE), as maintained/updated
by NL Analytics (apps.nlanalytics.tech).

Run with:  streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.corpus_builder import build_library
from engine.extract import extract_text
from engine.scoring import DEFAULT_WEIGHT_DIVISOR, score_document
from engine.textproc import normalize_bigram_string, tokenize

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="PRisk-India", layout="wide")


# --------------------------------------------------------------------------
# Session-state bootstrapping
# --------------------------------------------------------------------------

def _init_state():
    if "bigram_lib" not in st.session_state:
        st.session_state.bigram_lib = {}
        st.session_state.bigram_lib_label = "None loaded yet"

    if "risk_synonyms_df" not in st.session_state:
        st.session_state.risk_synonyms_df = pd.read_csv(DATA_DIR / "risk_synonyms.csv")

    if "topic_df" not in st.session_state:
        st.session_state.topic_df = pd.read_csv(DATA_DIR / "topic_keywords.csv")

    if "pos_words" not in st.session_state:
        st.session_state.pos_words = set()
        st.session_state.neg_words = set()
        st.session_state.sentiment_label = "None loaded (PSentiment will be skipped)"

    if "results" not in st.session_state:
        st.session_state.results = pd.DataFrame()

    if "built_library_preview" not in st.session_state:
        st.session_state.built_library_preview = pd.DataFrame()


_init_state()


def active_risk_synonyms() -> set:
    return set(st.session_state.risk_synonyms_df["word"].dropna().str.strip().str.lower())


def active_topic_map() -> dict:
    df = st.session_state.topic_df.dropna()
    topic_map = {}
    for topic, group in df.groupby("topic"):
        bigs = set()
        for raw in group["bigram"]:
            nb = normalize_bigram_string(raw)
            if nb:
                bigs.add(nb)
        if bigs:
            topic_map[topic] = bigs
    return topic_map


def parse_library_csv(df: pd.DataFrame, bigram_col: str, weight_col: str) -> dict:
    lib = {}
    for _, row in df.iterrows():
        nb = normalize_bigram_string(str(row[bigram_col]))
        if not nb:
            continue
        try:
            w = float(row[weight_col])
        except (ValueError, TypeError):
            continue
        lib[nb] = w
    return lib


# --------------------------------------------------------------------------
# Sidebar: scoring settings + active-library status
# --------------------------------------------------------------------------

with st.sidebar:
    st.header("Scoring settings")
    prisk_window = st.number_input("PRisk / PSentiment window (+/- words)", min_value=1,
                                    max_value=50, value=10)
    topic_window = st.number_input("Topic PRiskT window (+/- words)", min_value=1,
                                    max_value=50, value=20)
    denom_mode = st.radio("Denominator", ["bigram count (HHLT original)", "word count"], index=0)
    denom_mode_key = "bigram" if denom_mode.startswith("bigram") else "word"
    weight_divisor = st.number_input(
        "Weight divisor", min_value=1.0, value=float(DEFAULT_WEIGHT_DIVISOR), step=1.0,
        help="HHLT's published bigram weights are pre-scaled by 100,000. "
             "Set to 1 if your library's weights are not pre-scaled."
    )

    st.divider()
    st.header("Active political-bigram library")
    st.caption(st.session_state.bigram_lib_label)
    st.metric("Bigrams loaded", len(st.session_state.bigram_lib))

    st.divider()
    st.header("Active sentiment dictionary")
    st.caption(st.session_state.sentiment_label)


st.title("PRisk-India")
st.markdown(
    "Score Indian-firm annual reports / MD&A text or earnings-call transcripts "
    "for political risk (**PRisk**), political sentiment (**PSentiment**), and "
    "topic-level political risk (**PRiskT**) using the bigram + risk-synonym-window "
    "approach behind [Hassan, Hollander, van Lent & Tahoun (2019)](https://doi.org/10.1093/qje/qjz021) "
    "and NL Analytics' maintained vintage."
)

tab_score, tab_library, tab_lexicons, tab_method = st.tabs(
    ["Score Documents", "Build India Library", "Manage Lexicons", "Methodology & References"]
)

# --------------------------------------------------------------------------
# TAB 1: Score Documents
# --------------------------------------------------------------------------

with tab_score:
    st.subheader("1. Load a political-bigram library")
    st.markdown(
        "Either library works here — pick whichever is loaded (see the sidebar). "
        "The **US baseline** is the [Hassan et al. list](https://www.firmlevelrisk.com/download) "
        "(bigram + tf·idf weight, pre-scaled by 100,000); download it there and upload the CSV below. "
        "The **India library** is one you build in the next tab, or your own CSV in the same "
        "`bigram, weight` shape."
    )

    lib_source = st.radio("Library source", ["Upload a CSV", "Use library built in 'Build India Library' tab"],
                           horizontal=True)

    if lib_source == "Upload a CSV":
        lib_label = st.text_input("Label for this library (shown in the sidebar)",
                                   value="US baseline (HHLT)")
        lib_file = st.file_uploader("Bigram library CSV", type=["csv"], key="lib_upload")
        if lib_file is not None:
            try:
                raw_df = pd.read_csv(lib_file)
                st.dataframe(raw_df.head(5), width="stretch")
                cols = list(raw_df.columns)
                c1, c2 = st.columns(2)
                bigram_col = c1.selectbox("Column with the bigram text", cols, key="bg_col")
                weight_col = c2.selectbox("Column with the weight", cols,
                                           index=min(1, len(cols) - 1), key="wt_col")
                if st.button("Load this library", type="primary"):
                    lib = parse_library_csv(raw_df, bigram_col, weight_col)
                    st.session_state.bigram_lib = lib
                    st.session_state.bigram_lib_label = f"{lib_label} ({len(lib)} bigrams)"
                    st.success(f"Loaded {len(lib)} bigrams as the active library.")
            except Exception as exc:
                st.error(f"Could not read that CSV: {exc}")
    else:
        if st.session_state.built_library_preview.empty:
            st.info("No library built yet — go to the 'Build India Library' tab first.")
        else:
            st.success("Using the library most recently built in the 'Build India Library' tab.")

    st.divider()
    st.subheader("2. Upload documents")
    st.markdown(
        "Upload annual-report/MD&A extracts and/or earnings-call transcripts — mix freely, "
        "the **doc type** column below is per document. PDF, DOCX, or TXT."
    )

    uploaded_files = st.file_uploader(
        "Documents", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True, key="doc_upload"
    )

    if uploaded_files:
        meta_rows = []
        for f in uploaded_files:
            meta_rows.append({
                "filename": f.name,
                "firm": Path(f.name).stem,
                "ticker": "",
                "period": "",
                "doc_type": "Annual report / MD&A",
            })
        meta_df = pd.DataFrame(meta_rows)

        st.markdown("Fix up firm name / ticker / period / doc type before scoring:")
        edited_meta = st.data_editor(
            meta_df,
            column_config={
                "doc_type": st.column_config.SelectboxColumn(
                    options=["Annual report / MD&A", "Earnings call transcript"]
                ),
                "filename": st.column_config.TextColumn(disabled=True),
            },
            width="stretch",
            hide_index=True,
            key="meta_editor",
        )

        compute_sentiment = bool(st.session_state.pos_words or st.session_state.neg_words)
        topic_map = active_topic_map()
        compute_topics = st.checkbox(
            f"Also compute topic-level PRiskT ({len(topic_map)} topics tagged)",
            value=bool(topic_map),
        )

        if st.button("Score all documents", type="primary"):
            if not st.session_state.bigram_lib:
                st.error("Load a political-bigram library first (step 1 above).")
            else:
                file_map = {f.name: f for f in uploaded_files}
                rows = []
                progress = st.progress(0.0)
                for i, (_, meta) in enumerate(edited_meta.iterrows()):
                    f = file_map[meta["filename"]]
                    try:
                        text = extract_text(f.getvalue(), f.name)
                    except Exception as exc:
                        st.warning(f"Skipped {meta['filename']}: {exc}")
                        continue

                    result = score_document(
                        text,
                        bigram_lib=st.session_state.bigram_lib,
                        risk_synonyms=active_risk_synonyms(),
                        positive_words=st.session_state.pos_words if compute_sentiment else None,
                        negative_words=st.session_state.neg_words if compute_sentiment else None,
                        topic_bigrams=topic_map if compute_topics else None,
                        prisk_window=int(prisk_window),
                        topic_window=int(topic_window),
                        denom_mode=denom_mode_key,
                        weight_divisor=float(weight_divisor),
                    )
                    result.pop("_prisk_matches", None)
                    row = {
                        "firm": meta["firm"], "ticker": meta["ticker"], "period": meta["period"],
                        "doc_type": meta["doc_type"], "filename": meta["filename"],
                    }
                    row.update(result)
                    rows.append(row)
                    progress.progress((i + 1) / len(edited_meta))

                if rows:
                    new_df = pd.DataFrame(rows)
                    st.session_state.results = pd.concat(
                        [st.session_state.results, new_df], ignore_index=True
                    )
                    st.success(f"Scored {len(rows)} document(s).")

    st.divider()
    st.subheader("3. Results")
    if st.session_state.results.empty:
        st.caption("No results yet.")
    else:
        st.dataframe(st.session_state.results, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            csv_bytes = st.session_state.results.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_bytes, "prisk_india_results.csv",
                                "text/csv")
        with c2:
            if st.button("Clear results table"):
                st.session_state.results = pd.DataFrame()
                st.rerun()

        if "PRisk" in st.session_state.results.columns and len(st.session_state.results) > 1:
            chart_df = st.session_state.results.set_index("firm")[["PRisk"]]
            st.bar_chart(chart_df)

# --------------------------------------------------------------------------
# TAB 2: Build India Library
# --------------------------------------------------------------------------

with tab_library:
    st.subheader("Build an India-specific political-bigram library")
    st.markdown(
        "Supply a **political** training corpus (e.g. Lok Sabha/Rajya Sabha debate excerpts, "
        "PIB releases, political-desk newspaper coverage) and a **non-political** training corpus "
        "(e.g. business/finance reporting, an economics or management textbook chapter) covering "
        "similar topics but without the political framing. The builder flags bigrams that show up "
        "disproportionately in the political corpus and weights them by how specific they are to it — "
        "see the Methodology tab for the exact formula and its limits."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Political corpus**")
        pol_files = st.file_uploader("Upload file(s)", type=["pdf", "docx", "txt", "md"],
                                      accept_multiple_files=True, key="pol_files")
        pol_paste = st.text_area("...or paste text", height=150, key="pol_paste")
    with c2:
        st.markdown("**Non-political corpus**")
        nonpol_files = st.file_uploader("Upload file(s)", type=["pdf", "docx", "txt", "md"],
                                         accept_multiple_files=True, key="nonpol_files")
        nonpol_paste = st.text_area("...or paste text", height=150, key="nonpol_paste")

    c3, c4 = st.columns(2)
    min_freq = c3.number_input("Minimum frequency in political corpus", min_value=1, value=2)
    min_spec = c4.slider("Minimum specificity ratio (higher = more political-only)",
                          0.0, 1.0, 0.6, 0.05)

    if st.button("Build library", type="primary"):
        pol_texts, nonpol_texts = [], []
        for f in (pol_files or []):
            try:
                pol_texts.append(extract_text(f.getvalue(), f.name))
            except Exception as exc:
                st.warning(f"Skipped {f.name}: {exc}")
        if pol_paste.strip():
            pol_texts.append(pol_paste)

        for f in (nonpol_files or []):
            try:
                nonpol_texts.append(extract_text(f.getvalue(), f.name))
            except Exception as exc:
                st.warning(f"Skipped {f.name}: {exc}")
        if nonpol_paste.strip():
            nonpol_texts.append(nonpol_paste)

        if not pol_texts or not nonpol_texts:
            st.error("Provide at least one political and one non-political text.")
        else:
            lib_df = build_library(pol_texts, nonpol_texts,
                                    min_freq_political=int(min_freq), min_specificity=float(min_spec))
            st.session_state.built_library_preview = lib_df
            if lib_df.empty:
                st.warning("No bigrams survived the frequency/specificity filters — "
                           "try lowering the thresholds or adding more text.")
            else:
                st.success(f"Built a library of {len(lib_df)} bigrams.")

    if not st.session_state.built_library_preview.empty:
        st.dataframe(st.session_state.built_library_preview.head(100), width="stretch")
        csv_bytes = st.session_state.built_library_preview.to_csv(index=False).encode("utf-8")
        c5, c6 = st.columns(2)
        with c5:
            st.download_button("Download library CSV", csv_bytes, "india_bigram_library.csv",
                                "text/csv")
        with c6:
            if st.button("Use this as the active library for scoring"):
                lib = dict(zip(st.session_state.built_library_preview["bigram"],
                                st.session_state.built_library_preview["weight"]))
                st.session_state.bigram_lib = lib
                st.session_state.bigram_lib_label = f"India custom, built in-app ({len(lib)} bigrams)"
                st.success("Set as the active library — switch to 'Score Documents' to use it.")

# --------------------------------------------------------------------------
# TAB 3: Manage Lexicons
# --------------------------------------------------------------------------

with tab_lexicons:
    st.subheader("Risk / uncertainty synonyms")
    st.caption(
        "Words searched for in the window around a political bigram. Starter list is composed "
        "for this app, in the spirit of HHLT's Oxford-dictionary-derived list — they explicitly "
        "excluded \"question(s)\" and \"venture\" as false positives in conference-call transcripts; "
        "keep that in mind if you extend the list from a thesaurus."
    )
    edited_syn = st.data_editor(st.session_state.risk_synonyms_df, num_rows="dynamic",
                                 width="stretch", key="syn_editor")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save as active synonym list"):
            st.session_state.risk_synonyms_df = edited_syn
            st.success(f"Active list now has {len(active_risk_synonyms())} words.")
    with c2:
        st.download_button("Download as CSV", edited_syn.to_csv(index=False).encode("utf-8"),
                            "risk_synonyms.csv", "text/csv")

    st.divider()
    st.subheader("Sentiment dictionary (for PSentiment)")
    st.markdown(
        "PSentiment needs a positive/negative tone-word list. The standard choice in accounting/"
        "finance research is the "
        "[Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) "
        "— download it from Notre Dame's site and upload the CSV here. Any word list with a "
        "positive/negative flag works."
    )
    sent_file = st.file_uploader("Sentiment dictionary CSV", type=["csv"], key="sent_upload")
    if sent_file is not None:
        try:
            sdf = pd.read_csv(sent_file)
            st.dataframe(sdf.head(5), width="stretch")
            cols = list(sdf.columns)
            word_col = st.selectbox("Column with the word", cols, key="sent_word_col")
            mode = st.radio("Dictionary format", ["Separate positive/negative flag columns",
                                                    "Single polarity column (pos/neg text or +1/-1)"])
            if mode.startswith("Separate"):
                c1, c2 = st.columns(2)
                pos_col = c1.selectbox("Positive flag column (nonzero/truthy = positive)", cols,
                                        key="pos_col")
                neg_col = c2.selectbox("Negative flag column (nonzero/truthy = negative)", cols,
                                        key="neg_col")
                if st.button("Load sentiment dictionary"):
                    pos_words = set(sdf.loc[sdf[pos_col].astype(bool), word_col].str.lower().dropna())
                    neg_words = set(sdf.loc[sdf[neg_col].astype(bool), word_col].str.lower().dropna())
                    st.session_state.pos_words = pos_words
                    st.session_state.neg_words = neg_words
                    st.session_state.sentiment_label = (
                        f"{len(pos_words)} positive / {len(neg_words)} negative words loaded"
                    )
                    st.success(st.session_state.sentiment_label)
            else:
                pol_col = st.selectbox("Polarity column", cols, key="pol_col")
                if st.button("Load sentiment dictionary"):
                    s = sdf[pol_col].astype(str).str.lower()
                    pos_mask = s.isin(["pos", "positive", "1", "+1"])
                    neg_mask = s.isin(["neg", "negative", "-1"])
                    pos_words = set(sdf.loc[pos_mask, word_col].str.lower().dropna())
                    neg_words = set(sdf.loc[neg_mask, word_col].str.lower().dropna())
                    st.session_state.pos_words = pos_words
                    st.session_state.neg_words = neg_words
                    st.session_state.sentiment_label = (
                        f"{len(pos_words)} positive / {len(neg_words)} negative words loaded"
                    )
                    st.success(st.session_state.sentiment_label)
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")

    st.divider()
    st.subheader("Topic-tagging rules (for PRiskT)")
    st.caption(
        "Which bigrams count toward each topic (Economy, Trade, Tax, Security, Institutions, "
        "Health, Environment, Technology). Starter set is illustrative for an Indian context — "
        "expand it, especially with bigrams that actually show up in your bigram library."
    )
    edited_topics = st.data_editor(st.session_state.topic_df, num_rows="dynamic",
                                    width="stretch", key="topic_editor")
    c3, c4 = st.columns(2)
    with c3:
        if st.button("Save as active topic map"):
            st.session_state.topic_df = edited_topics
            st.success(f"Active topic map now covers {len(active_topic_map())} topics.")
    with c4:
        st.download_button("Download as CSV", edited_topics.to_csv(index=False).encode("utf-8"),
                            "topic_keywords.csv", "text/csv")

# --------------------------------------------------------------------------
# TAB 4: Methodology & References
# --------------------------------------------------------------------------

with tab_method:
    st.markdown(Path(__file__).with_name("METHODOLOGY.md").read_text())
