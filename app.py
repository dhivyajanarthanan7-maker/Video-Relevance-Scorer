"""
app.py - AI Video Relevance Scorer (Streamlit)

- Uses youtube-transcript-api (primary) + manual transcript (fallback)
- No yt_dlp (removed for Streamlit Cloud compatibility)
- Real-time debug logs visible in the UI
- Lazy model loading via st.cache_resource
"""

import os
import re
import tempfile
import traceback
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px

# Attempt imports that may fail on misconfigured environments.
missing_libs = []
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    missing_libs.append("sentence-transformers (and its dependencies like torch)")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None
    missing_libs.append("scikit-learn")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
except Exception:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = NoTranscriptFound = None
    missing_libs.append("youtube-transcript-api")

try:
    import numpy as np  # noqa: F401
except Exception:
    np = None
    missing_libs.append("numpy")


# ---------- Simple in-app logger (updates live text area) ----------
def make_logger(container):
    """Return a logger function that appends messages to session_state['logs']
       and re-renders the text_area in the provided container."""
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    def log(msg: str):
        st.session_state["logs"].append(str(msg))
        # show last N lines (keeps UI responsive)
        last = "\n".join(st.session_state["logs"][-200:])
        container.code(last, language="text")
    return log


# ---------- Helper: friendly runtime checks ----------
def show_runtime_warnings():
    if missing_libs:
        st.warning(
            "Some Python packages are missing or failed to import: "
            + ", ".join(missing_libs)
            + ".\nCheck the installation / requirements on your deployment platform."
        )


# ---------- Model loader ----------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available.")
    return SentenceTransformer(model_name)


# ---------- Transcript fetcher via youtube_transcript_api ----------
def extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    if "youtube" in url_or_id or "youtu.be" in url_or_id:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        # fallback for youtu.be short links or /embed/
        return parsed.path.split("/")[-1] or None
    return url_or_id.strip()


def get_youtube_transcript_via_api(video_id_or_url: str, log=print):
    """
    Version-safe transcript fetcher:
    - If list_transcripts() exists → use multi-language robust method
    - Otherwise → use get_transcript() (older API)
    """
    if YouTubeTranscriptApi is None:
        log("youtube_transcript_api not available")
        return None, None

    vid = extract_video_id(video_id_or_url)
    if not vid:
        log("Could not extract video ID from URL")
        return None, None

    # ---------- NEW API (has list_transcripts) ----------
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(vid)

            languages = ["en", "en-US", "en-GB"]

            # 1. Try manual subtitles
            for lang in languages:
                try:
                    tr = transcripts.find_transcript([lang])
                    fetched = tr.fetch()
                    return _segments_from_api(fetched, log)
                except Exception:
                    pass

            # 2. Try auto-generated subtitles
            for lang in languages:
                try:
                    tr = transcripts.find_generated_transcript([lang])
                    fetched = tr.fetch()
                    return _segments_from_api(fetched, log)
                except Exception:
                    pass

            log("No transcripts found using list_transcripts()")
            return None, None

        except Exception as e:
            log(f"list_transcripts() failed, falling back: {e}")

    # ---------- OLD API (get_transcript only) ----------
    try:
        fetched = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        return _segments_from_api(fetched, log)
    except Exception as e:
        log(f"get_transcript() failed: {e}")
        return None, None


def _segments_from_api(fetched, log=print):
    """Convert transcript API output into (full_text, segments)."""
    if not fetched:
        return None, None

    segments = []
    for t in fetched:
        text = t.get("text", "").strip()
        if not text:
            continue
        start = float(t.get("start", 0.0))
        duration = float(t.get("duration", 0.0))
        end = start + duration

        segments.append({
            "start": start,
            "end": end,
            "text": text,
        })

    full_text = " ".join([s["text"] for s in segments])

    log(f"Fetched {len(segments)} transcript segments via API.")
    return full_text, segments
# ---------- Chunking ----------
def chunk_manual_text(text: str, max_words: int = 50) -> List[Dict]:
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return [{"start": i, "end": i + 1, "text": c} for i, c in enumerate(chunks)]


def chunk_youtube_segments(segments: List[Dict], max_words: int = 80, max_window_seconds: int = 30) -> List[Dict]:
    if not segments:
        return []
    chunks = []
    current_parts = []
    current_start = None
    current_end = None
    current_wc = 0
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        wc = len(text.split())
        if current_start is None:
            current_start = seg["start"]
            current_end = seg["end"]
            current_parts = [text]
            current_wc = wc
            continue
        tentative_duration = seg["end"] - current_start
        tentative_wc = current_wc + wc
        if tentative_wc > max_words or tentative_duration > max_window_seconds:
            chunks.append({"start": float(current_start), "end": float(current_end), "text": " ".join(current_parts)})
            current_start = seg["start"]
            current_end = seg["end"]
            current_parts = [text]
            current_wc = wc
        else:
            current_parts.append(text)
            current_end = seg["end"]
            current_wc = tentative_wc
    if current_start is not None and current_parts:
        chunks.append({"start": float(current_start), "end": float(current_end), "text": " ".join(current_parts)})
    return chunks


# ---------- Similarity & scoring ----------
def get_similarity_scores(embedder, title: str, description: str, segments: List[Dict]):
    if not segments:
        return []
    if cosine_similarity is None:
        raise RuntimeError("scikit-learn is required for cosine similarity calculation")
    query_text = title + " " + (description or "")
    query_emb = embedder.encode([query_text])
    seg_texts = [s["text"] for s in segments]
    seg_embs = embedder.encode(seg_texts)
    sims = cosine_similarity(query_emb, seg_embs)[0]
    return sims


def compute_score_from_sims(sims) -> float:
    if sims is None or len(sims) == 0:
        return 0.0
    return round(float(sims.mean() * 100.0), 2)


def build_df(segments: List[Dict], sims) -> pd.DataFrame:
    if not segments:
        return pd.DataFrame()
    df = pd.DataFrame({
        "start": [s.get("start") for s in segments],
        "end": [s.get("end") for s in segments],
        "text": [s.get("text") for s in segments],
        "similarity": list(sims) if sims is not None and len(sims) == len(segments) else [None] * len(segments)
    })
    try:
        df = df.sort_values("start").reset_index(drop=True)
    except Exception:
        pass
    return df


# ---------- Reasoning engine (kept from your version) ----------
_DEFAULT_STOPWORDS = {
    "the","and","is","in","to","a","an","for","of","on","with","that","this","it","are","as",
    "by","from","at","be","or","we","you","your","our","they","their","i","my","me","was","but"
}

def extract_keywords(text: str, top_n: int = 10):
    if not text:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in items[:top_n]]


def generate_reasoning(title: str, description: str, df: pd.DataFrame, sims, score: float) -> str:
    if df is None or df.empty:
        return "No transcript available to evaluate relevance."

    HIGH = 0.60
    MID = 0.40

    n_segments = len(df)
    avg_sim = df["similarity"].mean()
    std_sim = df["similarity"].std()

    df["label"] = df["similarity"].apply(
        lambda x: "Highly Relevant" if x >= HIGH else
                  ("Partially Relevant" if x >= MID else "Irrelevant")
    )

    promo_words = ["sponsor", "subscribe", "promo", "discount",
                   "offer", "follow me", "link below", "affiliate"]
    df["is_promo"] = df["text"].str.lower().apply(
        lambda t: any(p in t for p in promo_words)
    )

    title_desc = (title + " " + (description or "")).lower()
    td_keywords = extract_keywords(title_desc, top_n=10)
    transcript_text = " ".join(df["text"].tolist()).lower()
    keyword_hits = {k: transcript_text.count(k) for k in td_keywords}
    overlap_score = sum(1 for v in keyword_hits.values() if v > 0) / (len(td_keywords) or 1)

    earliest_rel = df[df["similarity"] >= HIGH].head(1)

    parts = []
    verdict = (
        "Highly relevant" if score >= 70 else
        "Moderately relevant" if score >= 40 else
        "Low relevance"
    )
    parts.append(f"### 🧠 Final Verdict: **{verdict} ({score}%)**")
    parts.append(
        f"- Total segments analyzed: **{n_segments}**  \n"
        f"- Avg similarity: **{avg_sim:.3f}**, Std: **{std_sim:.3f}**  \n"
        f"- Keyword overlap score: **{overlap_score*100:.1f}%**  \n"
        f"- Promotional segments detected: **{df['is_promo'].sum()}**"
    )
    if not earliest_rel.empty:
        row = earliest_rel.iloc[0]
        parts.append(
            f"📌 **First strong relevance appears at {int(row['start'])}s** "
            f"(similarity {row['similarity']:.2f})."
        )
    else:
        parts.append("⚠ No strongly relevant segment detected early in the video.")
    parts.append("### 📊 Segment Classification")
    parts.append(
        f"- Highly Relevant: **{(df['label']=='Highly Relevant').mean()*100:.1f}%**  \n"
        f"- Partially Relevant: **{(df['label']=='Partially Relevant').mean()*100:.1f}%**  \n"
        f"- Irrelevant: **{(df['label']=='Irrelevant').mean()*100:.1f}%**"
    )
    if df["is_promo"].sum() > 0:
        parts.append("### 🚨 Promotional Content Detected")
        promo_rows = df[df["is_promo"]].head(3)
        for _, r in promo_rows.iterrows():
            snippet = r["text"][:150].replace("\n", " ")
            parts.append(f"- {int(r['start'])}s → “{snippet}...”")
    parts.append("### 🔍 Top Strong Evidence")
    top_evidence = df.sort_values("similarity", ascending=False).head(3)
    for _, r in top_evidence.iterrows():
        snippet = r["text"][:180].replace("\n", " ")
        parts.append(
            f"- **{int(r['start'])}s** (sim {r['similarity']:.2f}): “{snippet}...”"
        )
    parts.append("### ⚠ Least Relevant Segments")
    low_evidence = df.sort_values("similarity", ascending=True).head(3)
    for _, r in low_evidence.iterrows():
        snippet = r["text"][:180].replace("\n", " ")
        parts.append(
            f"- {int(r['start'])}s (sim {r['similarity']:.2f}): “{snippet}...”"
        )
    parts.append("### 📝 Keyword Matching")
    if len(td_keywords) > 0:
        keyword_info = ", ".join([f"{k}({keyword_hits[k]})" for k in td_keywords])
        parts.append(f"Title/description keywords found in transcript: {keyword_info}")
    else:
        parts.append("No meaningful keywords were extractable from title/description.")
    parts.append("### 🏁 Summary")
    if score < 40:
        parts.append(
            "The video diverges significantly from the title/topic, containing mostly irrelevant "
            "or promotional content, with limited keyword alignment."
        )
    elif score < 70:
        parts.append(
            "The video covers the topic partially with some unrelated or filler segments. "
            "Relevant content exists but is not consistent across the timeline."
        )
    else:
        parts.append(
            "The content strongly aligns with the provided topic, with consistent high-similarity "
            "segments and good keyword alignment."
        )

    return "\n\n".join(parts)


# ---------- Evaluate wrapper ----------
def evaluate_video(title: str, description: str, url: Optional[str], manual_transcript: Optional[str],
                   chunk_size_words: int, chunk_window_seconds: int, log=None):
    full_text = None
    segments = None

    if url:
        if log:
            log("Attempting to fetch transcript via youtube_transcript_api...")
        full_text, segments = get_youtube_transcript_via_api(url, log)

    if manual_transcript and manual_transcript.strip():
        if log:
            log("Using manual transcript provided by user.")
        full_text = manual_transcript.strip()
        segments = chunk_manual_text(full_text, max_words=chunk_size_words)

    if not segments:
        return 0.0, None, pd.DataFrame(), None, "No transcript available", None

    if isinstance(segments[0].get("start"), (int, float)):
        merged = chunk_youtube_segments(segments, max_words=chunk_size_words, max_window_seconds=chunk_window_seconds)
        if merged:
            segments = merged
            if log:
                log(f"Chunked into {len(segments)} timestamp-aware segments.")

    try:
        if log:
            log("Loading embedder model...")
        embedder = load_embedder()
        if log:
            log("Model loaded.")
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Model load failed: {e}\n\n{tb}", None

    try:
        if log:
            log("Computing embeddings and similarity scores...")
        sims = get_similarity_scores(embedder, title, description, segments)
        if log:
            log("Similarity computation done.")
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Embedding / similarity error: {e}\n\n{tb}", None

    score = compute_score_from_sims(sims)
    df = build_df(segments, sims)

    try:
        fig = px.bar(df, x="start", y="similarity", hover_data=["start", "end", "text"], title="Relevance Over Time")
        fig.update_layout(xaxis_title="Segment start (seconds or index)", yaxis_title="Cosine similarity")
    except Exception:
        fig = None

    reasoning = generate_reasoning(title, description, df, sims, score)
    return score, fig, df, full_text, None, reasoning


# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer (Streamlit)")

show_runtime_warnings()

# Sidebar settings
st.sidebar.header("Settings")
chunk_size_words = st.sidebar.number_input("Chunk size (words per segment)", min_value=10, max_value=500, value=80, step=10)
chunk_window_seconds = st.sidebar.number_input("Max window length (seconds) for timestamped chunks", min_value=5, max_value=300, value=30, step=5)
top_k = st.sidebar.number_input("Top K segments to show", min_value=1, max_value=50, value=5, step=1)
show_low_similarity = st.sidebar.checkbox("Also show low-similarity segments", value=False)

# Logging panel UI
log_container = st.sidebar.empty()
log_container.markdown("**Realtime logs**")
logger = make_logger(log_container)

with st.form(key="eval_form"):
    title = st.text_input("Video Title")
    description = st.text_input("Video Description (optional)")
    url = st.text_input("YouTube URL (optional)")
    manual_transcript = st.text_area("OR Paste Transcript Manually (optional)", height=160)
    submitted = st.form_submit_button("Evaluate")

if submitted:
    # reset logs for this run
    st.session_state["logs"] = []
    logger("Starting evaluation...")

    if not title:
        st.error("Please enter a title")
        st.stop()
    if not url and not manual_transcript.strip():
        st.error("Please enter a YouTube URL or paste a transcript")
        st.stop()

    # run evaluation (logs will be appended)
    try:
        score, fig, df, transcript, error_msg, reasoning = evaluate_video(
            title=title,
            description=description,
            url=url.strip() if url else None,
            manual_transcript=manual_transcript.strip(),
            chunk_size_words=int(chunk_size_words),
            chunk_window_seconds=int(chunk_window_seconds),
            log=logger
        )
    except Exception as e:
        logger(f"Unhandled error: {e}")
        st.error(f"Unhandled error during evaluation: {e}")
        st.stop()

    if error_msg:
        logger(f"Error: {error_msg}")
        st.error(error_msg)
        if transcript:
            with st.expander("Transcript (fetched)"):
                st.write(transcript)
        st.stop()

    if transcript is None or df.empty:
        logger("No transcript / empty dataframe after processing.")
        st.error("Could not retrieve or chunk the transcript. Check the URL or paste the transcript manually.")
        st.stop()

    st.metric("Overall relevance score", f"{score} %")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No chart available.")

    df_sorted = df.sort_values("similarity", ascending=False).reset_index(drop=True)
    top_df = df_sorted.head(int(top_k))
    st.subheader("Top segments")
    st.dataframe(top_df[["start", "end", "similarity", "text"]], use_container_width=True)

    if show_low_similarity:
        st.subheader("Lowest similarity segments")
        low_df = df.sort_values("similarity", ascending=True).head(int(top_k))
        st.dataframe(low_df[["start", "end", "similarity", "text"]], use_container_width=True)

    with st.expander("Show full segments table"):
        st.dataframe(df, use_container_width=True)

    with st.expander("Transcript (full)"):
        st.write(transcript)

    if reasoning:
        st.subheader("Why this score? (Reasoning)")
        st.markdown(reasoning)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download segments as CSV", csv_bytes, file_name="segments.csv", mime="text/csv")
    st.download_button("Download transcript (.txt)", transcript.encode("utf-8"), file_name="transcript.txt", mime="text/plain")

    concat_top = "\n\n".join(top_df["text"].tolist())
    st.text_area("Top segments (concatenated)", value=concat_top, height=160)
    logger("Evaluation finished.")
    st.success("Done ✅")
