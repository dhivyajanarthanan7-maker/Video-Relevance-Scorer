"""
video_relevance_app.py

Streamlit app to score YouTube video relevance to a topic using SentenceTransformers.

- Robust transcript fetching:
    1) youtube_transcript_api
    2) fallback: yt_dlp to download automatic captions (.vtt) and parse them
- Timestamp-aware chunking for YouTube transcripts
- Manual transcript fallback
- Friendly errors for missing numpy/torch
- NEW: human-readable reasoning section explaining the score
"""

import io
import os
import re
import tempfile
import math
import traceback
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
 
# Attempt imports that may fail on misconfigured environments.
# We'll check and show helpful instructions instead of crashing.
missing_libs = []
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
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

# yt_dlp is optional (used as a fallback to download .vtt)
try:
    import yt_dlp
except Exception:
    yt_dlp = None

# numpy is required by torch and transformers — check explicitly
try:
    import numpy as np  # noqa: F401
except Exception:
    np = None
    missing_libs.append("numpy")

# -------------------------
# Helper: friendly runtime checks
# -------------------------
def show_runtime_warnings():
    if missing_libs:
        st.warning(
            "Some Python packages are missing or failed to import: "
            + ", ".join(missing_libs)
            + ".\nCheck the installation instructions below the app."
        )

# -------------------------
# Lazy model loader (delayed until Evaluate pressed)
# -------------------------
@st.cache_resource
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available.")
    # SentenceTransformer will import torch/transformers internally.
    return SentenceTransformer(model_name)

# -------------------------
# Transcript fetchers
# -------------------------
def get_youtube_transcript_via_api(video_id_or_url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """Try youtube_transcript_api list_transcripts -> find -> fetch (robust)."""
    if YouTubeTranscriptApi is None:
        return None, None

    # extract id from URL if provided
    vid = extract_video_id(video_id_or_url)
    if not vid:
        return None, None

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
    except Exception:
        return None, None

    # Try manual transcripts first then generated transcripts (English preferences)
    for finder in (
        lambda tl: tl.find_transcript(["en", "en-US", "en-GB"]),
        lambda tl: tl.find_transcript(["en"]),
        lambda tl: tl.find_generated_transcript(["en", "en-US", "en-GB"]),
        lambda tl: tl.find_generated_transcript(["en"]),
    ):
        try:
            tr = finder(transcripts)
            if tr:
                fetched = tr.fetch()
                segments = [
                    {"start": float(t.get("start", 0.0)),
                     "end": float(t.get("start", 0.0) + t.get("duration", 0.0)),
                     "text": t.get("text", "").strip()}
                    for t in fetched if t.get("text", "").strip()
                ]
                full_text = " ".join([t["text"] for t in fetched if t.get("text", "").strip()])
                if segments:
                    return full_text, segments
        except Exception:
            # Continue to next fallback
            continue

    return None, None


def get_youtube_transcript_via_yt_dlp(video_id_or_url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Fallback: use yt_dlp to download the automatic captions (vtt) into a temp dir,
    then parse the vtt file into segments (start,end,text).
    """
    if yt_dlp is None:
        return None, None

    vid = extract_video_id(video_id_or_url)
    if not vid:
        return None, None

    ydl_opts = {
        "skip_download": True,
        # write subtitles and automatic subtitles
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
        # output template to temp dir
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # set outtmpl to tmpdir so subtitle files land there
        ydl_opts["outtmpl"] = os.path.join(tmpdir, "%(id)s.%(ext)s")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # download will populate the .vtt file in tmpdir
                res = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        except Exception:
            return None, None

        # find vtt file in tmpdir for this video id
        vtt_path = None
        for fname in os.listdir(tmpdir):
            if fname.startswith(vid) and fname.endswith(".en.vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                break
            # sometimes language code is omitted or different
            if fname.startswith(vid) and fname.endswith(".vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                break

        if not vtt_path:
            return None, None

        try:
            with open(vtt_path, "r", encoding="utf-8") as fh:
                vtt = fh.read()
        except Exception:
            return None, None

        segments = parse_vtt_to_segments(vtt)
        full_text = " ".join([s["text"] for s in segments])
        return full_text, segments


def parse_vtt_to_segments(vtt_text: str) -> List[Dict]:
    """A simple VTT parser: extracts cue start/end and text blocks."""
    # Remove the WEBVTT header if present
    vtt_text = vtt_text.strip()
    # Split into blocks separated by blank lines
    blocks = re.split(r"\n\s*\n", vtt_text)
    segments = []
    time_re = re.compile(r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3}|\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})")
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        times_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                times_line = line
            elif re.match(r"^\d+$", line.strip()):
                # cue index line, skip
                continue
            else:
                text_lines.append(line.strip())
        if not times_line:
            continue
        m = time_re.search(times_line)
        if not m:
            continue
        start_s = vtt_time_to_seconds(m.group(1))
        end_s = vtt_time_to_seconds(m.group(2))
        text = " ".join(text_lines).strip()
        if text:
            segments.append({"start": float(start_s), "end": float(end_s), "text": text})
    return segments


def vtt_time_to_seconds(t: str) -> float:
    """Convert VTT/WEBVTT time (HH:MM:SS.mmm or MM:SS.mmm or MM:SS) to seconds."""
    parts = t.split(":")
    parts = [p for p in parts]
    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        elif len(parts) == 2:
            h = 0
            m = int(parts[0])
            s = float(parts[1])
        else:
            return 0.0
        return h * 3600 + m * 60 + s
    except Exception:
        return 0.0


def extract_video_id(url_or_id: str) -> Optional[str]:
    """Return a YouTube video ID from either a URL or a raw id string."""
    if not url_or_id:
        return None
    # if looks like a full URL
    if "youtube" in url_or_id or "youtu.be" in url_or_id:
        # try parse query v=
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        # fallback for youtu.be short links or /embed/
        return parsed.path.split("/")[-1] or None
    # otherwise assume it's an id already
    return url_or_id.strip()


# -------------------------
# Chunkers
# -------------------------
def chunk_manual_text(text: str, max_words: int = 50) -> List[Dict]:
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return [{"start": i, "end": i + 1, "text": c} for i, c in enumerate(chunks)]


def chunk_youtube_segments(segments: List[Dict], max_words: int = 80, max_window_seconds: int = 30) -> List[Dict]:
    """Merge YouTube timestamped segments into larger timestamp-aware chunks."""
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


# -------------------------
# Similarity and scoring
# -------------------------
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


# -------------------------
# New: Reasoning about the score
# -------------------------
_DEFAULT_STOPWORDS = {
    "the","and","is","in","to","a","an","for","of","on","with","that","this","it","are","as",
    "by","from","at","be","or","we","you","your","our","they","their","i","my","me","was","but"
}

def extract_keywords(text: str, top_n: int = 10):
    if not text:
        return []
    # simple tokenization + frequency, filter stopwords and short tokens
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, _ in items[:top_n]]

# Updated app.py with precise reasoning engine
# (Full file inserted based on user-provided version with enhanced generate_reasoning_precise)

# NOTE: Due to length constraints, I will only insert the modified reasoning function here.
# In the next step, I can regenerate the complete file with all integrations.

def generate_reasoning(title: str, description: str, df: pd.DataFrame, sims, score: float) -> str:
    """
    Highly precise reasoning algorithm:
    - Classifies segments
    - Detects promotional/off-topic parts
    - Gives keyword overlap analysis
    - Provides timestamp-based evidence
    - Generates final verdict + justification
    """

    if df is None or df.empty:
        return "No transcript available to evaluate relevance."

    # thresholds
    HIGH = 0.60
    MID = 0.40

    n_segments = len(df)
    avg_sim = df["similarity"].mean()
    std_sim = df["similarity"].std()

    # classify segments
    df["label"] = df["similarity"].apply(
        lambda x: "Highly Relevant" if x >= HIGH else
                  ("Partially Relevant" if x >= MID else "Irrelevant")
    )

    # detect promotional language
    promo_words = ["sponsor", "subscribe", "promo", "discount",
                   "offer", "follow me", "link below", "affiliate"]
    df["is_promo"] = df["text"].str.lower().apply(
        lambda t: any(p in t for p in promo_words)
    )

    # keyword overlap analysis
    title_desc = (title + " " + description).lower()
    td_keywords = extract_keywords(title_desc, top_n=10)
    transcript_text = " ".join(df["text"].tolist()).lower()
    keyword_hits = {k: transcript_text.count(k) for k in td_keywords}
    overlap_score = sum(1 for v in keyword_hits.values() if v > 0) / (len(td_keywords) or 1)

    # timeline analysis
    earliest_rel = df[df["similarity"] >= HIGH].head(1)

    # build reasoning
    parts = []

    # VERDICT
    verdict = (
        "Highly relevant" if score >= 70 else
        "Moderately relevant" if score >= 40 else
        "Low relevance"
    )
    parts.append(f"### 🧠 Final Verdict: **{verdict} ({score}%)**")

    # SUMMARY STATS
    parts.append(
        f"- Total segments analyzed: **{n_segments}**  \n"
        f"- Avg similarity: **{avg_sim:.3f}**, Std: **{std_sim:.3f}**  \n"
        f"- Keyword overlap score: **{overlap_score*100:.1f}%**  \n"
        f"- Promotional segments detected: **{df['is_promo'].sum()}**"
    )

    # EARLY MATCH
    if not earliest_rel.empty:
        row = earliest_rel.iloc[0]
        parts.append(
            f"📌 **First strong relevance appears at {int(row['start'])}s** "
            f"(similarity {row['similarity']:.2f})."
        )
    else:
        parts.append("⚠ No strongly relevant segment detected early in the video.")

    # SEGMENT DISTRIBUTION
    parts.append("### 📊 Segment Classification")
    parts.append(
        f"- Highly Relevant: **{(df['label']=='Highly Relevant').mean()*100:.1f}%**  \n"
        f"- Partially Relevant: **{(df['label']=='Partially Relevant').mean()*100:.1f}%**  \n"
        f"- Irrelevant: **{(df['label']=='Irrelevant').mean()*100:.1f}%**"
    )

    # PROMOTIONAL CONTENT
    if df["is_promo"].sum() > 0:
        parts.append("### 🚨 Promotional Content Detected")
        promo_rows = df[df["is_promo"]].head(3)
        for _, r in promo_rows.iterrows():
            snippet = r["text"][:150].replace("\n", " ")
            parts.append(f"- {int(r['start'])}s → “{snippet}...”")

    # TOP EVIDENCE (Most relevant)
    parts.append("### 🔍 Top Strong Evidence")
    top_evidence = df.sort_values("similarity", ascending=False).head(3)
    for _, r in top_evidence.iterrows():
        snippet = r["text"][:180].replace("\n", " ")
        parts.append(
            f"- **{int(r['start'])}s** (sim {r['similarity']:.2f}): “{snippet}...”"
        )

    # LOW EVIDENCE (Most irrelevant)
    parts.append("### ⚠ Least Relevant Segments")
    low_evidence = df.sort_values("similarity", ascending=True).head(3)
    for _, r in low_evidence.iterrows():
        snippet = r["text"][:180].replace("\n", " ")
        parts.append(
            f"- {int(r['start'])}s (sim {r['similarity']:.2f}): “{snippet}...”"
        )

    # KEYWORD FINDINGS
    parts.append("### 📝 Keyword Matching")
    if len(td_keywords) > 0:
        keyword_info = ", ".join([f"{k}({keyword_hits[k]})" for k in td_keywords])
        parts.append(f"Title/description keywords found in transcript: {keyword_info}")
    else:
        parts.append("No meaningful keywords were extractable from title/description.")

    # FINAL SUMMARY
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
# -------------------------
# Main evaluate wrapper
# -------------------------
def evaluate_video(title: str, description: str, url: Optional[str], manual_transcript: Optional[str], chunk_size_words: int, chunk_window_seconds: int):
    # 1) Try youtube_transcript_api
    full_text = None
    segments = None

    if url:
        full_text, segments = get_youtube_transcript_via_api(url)
        if not full_text and not segments:
            # fallback to yt_dlp
            full_text, segments = get_youtube_transcript_via_yt_dlp(url)

    if manual_transcript and manual_transcript.strip():
        # use manual transcript
        full_text = manual_transcript.strip()
        segments = chunk_manual_text(full_text, max_words=chunk_size_words)

    if not segments:
        return 0.0, None, pd.DataFrame(), None, "No transcript available", None

    # chunk timestamped segments into larger windows if needed
    # If the segments look timestamped (float start/ends), merge them
    if isinstance(segments[0].get("start"), (int, float)):
        merged = chunk_youtube_segments(segments, max_words=chunk_size_words, max_window_seconds=chunk_window_seconds)
        if merged:
            segments = merged

    # load model lazily
    try:
        embedder = load_embedder()
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Model load failed: {e}\n\n{tb}", None

    try:
        sims = get_similarity_scores(embedder, title, description, segments)
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Embedding / similarity error: {e}\n\n{tb}", None

    score = compute_score_from_sims(sims)
    df = build_df(segments, sims)

    # build plot
    try:
        fig = px.bar(df, x="start", y="similarity", hover_data=["start", "end", "text"], title="Relevance Over Time")
        fig.update_layout(xaxis_title="Segment start (seconds or index)", yaxis_title="Cosine similarity")
    except Exception:
        fig = None

    # generate reasoning string
    reasoning = generate_reasoning(title, description, df, sims, score)

    return score, fig, df, full_text, None, reasoning


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer")
st.write("Evaluate how relevant a YouTube video is to your topic using embeddings (SentenceTransformers).")

show_runtime_warnings()

# Sidebar
st.sidebar.header("Settings")
chunk_size_words = st.sidebar.number_input("Chunk size (words per segment)", min_value=10, max_value=500, value=80, step=10)
chunk_window_seconds = st.sidebar.number_input("Max window length (seconds) for timestamped chunks", min_value=5, max_value=300, value=30, step=5)
top_k = st.sidebar.number_input("Top K segments to show", min_value=1, max_value=50, value=5, step=1)
show_low_similarity = st.sidebar.checkbox("Also show low-similarity segments", value=False)

with st.form(key="eval_form"):
    title = st.text_input("Video Title")
    description = st.text_input("Video Description (optional)")
    url = st.text_input("YouTube URL (optional)")
    manual_transcript = st.text_area("OR Paste Transcript Manually (optional)", height=160)
    submitted = st.form_submit_button("Evaluate")

if submitted:
    if not title:
        st.error("Please enter a title")
        st.stop()
    if not url and not manual_transcript.strip():
        st.error("Please enter a YouTube URL or paste a transcript")
        st.stop()

    with st.spinner("Attempting to fetch transcript and compute relevance..."):
        score, fig, df, transcript, error_msg, reasoning = evaluate_video(
            title=title,
            description=description,
            url=url.strip() if url else None,
            manual_transcript=manual_transcript.strip(),
            chunk_size_words=int(chunk_size_words),
            chunk_window_seconds=int(chunk_window_seconds)
        )

    if error_msg:
        st.error(error_msg)
        # If transcript exists but model failed, show transcript for debug
        if transcript:
            with st.expander("Transcript (fetched)"):
                st.write(transcript)
        st.stop()

    if transcript is None or df.empty:
        st.error("Could not retrieve or chunk the transcript. Check the URL or paste the transcript manually.")
        st.stop()

    st.metric("Overall relevance score", f"{score} %")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No chart available.")

    # show top K segments
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

    # Reasoning section
    if reasoning:
        st.subheader("Why this score? (Reasoning)")
        st.markdown(reasoning)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download segments as CSV", csv_bytes, file_name="segments.csv", mime="text/csv")
    st.download_button("Download transcript (.txt)", transcript.encode("utf-8"), file_name="transcript.txt", mime="text/plain")

    concat_top = "\n\n".join(top_df["text"].tolist())
    st.text_area("Top segments (concatenated)", value=concat_top, height=160)
    st.success("Done ✅")

