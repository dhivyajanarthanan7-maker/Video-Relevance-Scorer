# app.py - Final merged & cleaned Video Relevance Scorer
# Robust transcript fetching: youtube_transcript_api, yt_dlp, OpenAI fallback
# Chunking, embeddings (SentenceTransformers), cosine similarity, precise reasoning
import os
import re
import tempfile
import traceback
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px

# Optional OpenAI client (used only for transcription fallback if available)
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    _HAS_OPENAI = True
except Exception:
    client = None
    _HAS_OPENAI = False

# Dependency flags (we'll warn the user at runtime if some libs are missing)
missing_libs = []
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
    missing_libs.append("sentence-transformers")

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
    import yt_dlp
except Exception:
    yt_dlp = None
    # yt_dlp is optional; we won't add to missing_libs list to avoid scaring user

try:
    import numpy as np  # used by embeddings and similarity
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
            + ".\nCheck the installation instructions in the README or pip install the packages."
        )


# ========= LOGGER =========
def make_logger(container):
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    def log(msg):
        st.session_state["logs"].append(msg)
        # show last 200 lines
        container.code("\n".join(st.session_state["logs"][-200:]))
    return log


# -------------------------
# Lazy model loader (delayed until Evaluate pressed)
# -------------------------
@st.cache_resource
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available.")
    return SentenceTransformer(model_name)


# -------------------------
# Video id extractor
# -------------------------
def extract_video_id(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    if "youtube" in url_or_id or "youtu.be" in url_or_id:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        return parsed.path.split("/")[-1] or None
    return url_or_id.strip()


# -------------------------
# YouTube transcript via API
# -------------------------
def get_youtube_transcript_via_api(video_id_or_url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    if YouTubeTranscriptApi is None:
        return None, None
    vid = extract_video_id(video_id_or_url)
    if not vid:
        return None, None
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
    except Exception:
        return None, None

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
            continue
    return None, None


# -------------------------
# Fallback: yt_dlp to extract .vtt
# -------------------------
def parse_vtt_to_segments(vtt_text: str) -> List[Dict]:
    vtt_text = vtt_text.strip()
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
    parts = t.split(":")
    try:
        if len(parts) == 3:
            h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
        elif len(parts) == 2:
            h = 0; m = int(parts[0]); s = float(parts[1])
        else:
            return 0.0
        return h * 3600 + m * 60 + s
    except Exception:
        return 0.0


def get_youtube_transcript_via_yt_dlp(video_id_or_url: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
    if yt_dlp is None:
        return None, None
    vid = extract_video_id(video_id_or_url)
    if not vid:
        return None, None

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-US", "en-GB"],
        "subtitlesformat": "vtt",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts["outtmpl"] = os.path.join(tmpdir, "%(id)s.%(ext)s")
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
        except Exception:
            return None, None

        vtt_path = None
        for fname in os.listdir(tmpdir):
            if fname.startswith(vid) and fname.endswith(".en.vtt"):
                vtt_path = os.path.join(tmpdir, fname)
                break
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


# -------------------------
# OpenAI transcription fallback (optional)
# -------------------------
def fetch_transcript_openai(url_or_audio, log) -> Tuple[Optional[str], Optional[List[Dict]]]:
    if not _HAS_OPENAI or client is None:
        log("OpenAI client not configured - skipping OpenAI transcription.")
        return None, None
    try:
        log("Requesting OpenAI transcription fallback...")
        # Note: behaviour may depend on your OpenAI SDK version.
        # Using the pattern the user previously had: client.audio.transcriptions.create(...)
        audio = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            input_url=url_or_audio
        )
        transcript_text = audio.text
        # Chunk into ~80-word chunks for segmenting
        words = transcript_text.split()
        chunks = [" ".join(words[i:i+80]) for i in range(0, len(words), 80)]
        segments = [{"start": i, "end": i + 1, "text": c} for i, c in enumerate(chunks)]
        log(f"OpenAI transcription created {len(segments)} segments.")
        return transcript_text, segments
    except Exception as e:
        log(f"OpenAI transcription failed: {e}")
        return None, None


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
    query_emb = embedder.encode([query_text], convert_to_numpy=True)
    seg_texts = [s["text"] for s in segments]
    seg_embs = embedder.encode(seg_texts, convert_to_numpy=True)
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
# New: Precise Reasoning about the score
# -------------------------
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


def generate_reasoning_precise(title: str, description: str, df: pd.DataFrame, sims, score: float, top_k: int = 5) -> str:
    if df is None or df.empty:
        return "No transcript data available to provide reasoning."

    n_segments = len(df)
    avg_sim = float(df["similarity"].mean()) if "similarity" in df.columns else 0.0
    std_sim = float(df["similarity"].std()) if "similarity" in df.columns else 0.0
    top_thresh = 0.6
    mid_thresh = 0.4

    df["label"] = df["similarity"].apply(
        lambda x: "Highly Relevant" if x >= top_thresh else ("Partially Relevant" if x >= mid_thresh else "Irrelevant")
    )

    promo_words = ["sponsor", "subscribe", "promo", "discount", "offer", "follow me", "link below", "affiliate"]
    df["is_promo"] = df["text"].str.lower().apply(lambda t: any(p in t for p in promo_words))

    title_desc_text = " ".join([title or "", description or ""])
    q_keywords = extract_keywords(title_desc_text, top_n=8)
    high_texts = " ".join(df[df["similarity"]>=mid_thresh]["text"].astype(str).tolist())
    keyword_counts = {k: high_texts.lower().count(k.lower()) for k in q_keywords}
    overlap_score = sum(1 for v in keyword_counts.values() if v > 0) / (len(q_keywords) or 1)

    parts = []
    verdict = "Highly relevant" if score >= 70 else ("Moderately relevant" if score >= 40 else "Low relevance")
    parts.append(f"### 🧠 Final Verdict: **{verdict} ({score}%)**")
    parts.append(f"- Segments analyzed: **{n_segments}**  \n- Avg similarity: **{avg_sim:.3f}**  \n- Std similarity: **{std_sim:.3f}**  \n- Keyword overlap: **{overlap_score*100:.1f}%**  \n- Promotional segments: **{int(df['is_promo'].sum())}**")

    earliest_high = df[df["similarity"] >= mid_thresh].sort_values("start").head(1)
    if not earliest_high.empty:
        r = earliest_high.iloc[0]
        parts.append(f"📌 First meaningful match at ~{int(r['start'])}s (sim {r['similarity']:.2f}).")
    else:
        parts.append("⚠ No early high-similarity segments detected.")

    parts.append("### 📊 Segment Classification")
    parts.append(f"- Highly Relevant: **{(df['label']=='Highly Relevant').mean()*100:.1f}%**  \n- Partially Relevant: **{(df['label']=='Partially Relevant').mean()*100:.1f}%**  \n- Irrelevant: **{(df['label']=='Irrelevant').mean()*100:.1f}%**")

    if df["is_promo"].sum() > 0:
        parts.append("### 🚨 Promotional / Call-to-action Segments")
        for _, r in df[df["is_promo"]].head(3).iterrows():
            snippet = r["text"][:150].replace("\n", " ")
            parts.append(f"- {int(r['start'])}s → “{snippet}...”")

    parts.append("### 🔍 Top matching segments (evidence)")
    for _, r in df.sort_values("similarity", ascending=False).head(top_k).iterrows():
        snippet = r["text"][:200].replace("\n", " ")
        parts.append(f"- **{int(r['start'])}s** (sim {r['similarity']:.2f}): “{snippet}...”")

    parts.append("### ⚠ Top low-sim segments (possible off-topic/filler)")
    for _, r in df.sort_values("similarity", ascending=True).head(top_k).iterrows():
        snippet = r["text"][:200].replace("\n", " ")
        parts.append(f"- {int(r['start'])}s (sim {r['similarity']:.2f}): “{snippet}...”")

    if q_keywords:
        kw_info = ", ".join([f"{k}({keyword_counts[k]})" for k in q_keywords])
        parts.append(f"### 📝 Title/Description keywords found in transcript: {kw_info}")

    parts.append("### 🏁 Summary")
    if score < 40:
        parts.append("The video appears to diverge from the declared topic; it contains many low-similarity segments or promotional content.")
    elif score < 70:
        parts.append("The video covers the topic partially; several segments are either filler or off-topic.")
    else:
        parts.append("Content strongly aligns with the topic and the title/description.")

    return "\n\n".join(parts)


# -------------------------
# Main evaluate wrapper
# -------------------------
def evaluate_video(title: str, description: str, url: Optional[str], manual_transcript: Optional[str], chunk_size_words: int, chunk_window_seconds: int, log=None):
    full_text = None
    segments = None

    # Try YouTube API first
    if url:
        log and log("Attempting YouTube Transcript API...")
        full_text, segments = get_youtube_transcript_via_api(url)

    # Fallback: yt_dlp
    if (not full_text or not segments) and url:
        log and log("YT API failed or not available — trying yt_dlp fallback...")
        full_text, segments = get_youtube_transcript_via_yt_dlp(url)

    # Fallback: OpenAI transcription
    if (not full_text or not segments) and url:
        if _HAS_OPENAI:
            log and log("Trying OpenAI transcription fallback...")
            full_text, segments = fetch_transcript_openai(url, log or (lambda x: None))
        else:
            log and log("OpenAI client not available — skipping OpenAI fallback.")

    # Manual transcript override
    if manual_transcript and manual_transcript.strip():
        log and log("Using manual transcript provided by user.")
        full_text = manual_transcript.strip()
        segments = chunk_manual_text(full_text, max_words=chunk_size_words)

    if not segments:
        return 0.0, None, pd.DataFrame(), None, "No transcript available", None

    # If segments are timestamped floats, merge into larger chunks if requested
    if isinstance(segments[0].get("start"), (int, float)):
        merged = chunk_youtube_segments(segments, max_words=chunk_size_words, max_window_seconds=chunk_window_seconds)
        if merged:
            segments = merged

    # Load embedder
    try:
        embedder = load_embedder()
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Model load failed: {e}\n\n{tb}", None

    try:
        sims = get_similarity_scores(embedder, title or "", description or "", segments)
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Embedding/similarity error: {e}\n\n{tb}", None

    score = compute_score_from_sims(sims)
    df = build_df(segments, sims)

    try:
        fig = px.bar(df, x="start", y="similarity", hover_data=["start", "end", "text"], title="Relevance Over Time")
        fig.update_layout(xaxis_title="Segment start (seconds or index)", yaxis_title="Cosine similarity")
    except Exception:
        fig = None

    reasoning = generate_reasoning_precise(title or "", description or "", df, sims, score, top_k=5)
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

log_area = st.sidebar.empty()
logger = make_logger(log_area)

with st.form(key="eval_form"):
    title = st.text_input("Video Title")
    description = st.text_input("Video Description (optional)")
    url = st.text_input("YouTube URL (optional)")
    manual_transcript = st.text_area("OR Paste Transcript Manually (Mandatory)", height=160)
    submitted = st.form_submit_button("Evaluate")

# Keep df local to the evaluation branch — avoids NameError
if submitted:
    if not title:
        st.error("Please enter a title")
        st.stop()
    if not url and not manual_transcript.strip():
        st.error("Please enter a YouTube URL or paste a transcript")
        st.stop()

    st.session_state["logs"] = []
    logger("Starting evaluation...")

    with st.spinner("Attempting to fetch transcript and compute relevance..."):
        score, fig, df, transcript, error_msg, reasoning = evaluate_video(
            title=title,
            description=description,
            url=url.strip() if url else None,
            manual_transcript=manual_transcript.strip(),
            chunk_size_words=int(chunk_size_words),
            chunk_window_seconds=int(chunk_window_seconds),
            log=logger
        )

    if error_msg:
        st.error(error_msg)
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
