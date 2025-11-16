"""
AI Video Relevance Scorer - FINAL VERSION
Includes:
✔ youtube-transcript-api
✔ yt-dlp fallback audio download
✔ OpenAI GPT-4o-mini-transcribe fallback
✔ Full logging in Streamlit
✔ Your scoring + reasoning engine
"""

import os
import re
import tempfile
import traceback
from typing import List, Dict, Optional, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from openai import OpenAI

# ---------------------------------------------------
# IMPORT LIBRARIES WITH SAFE FALLBACKS
# ---------------------------------------------------
missing_libs = []

try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None
    missing_libs.append("sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except:
    cosine_similarity = None
    missing_libs.append("scikit-learn")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except:
    YouTubeTranscriptApi = None
    missing_libs.append("youtube-transcript-api")

try:
    import yt_dlp
except:
    yt_dlp = None
    missing_libs.append("yt-dlp (required for audio fallback)")

try:
    import numpy as np
except:
    np = None
    missing_libs.append("numpy")


# ---------------------------------------------------
# REALTIME LOGGER
# ---------------------------------------------------
def make_logger(container):
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    def log(msg: str):
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        full = f"{timestamp} {msg}"
        print(full)
        st.session_state["logs"].append(full)
        container.code("\n".join(st.session_state["logs"][-300:]), language="text")
    return log


# ---------------------------------------------------
# WARN ABOUT MISSING LIBS
# ---------------------------------------------------
def show_runtime_warnings():
    if missing_libs:
        st.warning(
            "Missing packages: " + ", ".join(missing_libs) +
            "\nPlease ensure they are included in requirements.txt"
        )


# ---------------------------------------------------
# LOAD SENTENCE TRANSFORMER
# ---------------------------------------------------
@st.cache_resource
def load_embedder(model_name="all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    return SentenceTransformer(model_name)


# ---------------------------------------------------
# HELPER: Extract Video ID
# ---------------------------------------------------
def extract_video_id(url_or_id: str) -> Optional[str]:
    if "youtube" in url_or_id or "youtu.be" in url_or_id:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url_or_id)
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        return parsed.path.split("/")[-1]
    return url_or_id.strip()


# ---------------------------------------------------
# 1️⃣ TRY TO GET TRANSCRIPT USING youtube-transcript-api
# ---------------------------------------------------
def get_transcript_youtube_api(url, log):
    if YouTubeTranscriptApi is None:
        log("youtube-transcript-api is missing.")
        return None, None

    vid = extract_video_id(url)
    if not vid:
        log("Could not extract video ID.")
        return None, None

    try:
        log("Trying list_transcripts() ...")
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
    except Exception as e:
        log(f"list_transcripts() raised: {e}")
        transcripts = None

    if not transcripts:
        return None, None

    # Try manual + auto transcripts
    for method in [
        lambda t: t.find_manually_created_transcript(t._manually_created_transcripts),
        lambda t: t.find_generated_transcript(t._generated_transcripts),
    ]:
        try:
            tr = method(transcripts)
            if tr:
                log(f"Found transcript: {tr.language_code}")
                fetched = tr.fetch()
                text = " ".join([x["text"] for x in fetched if x["text"].strip()])
                segments = [
                    {"start": x["start"], "end": x["start"] + x["duration"], "text": x["text"]}
                    for x in fetched
                ]
                return text, segments
        except:
            continue

    log("Transcript API: No transcript available.")
    return None, None


# ---------------------------------------------------
# 2️⃣ FALLBACK: DOWNLOAD AUDIO USING yt-dlp
# ---------------------------------------------------
def download_audio(url, log):
    if yt_dlp is None:
        log("yt-dlp is not installed — cannot download audio.")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "audio.m4a")
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": out_path,
                "quiet": True,
                "no_warnings": True,
            }
            log("Downloading audio via yt-dlp...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            log(f"Audio downloaded: {out_path}")
            return out_path
    except Exception as e:
        log(f"yt-dlp audio download failed: {e}")
        return None


# ---------------------------------------------------
# 3️⃣ FALLBACK: OPENAI TRANSCRIPTION (GPT-4o-mini-transcribe)
# ---------------------------------------------------
def openai_transcribe(audio_path, log):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            log("❌ OPENAI_API_KEY not set in Streamlit secrets.")
            return None, None

        client = OpenAI(api_key=api_key)

        log("Sending audio to OpenAI for transcription...")
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                file=f,
                model="gpt-4o-mini-transcribe"
            )

        text = result.text
        segments = [{"start": i, "end": i + 1, "text": t}
                    for i, t in enumerate(text.split(". "))]

        log("OpenAI transcription completed.")
        return text, segments

    except Exception as e:
        log(f"OpenAI transcription error: {e}")
        return None, None


# ---------------------------------------------------
# CHUNKERS
# ---------------------------------------------------
def chunk_manual_text(text, size=80):
    words = text.split()
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    return [{"start": i, "end": i + 1, "text": c} for i, c in enumerate(chunks)]


# ---------------------------------------------------
# SIMILARITY + SCORING
# ---------------------------------------------------
def get_similarity_scores(embedder, title, description, segments):
    query = f"{title} {description}"
    q_emb = embedder.encode([query])
    seg_embs = embedder.encode([s["text"] for s in segments])
    return cosine_similarity(q_emb, seg_embs)[0]


def compute_score_from_sims(s):
    return round(float(s.mean() * 100.0), 2)


def build_df(segments, sims):
    df = pd.DataFrame(segments)
    df["similarity"] = sims
    return df.sort_values("start").reset_index(drop=True)


# ---------------------------------------------------
# ⭐ REASONING ENGINE (same as your logic)
# ---------------------------------------------------
_DEFAULT_STOPWORDS = {
    "the","and","is","in","to","a","an","for","of","on","with","that","this","it",
    "are","as","by","from","at","be","or","we","you","your","our","they","their",
    "i","my","me","was","but"
}

def extract_keywords(text, top_n=10):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in _DEFAULT_STOPWORDS and len(t) > 2]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]]


def generate_reasoning(title, description, df, sims, score):
    if df.empty:
        return "No transcript available."

    HIGH = 0.60
    MID = 0.40

    df["label"] = df["similarity"].apply(
        lambda x: "Highly Relevant" if x >= HIGH else
                  ("Partially Relevant" if x >= MID else "Irrelevant")
    )

    verdict = (
        "Highly Relevant" if score >= 70 else
        "Moderately Relevant" if score >= 40 else
        "Low Relevance"
    )

    reasoning = f"""
### 🧠 Final Verdict: **{verdict} ({score}%)**

- Segments analyzed: {len(df)}
- Avg similarity: {df["similarity"].mean():.3f}

### 📊 Segment Breakdown
- Highly Relevant: {(df['label']=='Highly Relevant').mean()*100:.1f}%
- Partially Relevant: {(df['label']=='Partially Relevant').mean()*100:.1f}%
- Irrelevant: {(df['label']=='Irrelevant').mean()*100:.1f}%

### 🔍 Top Evidence
"""
    top = df.sort_values("similarity", ascending=False).head(3)
    for _, r in top.iterrows():
        reasoning += f"- {int(r['start'])}s (sim {r['similarity']:.2f}) → {r['text'][:120]}...\n"

    return reasoning


# ---------------------------------------------------
# MASTER EVALUATION FUNCTION
# ---------------------------------------------------
def evaluate_video(title, description, url, manual, log, chunk_size):

    # 1️⃣ Try YouTube Transcript API
    if url:
        log("Fetching transcript via YouTube Transcript API...")
        txt, segs = get_transcript_youtube_api(url, log)
    else:
        txt, segs = None, None

    # 2️⃣ If transcript API fails → use OpenAI fallback
    if txt is None:
        log("Transcript API failed → Switching to OpenAI audio fallback...")

        audio = download_audio(url, log)
        if not audio:
            return 0, None, pd.DataFrame(), None, "Cannot download audio.", None

        txt, segs = openai_transcribe(audio, log)

    # 3️⃣ If still no transcript, stop
    if txt is None:
        return 0, None, pd.DataFrame(), None, "Transcript unavailable.", None

    # Use manual transcript if user provided
    if manual.strip():
        log("Using manually provided transcript.")
        txt = manual.strip()
        segs = chunk_manual_text(txt, size=chunk_size)

    # Chunk merging (if timestamped)
    log("Chunking segments...")
    segs = chunk_manual_text(txt, size=chunk_size)

    # Load model
    log("Loading SentenceTransformer model...")
    embedder = load_embedder()

    log("Computing similarity...")
    sims = get_similarity_scores(embedder, title, description, segs)

    df = build_df(segs, sims)
    score = compute_score_from_sims(sims)

    fig = px.bar(df, x="start", y="similarity")

    reasoning = generate_reasoning(title, description, df, sims, score)
    return score, fig, df, txt, None, reasoning


# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer")

show_runtime_warnings()

# LOG PANEL
log_container = st.sidebar.empty()
logger = make_logger(log_container)

# FORM
with st.form("form"):
    title = st.text_input("Video Title")
    description = st.text_input("Description")
    url = st.text_input("YouTube URL")
    manual = st.text_area("OR Paste Transcript Manually")
    chunk_size = st.slider("Words per chunk", 20, 200, 80)
    submit = st.form_submit_button("Evaluate")

if submit:
    st.session_state["logs"] = []
    logger("Starting evaluation...")

    score, fig, df, txt, err, reasoning = evaluate_video(
        title, description, url, manual, logger, chunk_size
    )

    if err:
        st.error(err)
        st.stop()

    st.metric("Relevance Score", f"{score}%")
    st.plotly_chart(fig)

    st.subheader("Reasoning")
    st.markdown(reasoning)

    st.subheader("Transcript")
    st.write(txt)

    st.subheader("Segments Table")
    st.dataframe(df)
