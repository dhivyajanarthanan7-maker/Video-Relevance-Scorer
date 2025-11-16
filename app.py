"""
app.py - AI Video Relevance Scorer (Streamlit)

Features:
- Primary transcript fetch: youtube-transcript-api (if available)
- Fallback transcription using OpenAI's speech-to-text model (gpt-4o-mini-transcribe) when enabled
  (requires OPENAI_API_KEY set in environment)
- Manual transcript paste or file upload fallback
- Lazy SentenceTransformer embeddings and similarity scoring
- Real-time logs shown in the sidebar
- Exports: CSV of segments and transcript text download
- Robust error handling and helpful messages when libraries are missing
"""

import os
import re
import tempfile
import traceback
import time
from typing import List, Dict, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px

# Optional imports (may not be present in some deployment environments)
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

# yt_dlp is optional, used to extract audio for OpenAI transcription fallback
try:
    import yt_dlp  # noqa: F401
except Exception:
    yt_dlp = None

# openai client optional (used for gpt-4o-mini-transcribe fallback)
try:
    import openai
except Exception:
    openai = None
    missing_libs.append("openai (for transcription fallback)")


# -------------------------
# Simple in-app logger (sidebar)
# -------------------------
def make_logger(container):
    if "logs" not in st.session_state:
        st.session_state["logs"] = []
    def log(msg: str):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state["logs"].append(f"[{timestamp}] {msg}")
        last = "\n".join(st.session_state["logs"][-300:])
        container.code(last, language="text")
    return log


# -------------------------
# Runtime warnings helper
# -------------------------
def show_runtime_warnings():
    if missing_libs:
        st.warning(
            "Some Python packages are missing or failed to import: "
            + ", ".join(missing_libs)
            + ".\nCheck the installation / requirements on your deployment platform."
        )


# -------------------------
# Lazy embedder loader
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available.")
    return SentenceTransformer(model_name)


# -------------------------
# Utilities: video id extraction & vtt parsing
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


# -------------------------
# Transcript fetchers
# -------------------------
def get_youtube_transcript_via_api(video_id_or_url: str, log=print) -> Tuple[Optional[str], Optional[List[Dict]]]:
    if YouTubeTranscriptApi is None:
        log("youtube_transcript_api not available")
        return None, None

    vid = extract_video_id(video_id_or_url)
    if not vid:
        log("Could not extract video id")
        return None, None

    # Some youtube-transcript-api releases have different APIs (list_transcripts vs get_transcript)
    # Try both popular variants gracefully.
    try:
        log("Trying list_transcripts() ...")
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
        # prefer manually created first, then generated, prefer English
        try:
            # try find_transcript with preference list
            for pref in (["en", "en-US", "en-GB"], ["en"]):
                try:
                    tr = transcripts.find_transcript(pref)
                    if tr:
                        fetched = tr.fetch()
                        segs = [
                            {"start": float(t.get("start", 0.0)),
                             "end": float(t.get("start", 0.0) + t.get("duration", 0.0)),
                             "text": t.get("text", "").strip()}
                            for t in fetched if t.get("text", "").strip()
                        ]
                        if segs:
                            full = " ".join([s["text"] for s in segs])
                            log(f"Fetched {len(segs)} segments via list_transcripts()")
                            return full, segs
                except Exception:
                    continue
        except Exception:
            pass
    except AttributeError as e:
        log(f"list_transcripts() failed → {e}")
    except Exception as e:
        log(f"list_transcripts() raised: {e}")

    # Try older API variant get_transcript
    try:
        log("Trying get_transcript() ...")
        # get_transcript returns list of dicts
        fetched = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US", "en-GB"])
        if fetched:
            segs = [
                {"start": float(t.get("start", 0.0)),
                 "end": float(t.get("start", 0.0) + t.get("duration", 0.0)),
                 "text": t.get("text", "").strip()}
                for t in fetched if t.get("text", "").strip()
            ]
            full = " ".join([s["text"] for s in segs])
            log(f"Fetched {len(segs)} segments via get_transcript()")
            return full, segs
    except AttributeError as e:
        log(f"get_transcript() failed → {e}")
    except Exception as e:
        log(f"get_transcript() raised: {e}")

    log("❌ No transcript returned by API.")
    return None, None


# -------------------------
# OpenAI transcription fallback (gpt-4o-mini-transcribe)
# -------------------------
def transcribe_with_openai_from_file(audio_path: str, openai_api_key: Optional[str], log=print) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Send audio file to OpenAI transcription model (gpt-4o-mini-transcribe).
    Requirements: openai package installed and OPENAI_API_KEY set.

    Returns (full_text, segments=None) - segments can be None because transcription returns raw text.
    """
    if openai is None:
        log("openai package not installed; cannot use OpenAI transcription fallback.")
        return None, None
    if not openai_api_key:
        log("OPENAI_API_KEY not provided; cannot transcribe via OpenAI.")
        return None, None

    try:
        openai.api_key = openai_api_key
        log(f"Sending audio to OpenAI transcribe model... (this may take a while)")
        # NOTE: method name depends on openai package version.
        # We'll try the standard Audio Transcriptions interface used in many examples.
        with open(audio_path, "rb") as fh:
            # modern openai sdk: openai.Audio.transcriptions.create(...)
            try:
                resp = openai.Audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=fh
                )
            except Exception:
                # fallback to older wrapper
                resp = openai.Audio.transcription.create(
                    model="gpt-4o-mini-transcribe",
                    file=fh
                )
        # resp is expected to contain 'text' or similar
        text = None
        if isinstance(resp, dict):
            text = resp.get("text") or resp.get("transcript") or resp.get("data") and resp.get("data")[0].get("text")
        else:
            # try attribute access
            text = getattr(resp, "text", None) or getattr(resp, "transcript", None)

        if not text:
            log("OpenAI returned no text in response.")
            return None, None
        return text, None
    except Exception as e:
        log(f"OpenAI transcription error: {e}")
        return None, None


def transcribe_from_youtube_with_openai(video_url: str, openai_api_key: Optional[str], log=print) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    If yt_dlp is available, extract audio from the youtube URL into a temp file
    and call OpenAI transcription on it.
    """
    if yt_dlp is None:
        log("yt_dlp not installed; can't download audio from YouTube for transcription. Please upload audio/video file instead.")
        return None, None

    vid = extract_video_id(video_url)
    if not vid:
        log("Could not extract video ID from URL for audio extraction.")
        return None, None

    with tempfile.TemporaryDirectory() as tmpdir:
        # output audio file
        out_path = os.path.join(tmpdir, f"{vid}.mp3")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_path,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
        }
        try:
            log("Downloading audio via yt_dlp...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(video_url, download=True)
            if not os.path.exists(out_path):
                log("Audio download failed (no output file).")
                return None, None
            return transcribe_with_openai_from_file(out_path, openai_api_key, log=log)
        except Exception as e:
            log(f"yt_dlp audio download failed: {e}")
            return None, None


# -------------------------
# Chunkers: manual & timestamp-aware chunking
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
# Similarity and scoring helpers
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
# Reasoning engine (kept from your version)
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


# -------------------------
# Evaluate wrapper: combines all fallbacks
# -------------------------
def evaluate_video(title: str, description: str, url: Optional[str], manual_transcript: Optional[str],
                   chunk_size_words: int, chunk_window_seconds: int, openai_api_key: Optional[str], log=None):
    if log is None:
        log = print

    full_text = None
    segments = None

    # 1) Try youtube_transcript_api
    if url:
        log("Fetching transcript via YouTube Transcript API (if available)...")
        full_text, segments = get_youtube_transcript_via_api(url, log=log)

    # 2) If API didn't work and OPENAI key provided, try OpenAI transcription (download audio via yt_dlp if available)
    if (not full_text or not segments) and openai_api_key:
        log("Attempting OpenAI transcription fallback (gpt-4o-mini-transcribe) ...")
        # First try to fetch via API again (in case different endpoints exist)
        try:
            full_text_api, segments_api = get_youtube_transcript_via_api(url, log=log)
            if full_text_api and segments_api:
                full_text, segments = full_text_api, segments_api
        except Exception:
            pass
        if not full_text and url:
            # try audio extraction + openai transcription
            txt, _ = transcribe_from_youtube_with_openai(url, openai_api_key, log=log)
            if txt:
                full_text = txt
                segments = None  # raw transcript without timestamps

    # 3) If still no transcript and user provided manual transcript, use it
    if manual_transcript and manual_transcript.strip():
        log("Using manual transcript provided by user.")
        full_text = manual_transcript.strip()
        segments = chunk_manual_text(full_text, max_words=chunk_size_words)

    # 4) If still no transcript, ask user to upload audio/video file (handled in UI)
    if not segments and full_text and isinstance(full_text, str):
        # create simple chunks from the raw text
        segments = chunk_manual_text(full_text, max_words=chunk_size_words)
        log(f"Created {len(segments)} chunks from full transcript text.")

    if not segments:
        return 0.0, None, pd.DataFrame(), None, "No transcript available (try uploading audio/video or paste transcript)", None

    # If segments have timestamps, merge timestamp-aware windows
    if isinstance(segments[0].get("start"), (int, float)):
        merged = chunk_youtube_segments(segments, max_words=chunk_size_words, max_window_seconds=chunk_window_seconds)
        if merged:
            segments = merged
            log(f"Chunked into {len(segments)} timestamp-aware segments.")

    # Load model
    try:
        log("Loading embedder model...")
        embedder = load_embedder()
        log("Model loaded.")
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Model load failed: {e}\n\n{tb}", None

    # Compute similarity
    try:
        log("Computing embeddings and similarity...")
        sims = get_similarity_scores(embedder, title, description, segments)
        log("Similarity done.")
    except Exception as e:
        tb = traceback.format_exc()
        return 0.0, None, pd.DataFrame(), full_text, f"Embedding / similarity error: {e}\n\n{tb}", None

    score = compute_score_from_sims(sims)
    df = build_df(segments, sims)

    # Build plot
    try:
        fig = px.bar(df, x="start", y="similarity", hover_data=["start", "end", "text"], title="Relevance Over Time")
        fig.update_layout(xaxis_title="Segment start (seconds or index)", yaxis_title="Cosine similarity")
    except Exception:
        fig = None

    reasoning = generate_reasoning(title, description, df, sims, score)
    return score, fig, df, full_text, None, reasoning


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer (with OpenAI fallback)")

show_runtime_warnings()

# Sidebar settings + logs
st.sidebar.header("Settings")
chunk_size_words = st.sidebar.number_input("Chunk size (words per segment)", min_value=10, max_value=500, value=80, step=10)
chunk_window_seconds = st.sidebar.number_input("Max window length (seconds) for timestamped chunks", min_value=5, max_value=300, value=30, step=5)
top_k = st.sidebar.number_input("Top K segments to show", min_value=1, max_value=50, value=5, step=1)
show_low_similarity = st.sidebar.checkbox("Also show low-similarity segments", value=False)

# OpenAI toggle
st.sidebar.markdown("**OpenAI transcription fallback (optional)**")
openai_api_key = st.sidebar.text_input("OpenAI API Key (paste or set OPENAI_API_KEY env var)", value=os.getenv("OPENAI_API_KEY", ""), type="password")
use_openai_fallback = st.sidebar.checkbox("Enable OpenAI fallback (gpt-4o-mini-transcribe)", value=bool(openai_api_key))

# Logging panel UI
log_container = st.sidebar.empty()
log_container.markdown("**Realtime logs**")
logger = make_logger(log_container)

with st.form(key="eval_form"):
    title = st.text_input("Video Title")
    description = st.text_input("Video Description (optional)")
    url = st.text_input("YouTube URL (optional)")
    manual_transcript = st.text_area("OR Paste Transcript Manually (optional)", height=160)
    upload_file = st.file_uploader("Or upload a video/audio file for transcription (mp3, mp4, wav). If provided and API fallback enabled, it will be used.", type=["mp3", "wav", "m4a", "mp4", "webm"])
    submitted = st.form_submit_button("Evaluate")

if submitted:
    st.session_state["logs"] = []
    logger("Starting evaluation...")

    if not title:
        st.error("Please enter a title")
        st.stop()
    if not url and not manual_transcript.strip() and upload_file is None:
        st.error("Please enter a YouTube URL, paste a transcript, or upload a media file")
        st.stop()

    # If upload_file provided and openai fallback enabled, save temp file and transcribe
    file_temp_path = None
    uploaded_transcript_text = None
    if upload_file is not None and use_openai_fallback and openai_api_key:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.name)[1])
            tmp.write(upload_file.getbuffer())
            tmp.flush()
            tmp.close()
            file_temp_path = tmp.name
            logger("Saved uploaded media to temporary file for transcription.")
            txt, _ = transcribe_with_openai_from_file(file_temp_path, openai_api_key, log=logger)
            if txt:
                uploaded_transcript_text = txt
                logger("Uploaded media transcribed via OpenAI fallback.")
        except Exception as e:
            logger(f"Upload transcription failed: {e}")

    # If url present, attempt evaluation; pass openai key to allow fallback
    try:
        score, fig, df, transcript, error_msg, reasoning = evaluate_video(
            title=title,
            description=description,
            url=url.strip() if url else None,
            manual_transcript=(uploaded_transcript_text or manual_transcript).strip() if (uploaded_transcript_text or manual_transcript) else None,
            chunk_size_words=int(chunk_size_words),
            chunk_window_seconds=int(chunk_window_seconds),
            openai_api_key=openai_api_key if use_openai_fallback else None,
            log=logger
        )
    except Exception as e:
        logger(f"Unhandled error during evaluation: {e}")
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
        st.error("Could not retrieve or chunk the transcript. Try uploading a file, paste transcript manually, or enable OpenAI fallback with a valid API key.")
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

