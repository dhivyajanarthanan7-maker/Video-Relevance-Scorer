import streamlit as st
import pandas as pd
import plotly.express as px
import traceback
import re
from typing import List, Dict, Optional

from openai import OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Try imports
missing = []

try:
    from sentence_transformers import SentenceTransformer
except:
except:
    SentenceTransformer = None
    missing.append("sentence-transformers")

try:
    from sklearn.metrics.pairwise import cosine_similarity
except:
except:
    cosine_similarity = None
    missing.append("scikit-learn")

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except:
except:
    YouTubeTranscriptApi = None
    missing.append("youtube-transcript-api")

try:
    import numpy as np
except:
    import numpy as np
except:
    np = None
    missing.append("numpy")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def extract_video_id(url):
    if "youtu" not in url:
        return url
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    vid = parse_qs(parsed.query).get("v", [None])[0]
    if vid:
        return vid
    return parsed.path.split("/")[-1]


def get_youtube_transcript_api(url, log):
    """Primary attempt: youtube_transcript_api"""
    if YouTubeTranscriptApi is None:
        log("youtube_transcript_api not available.")
        return None, None

    vid = extract_video_id(url)
    log(f"Extracted video ID: {vid}")

    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(vid)
    except Exception as e:
        log(f"list_transcripts() failed:\n{e}")
        return None, None

    # Try any transcript available
    for t in transcripts:
        try:
            data = t.fetch()
            segments = []
            for item in data:
                txt = item.get("text", "").strip()
                if not txt:
                    continue
                start = float(item["start"])
                end = start + float(item["duration"])
                segments.append({"start": start, "end": end, "text": txt})
            full_text = " ".join([s["text"] for s in segments])
            log(f"Fetched {len(segments)} transcript segments.")
            return full_text, segments
        except:
            continue

    log("No transcript found via YouTube API.")
    return None, None


def transcribe_via_openai(url, log):
    """Download + transcribe YouTube audio using OpenAI's transcription."""
    try:
        log("Downloading audio via OpenAI (YouTube URL)...")

        # Step 1 — Download audio directly from YouTube using OpenAI
        audio_file = client.audio.from_url(
            url=url,
            format="wav"    # or mp3
        )

        log("Transcribing using gpt-4o-mini-transcribe...")

        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe"
        )

        text = transcript.text
        log("Transcription completed.")

        # Convert to segments (simple chunking)
        words = text.split()
        segments = []
        size = 80
        for i in range(0, len(words), size):
            chunk = " ".join(words[i:i+size])
            segments.append({"start": i, "end": i+size, "text": chunk})

        return text, segments

    except Exception as e:
        log(f"OpenAI transcription failed: {e}")
        return None, None


def chunk_youtube(segments, max_words=80, max_seconds=30):
    return segments  # Already timestamped or chunked — skip merging


# ------------------------------------------------------------
# Similarity
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers missing")
    return SentenceTransformer("all-MiniLM-L6-v2")


def similarity(embedder, title, desc, segments):
    if cosine_similarity is None:
        raise RuntimeError("scikit-learn missing")

    q = title + " " + (desc or "")
    q_emb = embedder.encode([q])
    texts = [s["text"] for s in segments]
    emb = embedder.encode(texts)
    sims = cosine_similarity(q_emb, emb)[0]
    return sims


# ------------------------------------------------------------
# Reasoning
# ------------------------------------------------------------
STOPWORDS = {"the","and","is","in","to","a","an","for","of","on",
             "with","that","this","it","are","as","by","from","at",
             "be","or","we","you","your","our","they","their","i",
             "my","me","was","but"}

def keywords(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def reasoning(title, desc, df, sims, score):
    if df is None or df.empty:
        return "No transcript to analyze."

    verdict = ("Highly relevant" if score >= 70
               else "Moderately relevant" if score >= 40
               else "Low relevance")

    k = keywords(title + " " + (desc or ""))
    hits = {kw: " ".join(df["text"]).lower().count(kw) for kw in k}

    out = f"""
### 🧠 Final Verdict: **{verdict} ({score}%)**

- Segments analyzed: **{len(df)}**
- Avg similarity: **{df['similarity'].mean():.3f}**
- Keyword matches: {hits}

### 📊 Top Evidence
"""
    top = df.sort_values("similarity", ascending=False).head(3)
    for _, r in top.iterrows():
        out += f"- {int(r['start'])}s → {r['similarity']:.2f} → {r['text'][:120]}...\n"

    return out


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(title, desc, url, manual, log, chunk_words):

    # 1) Try YouTube API
    if url:
        log("Fetching transcript via YouTube...")
        full_text, segments = get_youtube_transcript_api(url, log)
    else:
        full_text, segments = None, None

    # 2) If not available → OpenAI fallback
    if not segments:
        log("YouTube API failed → switching to OpenAI transcription...")
        full_text, segments = transcribe_via_openai(url, log)

    # 3) Manual override
    if manual.strip():
        log("Using manual transcript.")
        text = manual.strip()
        words = text.split()
        segments = []
        for i in range(0, len(words), chunk_words):
            chunk = " ".join(words[i:i+chunk_words])
            segments.append({"start": i, "end": i+chunk_words, "text": chunk})
        full_text = text

    if not segments:
        return 0, None, pd.DataFrame(), None, "No transcript available", None

    # Similarity
    embedder = load_model()
    sims = similarity(embedder, title, desc, segments)
    score = round(float(sims.mean() * 100), 2)

    df = pd.DataFrame({
        "start":[s["start"] for s in segments],
        "end":[s["end"] for s in segments],
        "text":[s["text"] for s in segments],
        "similarity": sims
    })

    # Plot
    fig = px.bar(df, x="start", y="similarity", title="Relevance Over Time")

    # Reason
    reason = reasoning(title, desc, df, sims, score)

    return score, fig, df, full_text, None, reason


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer")

st.sidebar.header("Settings")
chunk_words = st.sidebar.number_input("Chunk size (words per segment)", 20, 300, 80)

# Logging panel
log_box = st.sidebar.empty()
logs = []
def log(msg):
    logs.append(msg)
    log_box.code("\n".join(logs[-50:]), language="text")

with st.form("form"):
    t = st.text_input("Title")
    d = st.text_input("Description (optional)")
    u = st.text_input("YouTube URL (optional)")
    m = st.text_area("OR paste transcript manually", height=150)
    go = st.form_submit_button("Evaluate")

if go:
    logs.clear()
    log("Starting evaluation...")

    score, fig, df, transcript, err, reason = evaluate(
        t, d, u, m, log, chunk_words
    )

    if err:
        st.error(err)
    if err:
        st.error(err)
        st.stop()

    st.metric("Overall Relevance", f"{score}%")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Segments")
    top = df.sort_values("similarity", ascending=False).head(5)
    st.dataframe(top)

    st.subheader("Transcript")
    st.write(transcript)

    st.subheader("Why this score?")
    st.markdown(reason)
