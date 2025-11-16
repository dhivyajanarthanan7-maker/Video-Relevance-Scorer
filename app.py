import streamlit as st
import pandas as pd
import plotly.express as px
import re
import traceback
from typing import List, Dict

# -----------------------------------------
# Chunk manual text into segments
# -----------------------------------------
def chunk_manual_text(text: str, max_words: int = 80):
    words = text.split()
    segments = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        segments.append({"start": i, "end": i+max_words, "text": chunk})
    return segments

# -----------------------------------------
# Load embedder model lazily
# -----------------------------------------
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------
# Compute similarity scores
# -----------------------------------------
def similarity(embedder, title, desc, segments):
    from sklearn.metrics.pairwise import cosine_similarity

    query = title + " " + (desc or "")
    q_emb = embedder.encode([query])
    seg_texts = [s["text"] for s in segments]
    emb = embedder.encode(seg_texts)

    sims = cosine_similarity(q_emb, emb)[0]
    return sims

# -----------------------------------------
# Reasoning
# -----------------------------------------
STOPWORDS = {
    "the","and","is","in","to","a","an","for","of","on","with","that",
    "this","it","are","as","by","from","at","be","or","we","you","your",
    "our","they","their","i","my","me","was","but"
}

def extract_keywords(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def reasoning(title, desc, df, score):
    if df.empty:
        return "No transcript available."

    verdict = (
        "Highly Relevant" if score >= 70 else
        "Moderately Relevant" if score >= 40 else
        "Low Relevance"
    )

    keywords_list = extract_keywords(title + " " + desc)
    transcript_text = " ".join(df["text"]).lower()

    keyword_hits = {k: transcript_text.count(k) for k in keywords_list}

    top_segments = df.sort_values("similarity", ascending=False).head(3)

    out = f"""
### 🧠 Final Verdict: **{verdict} ({score}%)**

- Segments analyzed: **{len(df)}**
- Keyword matches: {keyword_hits}

### 🔍 Top Evidence
"""
    for _, r in top_segments.iterrows():
        out += f"- {r['start']} → sim {r['similarity']:.2f} → {r['text'][:150]}...\n"

    return out

# -----------------------------------------
# Main evaluation
# -----------------------------------------
def evaluate(title, desc, transcript_text, chunk_size):

    # Must have transcript
    if not transcript_text.strip():
        return 0, None, pd.DataFrame(), None, "Please paste a transcript.", None

    segments = chunk_manual_text(transcript_text.strip(), max_words=chunk_size)

    embedder = load_model()
    sims = similarity(embedder, title, desc, segments)

    df = pd.DataFrame({
        "start": [s["start"] for s in segments],
        "end": [s["end"] for s in segments],
        "text": [s["text"] for s in segments],
        "similarity": sims
    })

    score = round(float(sims.mean() * 100), 2)

    fig = px.bar(df, x="start", y="similarity", title="Relevance Over Time")

    reason = reasoning(title, desc, df, score)

    return score, fig, df, transcript_text, None, reason

# -----------------------------------------
# UI
# -----------------------------------------
st.set_page_config(page_title="AI Video Relevance Scorer", layout="wide")
st.title("🎯 AI Video Relevance Scorer (Manual Transcript Version)")

st.sidebar.header("Settings")
chunk_size_words = st.sidebar.number_input(
    "Chunk size (words per segment)", 20, 300, 80
)

with st.form("form"):
    t = st.text_input("Video Title")
    d = st.text_input("Description (optional)")
    u = st.text_input("YouTube URL (optional — only for reference)")
    m = st.text_area("Paste Transcript Here (REQUIRED)", height=200)
    go = st.form_submit_button("Evaluate")

if go:
    if not t:
        st.error("Please enter a title.")
        st.stop()

    score, fig, df, txt, err, reason = evaluate(
        t, d, m, chunk_size_words
    )

    if err:
        st.error(err)
        st.stop()

    st.metric("Overall Relevance Score", f"{score}%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Segments")
    top = df.sort_values("similarity", ascending=False).head(5)
    st.dataframe(top)

    st.subheader("Transcript")
    st.write(txt)

    st.subheader("Why this score?")
    st.markdown(reason)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "segments.csv", "text/csv")
