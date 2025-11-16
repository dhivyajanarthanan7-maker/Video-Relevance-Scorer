AI Video Relevance Scorer

Automatically evaluate how relevant a YouTube video is to a topic using AI embeddings, transcript extraction, and reasoning.

This Streamlit application analyzes a video’s transcript (captions or AI-generated transcription) and computes a Relevance Score using semantic similarity. It also highlights the most relevant parts of the video and provides a detailed reasoning report.

🚀 Live App

👉 https://video-relevance-scorer-6dpjnanyp59d3gc9ae9pa6.streamlit.app/
 

✨ Features
🎤 Transcript Extraction (3-layer fallback)

YouTube Transcript API (preferred)

OpenAI Whisper (gpt-4o-mini-transcribe)

Automatic fallback when captions are unavailable / blocked

Uses audio downloaded via yt-dlp

Manual transcript input

🧠 Semantic Similarity Analysis

Uses SentenceTransformer (all-MiniLM-L6-v2)

Generates embeddings for:

Video title + description

Transcript segments

Computes cosine similarity → Relevance Score (0–100%)

📊 Visual Insights

Relevance-over-time bar chart

Top relevant segments

Irrelevant & promotional segments

Keyword match analysis

Timeline breakdown

📝 Export Options

Download segmented transcript as CSV

Download full transcript as text

📡 Complete Logging System

Real-time logs visible in the sidebar (debug-friendly)

🧩 System Architecture
YouTube URL
      │
      ▼
Transcript Engine
      │
      ├─ YouTube Transcript API (preferred)
      ├─ yt-dlp → audio.mp3
      └─ OpenAI Whisper (gpt-4o-mini-transcribe)
      ▼
Transcript Segments
      ▼
SentenceTransformers Embeddings
      ▼
Cosine Similarity
      ▼
Relevance Score + Reasoning
      ▼
Streamlit UI Output

📁 Project Structure
Video-Relevance-Scorer/
│
├── app.py
├── requirements.txt
│
└── .streamlit/
       └── secrets.toml    (contains your OPENAI_API_KEY)

🛠 Installation (Local)
1. Clone the Repository
git clone https://github.com/dhivyajanarthanan7-maker/Video-Relevance-Scorer.git
cd Video-Relevance-Scorer

2. Install Dependencies
pip install -r requirements.txt

3. Add Your OpenAI API Key

Create folder + secrets file:

mkdir .streamlit


Inside .streamlit/secrets.toml:

OPENAI_API_KEY = "sk-your-key-here"

4. Run the App
streamlit run app.py

☁️ Deploy to Streamlit Cloud
Step 1 — Push to GitHub
git add .
git commit -m "deploy version"
git push origin main

Step 2 — Open Streamlit Cloud

https://share.streamlit.io

Create new app with:

Repo: your GitHub repo

Branch: main

File: app.py

Step 3 — Add Secret

Under Settings → Secrets:

OPENAI_API_KEY="sk-your-key-here"


Deploy 🚀
