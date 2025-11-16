🎯 AI Video Relevance Scorer

A Streamlit web application that evaluates how relevant a video’s content is to a given topic using SentenceTransformers and cosine similarity.

This version is simple, stable, and API-free — no YouTube API, no OpenAI key, no yt-dlp required.
📌 Just paste the video transcript manually, and the system will analyze relevance!

🚀 Features
✅ 1. Manual Transcript Input (No API Needed)

YouTube transcripts often fail due to bot checks or location restrictions.

This version accepts manual transcript paste, ensuring 100% reliability.

✅ 2. AI-Powered Relevance Scoring

Uses sentence embeddings from:

all-MiniLM-L6-v2 (SentenceTransformers)


Computes relevance with:

Cosine similarity

✅ 3. Segment-Level Analysis

Breaks transcript into chunks (default: 80 words)

Computes similarity for each chunk

Displays most relevant and least relevant parts

✅ 4. Smart Explanation (Reasoning Engine)

You get:

Final Verdict (High / Moderate / Low Relevance)

Keyword match analysis

Top evidence segments

Timeline insights

✅ 5. Beautiful Plot

Interactive bar chart showing relevance over time.

✅ 6. Downloads

Export:

Segments CSV

Transcript

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Embeddings	SentenceTransformers
Similarity	scikit-learn cosine similarity
Plotting	Plotly
Language	Python

No external API keys. No YouTube API. No OpenAI usage.
💯 Fully free to run and deploy.

📦 Installation
1️⃣ Clone Repository
git clone https://github.com/dhivyajanarthanan7-maker/Video-Relevance-Scorer
cd Video-Relevance-Scorer

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the App
streamlit run app.py

🧪 How to Use
Step 1 — Enter Video Title

Describe the topic or subject of the video.

Step 2 — (Optional) Add Description

Helps improve relevance measurement.

Step 3 — (Optional) Paste YouTube URL

Only for visual reference — not used for fetching transcript.

Step 4 — Paste Transcript

Get transcript using any method:

YouTube “Show Transcript” option

Tools like downsub.com

Manual captions

Step 5 — Click Evaluate

You will get:

🎯 Overall Relevance Score (0–100)

📊 Relevance Over Time chart

🔍 Top relevant segments

⚠ Least relevant segments

🧠 Reasoning & explanation

📂 Project Structure
├── app.py                 # Main Streamlit app (manual transcript version)
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
└── .streamlit/
    └── config (if any)

📈 Example Output

Relevance score: 82%

Verdict: Highly Relevant

Top segments highlight where the video strongly matches the topic.

Timeline shows how relevance changes across the video.

❗ Why Manual Transcript Version?

YouTube has:

CAPTCHA blocks

bot detection

region restrictions

transcript not available

API blocked (429 errors)

OpenAI transcription:

Requires API key

Costs money

Hit your quota

Therefore, the manual-transcript version is the most stable and simplest for academic submission.

🏁 Conclusion

This project demonstrates:

Understanding of NLP embeddings

Practical cosine similarity scoring

Streamlit UI development

Full ML pipeline without needing heavy dependencies

Perfect for:

Capstone projects

Portfolio

Resume projects

College showcase

❤️ Author

Dhivya Janarthanan
