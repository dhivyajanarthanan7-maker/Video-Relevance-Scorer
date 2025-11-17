🎯 AI Video Relevance Scorer

Evaluate how relevant a YouTube video is to a given topic using embeddings, transcript extraction, and precise reasoning.

This Streamlit app analyzes a YouTube video’s transcript (or a manually provided transcript) and computes how closely it aligns with a provided Title + Description using SentenceTransformer embeddings. It includes:

Automatic transcript fetching (YouTube API, yt-dlp, OpenAI fallback)

Smart transcript chunking

Embedding-based similarity scoring

Relevance visualization over time

Deep reasoning using keyword overlap, promo detection & evidence segments

Downloadable CSV and TXT outputs

🚀 Features
🔍 Transcript Extraction (Robust Multi-Source Pipeline)

The app attempts 3 methods in order:

YouTubeTranscriptApi

yt-dlp subtitles (auto-generated or manual captions)

OpenAI transcription fallback (optional)

You may also paste a manual transcript directly.

🧩 Chunking Strategy

The tool supports two chunking modes:

YouTube timestamp-based merging

Configurable: max words & max window duration

Manual text chunking

Configurable: words per chunk

🧠 Relevance Scoring

Uses:

SentenceTransformer all-MiniLM-L6-v2 embeddings

Cosine similarity per segment

Segment-wise relevance bar chart

Overall score (%) = mean similarity × 100

📌 Precise Reasoning Engine

Generates an interpretable explanation including:

High/mid/low similarity segment distribution

Keyword overlap analysis

Promotional content detection

Early strong matches

Off-topic segments

Top & bottom evidence segments

📤 Exporting

Download segments CSV

Download full transcript TXT

View full transcript & segment table

Copy top segments easily

🛠 Installation
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

4. (Optional) Add OpenAI API key

Create .env or set environment variable:

export OPENAI_API_KEY="your_key_here"


Windows:

set OPENAI_API_KEY=your_key_here

▶️ Running the App
streamlit run app.py


Open your browser at:

http://localhost:8501

📦 Requirements List (recommended for requirements.txt)
streamlit
pandas
numpy
plotly
sentence-transformers
scikit-learn
youtube-transcript-api
yt-dlp
openai


(OpenAI is optional — only if you want fallback transcription.)

🗂 Project Structure
├── app.py
├── README.md
├── requirements.txt
└── assets/ (optional)

🧧 Environment Variables
Name	Purpose
OPENAI_API_KEY	Enables OpenAI fallback transcription
🔒 Notes & Limitations

Auto-captions may be noisy; manual transcript gives best results.

Similarity score depends on text semantics, not keyword matching.

OpenAI fallback costs API credits; disabled if key is missing.

✨ Future Improvements

Support for multilingual transcripts

Support for local video file upload

Support for alternative embedding models

API endpoint for programmatic scoring

🤝 Contributing

Pull requests are welcome!
Feel free to open issues/ideas for improvements.

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.

❤️ Author

Dhivya Janarthanan
