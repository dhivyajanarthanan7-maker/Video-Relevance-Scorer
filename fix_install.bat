@echo off
pip uninstall -y youtube-transcript-api
pip install youtube-transcript-api
python -c "from youtube_transcript_api import YouTubeTranscriptApi; print('Available methods:'); print([m for m in dir(YouTubeTranscriptApi) if not m.startswith('_')])"
