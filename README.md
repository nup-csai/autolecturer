# Avatar Project

Project for creating a talking avatar based on text or video using D-ID API.

## Features

- Extracting text from video files using speech recognition
- Creating a talking avatar that reads text from a file
- Downloading videos from YouTube by URL for processing
- Summarizing and highlighting main ideas from transcripts using Claude API

## Requirements

- Python 3.6+
- ffmpeg
- API keys:
  - D-ID API key (for avatar generation)
  - Claude API key (for text summarization)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/avatar_project.git
cd avatar_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
D_ID_API_KEY=your_did_api_key
ANTHROPIC_API_KEY=your_claude_api_key
```

## Usage

### Basic Usage

```bash
# Run both modes: extract text from video and create avatar
python run_avatar.py

# Only extract text from video
python run_avatar.py --extract

# Only create avatar
python run_avatar.py --avatar
```

### Using with YouTube

```bash
# Download video from YouTube and use it to extract text
python run_avatar.py --youtube "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID"

# Specify video language (for speech recognition)
python run_avatar.py --youtube "https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID" --language ru
```

### Text Summarization

```bash
# Summarize text from sample_text.txt and save to text_to_speak.txt
python summarize_text.py

# Customize summarization output
python summarize_text.py --max-length 400
```

## Project Structure

- `run_avatar.py` - Main script for running the complete pipeline
- `create_avatar_video.py` - Script for creating avatar with D-ID API
- `summarize_text.py` - Script for text summarization using Claude API
- `src/` - Source code for the project
  - `audio_transcriber.py` - Module for speech recognition from audio
  - `video_processor.py` - Module for video processing
- `Wav2Lip/` - Directory with resources for processing
  - `inputs/` - Input files
  - `outputs/` - Output files
  - `temp/` - Temporary files
- `sample_text.txt` - Source text (transcription)
- `text_to_speak.txt` - Summarized text for the avatar