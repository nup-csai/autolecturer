"""Configuration settings for the lecture summarizer."""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
OUTPUT_DIR = DATA_DIR / "output"
VIDEO_CLIPS_DIR = DATA_DIR / "video_clips"

# Create directories if they don't exist
for dir_path in [INPUT_DIR, PROCESSED_DIR, AUDIO_DIR, TRANSCRIPT_DIR, OUTPUT_DIR, VIDEO_CLIPS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Audio extraction settings
AUDIO_SETTINGS = {
    "sample_rate": 16000,
    "channels": 1,
    "format": "wav",
    "codec": "pcm_s16le",
}

# Speech-to-text settings
STT_SETTINGS = {
    "model_type": "faster-whisper",  # Options: "faster-whisper", "vosk", "nemo"
    "model_size": "base",  # Options for faster-whisper: "tiny", "base", "small", "medium", "large-v2"
    "language": "en",
    "beam_size": 5,
    "compute_type": "float32",  # Options: "float32", "float16", "int8"
    "device": "cpu",  # Options: "cpu", "cuda" (if GPU available)
    "vosk_model_path": "models/vosk",  # Path to Vosk model
    "sample_rate": 16000,
}

# Text processing settings
TEXT_PROCESSING = {
    # Basic cleaning options
    "remove_filler_words": True,
    "filler_words": [
        "um", "uh", "ah", "er", "like", "you know", "sort of", "kind of", 
        "basically", "actually", "literally", "so", "well", "right", "okay", 
        "so yeah", "I mean", "you see", "anyway", "anyhow", "all right", "alright"
    ],
    "spell_check": True,
    "normalize_case": True,
    "lemmatize": False,  # May change meaning in educational context
    
    # Enhanced text cleaning options
    "enhance_readability": True,
    "fix_punctuation": True,
}

# Chunking settings
CHUNKING = {
    "method": "timestamp",  # "sentence", "token", "timestamp"
    "max_tokens": 512,      # Maximum tokens per chunk
    "overlap_tokens": 50,   # Overlap tokens between chunks
    "min_chunk_size": 100,  # Minimum number of tokens in a chunk
    "max_video_duration": 120,  # Maximum duration for video clips in seconds (2 minutes)
}

# Video processing settings
VIDEO_SETTINGS = {
    "resolution": (1280, 720),   # Output video resolution
    "fps": 30,                   # Frames per second
    "codec": "libx264",          # Video codec
    "audio_codec": "aac",        # Audio codec
    "bitrate": "2000k",          # Video bitrate
    "audio_bitrate": "128k",     # Audio bitrate
    "thumbnail_interval": 30,    # Generate thumbnail every 30 seconds
}