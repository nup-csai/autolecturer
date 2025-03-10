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

# Create directories if they don't exist
for dir_path in [INPUT_DIR, PROCESSED_DIR, AUDIO_DIR, TRANSCRIPT_DIR]:
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
    "model_size": "base",  # for faster-whisper: "tiny", "base", "small", "medium", "large-v2"
    "language": "en",
    "beam_size": 5,
    "device": "cpu",  # "cpu" or "cuda"
    "compute_type": "float32",  # "float32", "float16", "int8"
}

# Text processing settings
TEXT_PROCESSING = {
    "remove_filler_words": True,
    "filler_words": ["um", "uh", "ah", "er", "like", "you know", "sort of", "kind of"],
    "spell_check": True,
    "normalize_case": True,
    "lemmatize": False,  # May change meaning in educational context
}

# Chunking settings
CHUNKING = {
    "method": "sentence",  # "sentence", "token", "timestamp"
    "max_tokens": 512,
    "overlap_tokens": 50,
    "min_chunk_size": 100,  # Minimum number of tokens in a chunk
}