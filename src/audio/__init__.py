"""Audio processing module for lecture summarizer."""
from src.audio.extractor import AudioExtractor
# Temporarily comment this out due to pydub/audioop issue
# from src.audio.processor import AudioProcessor

__all__ = ["AudioExtractor"]