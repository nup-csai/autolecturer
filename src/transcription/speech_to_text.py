"""Speech-to-text transcription module."""
import os
import json
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from src.config import STT_SETTINGS, TRANSCRIPT_DIR
from src.utils.helpers import timing_decorator, ensure_dir, save_json, generate_output_filename

logger = logging.getLogger(__name__)


class BaseTranscriber(ABC):
    """Abstract base class for all transcribers."""

    @abstractmethod
    def transcribe(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Transcribe an audio file to text."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the transcription model."""
        pass


class FasterWhisperTranscriber(BaseTranscriber):
    """Transcriber using Faster Whisper model."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the transcriber with settings."""
        self.settings = settings or STT_SETTINGS
        self.model_size = self.settings.get("model_size", "base")
        self.language = self.settings.get("language", "en")
        self.beam_size = self.settings.get("beam_size", 5)
        self.device = self.settings.get("device", "cpu")
        self.compute_type = self.settings.get("compute_type", "float32")
        self.model = None
        logger.info(f"Initialized FasterWhisperTranscriber with settings: {self.settings}")

    def load_model(self) -> None:
        """Load the Faster Whisper model."""
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster Whisper model: {self.model_size}")

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.expanduser("~"), ".cache", "faster_whisper")
            )
            logger.info("Successfully loaded Faster Whisper model")

        except ImportError:
            logger.error("Failed to import faster_whisper. Make sure it's installed.")
            raise
        except Exception as e:
            logger.error(f"Error loading Faster Whisper model: {str(e)}")
            raise

    @timing_decorator
    def transcribe(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Transcribe an audio file using Faster Whisper."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or TRANSCRIPT_DIR)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Run transcription
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size=self.beam_size,
                language=self.language,
            )

            # Process segments
            transcript_text = ""
            segments_data = []

            for segment in segments:
                transcript_text += segment.text + " "
                segments_data.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                })

            # Prepare metadata
            metadata = {
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segments_data,
                "model_size": self.model_size,
            }

            # Generate output filenames
            json_path = generate_output_filename(
                audio_path, output_dir, suffix="transcript", extension=".json"
            )
            text_path = generate_output_filename(
                audio_path, output_dir, suffix="transcript", extension=".txt"
            )

            # Save output files
            save_json(metadata, json_path)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(transcript_text.strip())

            logger.info(f"Transcription completed. Text saved to {text_path}, metadata to {json_path}")

            return transcript_text.strip(), metadata

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise


class VoskTranscriber(BaseTranscriber):
    """Transcriber using Vosk API."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the transcriber with settings."""
        self.settings = settings or STT_SETTINGS
        self.model_path = self.settings.get("vosk_model_path", "models/vosk")
        self.sample_rate = self.settings.get("sample_rate", 16000)
        self.model = None
        logger.info(f"Initialized VoskTranscriber with settings: {self.settings}")

    def load_model(self) -> None:
        """Load the Vosk model."""
        try:
            from vosk import Model, KaldiRecognizer
            import wave

            model_path = os.path.expanduser(self.model_path)
            if not os.path.exists(model_path) or not os.listdir(model_path):
                logger.error(f"Vosk model not found at {model_path}. Using system language model instead.")
                logger.info("Loading Vosk system model...")
                self.model = Model(lang="en-us")
            else:
                logger.info(f"Loading Vosk model from {model_path}")
                self.model = Model(model_path)
            
            logger.info("Successfully loaded Vosk model")

        except ImportError:
            logger.error("Failed to import vosk. Make sure it's installed.")
            raise
        except Exception as e:
            logger.error(f"Error loading Vosk model: {str(e)}")
            raise

    @timing_decorator
    def transcribe(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Transcribe an audio file using Vosk."""
        from vosk import KaldiRecognizer
        import wave
        import json

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or TRANSCRIPT_DIR)

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            # Open and read audio file
            with wave.open(str(audio_path), "rb") as wf:
                # Check audio format
                if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                    logger.warning("Audio file must be in WAV format (mono, 16-bit PCM)")

                # Create recognizer with sample rate
                rec = KaldiRecognizer(self.model, wf.getframerate())
                rec.SetWords(True)

                # Process audio file
                segments_data = []
                transcript_text = ""

                while True:
                    data = wf.readframes(4000)  # Read in chunks
                    if len(data) == 0:
                        break

                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        if result.get("text", ""):
                            transcript_text += result["text"] + " "
                            segments_data.append(result)

                # Get final result
                final_result = json.loads(rec.FinalResult())
                if final_result.get("text", ""):
                    transcript_text += final_result["text"] + " "
                    segments_data.append(final_result)

            # Prepare metadata
            metadata = {
                "segments": segments_data,
                "model": "vosk",
                "model_path": self.model_path,
            }

            # Generate output filenames
            json_path = generate_output_filename(
                audio_path, output_dir, suffix="transcript_vosk", extension=".json"
            )
            text_path = generate_output_filename(
                audio_path, output_dir, suffix="transcript_vosk", extension=".txt"
            )

            # Save output files
            save_json(metadata, json_path)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(transcript_text.strip())

            logger.info(f"Transcription completed. Text saved to {text_path}, metadata to {json_path}")

            return transcript_text.strip(), metadata

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise


class TranscriberFactory:
    """Factory for creating transcribers based on configuration."""

    @staticmethod
    def create_transcriber(settings: Optional[Dict[str, Any]] = None) -> BaseTranscriber:
        """Create a transcriber based on settings."""
        settings = settings or STT_SETTINGS
        model_type = settings.get("model_type", "faster-whisper")

        if model_type == "faster-whisper":
            return FasterWhisperTranscriber(settings)
        elif model_type == "vosk":
            return VoskTranscriber(settings)
        else:
            raise ValueError(f"Unsupported transcriber type: {model_type}")