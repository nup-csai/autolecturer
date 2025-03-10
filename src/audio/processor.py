"""Module for processing audio files."""
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np
import pydub
from pydub import AudioSegment
import ffmpeg

from src.config import AUDIO_SETTINGS, PROCESSED_DIR
from src.utils.helpers import timing_decorator, ensure_dir, generate_output_filename

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Class for processing audio files."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the audio processor with settings.

        Args:
            settings: Audio processing settings. If None, use default settings from config.
        """
        self.settings = settings or AUDIO_SETTINGS
        self.sample_rate = self.settings.get("sample_rate", 16000)
        self.channels = self.settings.get("channels", 1)
        logger.info(f"Initialized AudioProcessor with settings: {self.settings}")

    @timing_decorator
    def normalize_audio(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            target_db: float = -20.0,
    ) -> Path:
        """Normalize audio volume.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the processed audio. If None, use default processed directory.
            target_db: Target dB level for normalization.

        Returns:
            Path to the normalized audio file.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or PROCESSED_DIR)

        # Generate output filename
        output_path = generate_output_filename(
            audio_path, output_dir, suffix="normalized"
        )

        logger.info(f"Normalizing audio: {audio_path} to {output_path}")

        # Load audio using pydub
        audio = AudioSegment.from_file(audio_path)

        # Calculate current dB level
        current_db = audio.dBFS

        # Calculate the change needed
        db_change = target_db - current_db

        # Apply gain adjustment
        normalized_audio = audio.apply_gain(db_change)

        # Export the normalized audio
        normalized_audio.export(output_path, format=output_path.suffix.lstrip("."))

        logger.info(f"Successfully normalized audio to {output_path} (Target: {target_db} dB)")
        return output_path

    @timing_decorator
    def remove_noise(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            noise_reduction_amount: float = 0.2,
    ) -> Path:
        """Remove background noise from audio.

        Note: This is a simple implementation. For better results, consider using
              dedicated noise reduction libraries or tools.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the processed audio. If None, use default processed directory.
            noise_reduction_amount: Amount of noise reduction to apply (0.0-1.0).

        Returns:
            Path to the noise-reduced audio file.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or PROCESSED_DIR)

        # Generate output filename
        output_path = generate_output_filename(
            audio_path, output_dir, suffix="denoised"
        )

        logger.info(f"Removing noise from audio: {audio_path} to {output_path}")

        try:
            # Use ffmpeg with noise reduction filter
            # This uses the built-in ffmpeg highpass and lowpass filters for a simple noise reduction
            (
                ffmpeg
                .input(str(audio_path))
                .filter("highpass", f=200)  # Remove low frequencies (adjust as needed)
                .filter("lowpass", f=3000)  # Remove high frequencies (adjust as needed)
                .filter("afftdn", nr=noise_reduction_amount, nf=noise_reduction_amount)  # FFT-based denoiser
                .output(str(output_path), acodec="pcm_s16le", ar=self.sample_rate, ac=self.channels)
                .global_args("-y")  # Overwrite if exists
                .run(capture_stdout=True, capture_stderr=True)
            )

            logger.info(f"Successfully denoised audio to {output_path}")
            return output_path

        except ffmpeg.Error as e:
            logger.error(f"Error denoising audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    @timing_decorator
    def split_audio(
            self,
            audio_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            segment_length_ms: int = 60000,  # 1 minute
            min_silence_length_ms: int = 500,
            silence_threshold_db: int = -40,
    ) -> list[Path]:
        """Split audio into segments based on silence detection.

        Args:
            audio_path: Path to the audio file.
            output_dir: Directory to save the split audio segments.
            segment_length_ms: Maximum length of each segment in milliseconds.
            min_silence_length_ms: Minimum length of silence to use for splitting.
            silence_threshold_db: dB level below which is considered silence.

        Returns:
            List of paths to the split audio segments.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or PROCESSED_DIR / "segments")

        logger.info(f"Splitting audio: {audio_path}")

        # Load audio using pydub
        audio = AudioSegment.from_file(audio_path)

        # Split audio based on silence
        chunks = pydub.silence.split_on_silence(
            audio,
            min_silence_len=min_silence_length_ms,
            silence_thresh=silence_threshold_db,
            keep_silence=400,  # Keep some silence at the beginning and end
        )

        # If no silence detected or only one chunk, split by fixed length
        if len(chunks) <= 1:
            logger.info("No silence detected, splitting by fixed length")
            chunks = [audio[i:i + segment_length_ms] for i in range(0, len(audio), segment_length_ms)]

        # Save chunks to files
        output_paths = []
        for i, chunk in enumerate(chunks):
            chunk_path = output_dir / f"{audio_path.stem}_segment_{i:03d}{audio_path.suffix}"
            chunk.export(chunk_path, format=chunk_path.suffix.lstrip("."))
            output_paths.append(chunk_path)

        logger.info(f"Successfully split audio into {len(output_paths)} segments")
        return output_paths