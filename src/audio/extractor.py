"""Module for extracting audio from video files."""
import os
import subprocess
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

import ffmpeg

from src.config import AUDIO_SETTINGS, AUDIO_DIR
from src.utils.helpers import timing_decorator, ensure_dir, generate_output_filename

logger = logging.getLogger(__name__)


class AudioExtractor:
    """Class for extracting audio from video files."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the audio extractor with settings.

        Args:
            settings: Audio extraction settings. If None, use default settings from config.
        """
        self.settings = settings or AUDIO_SETTINGS
        self.sample_rate = self.settings.get("sample_rate", 16000)
        self.channels = self.settings.get("channels", 1)
        self.format = self.settings.get("format", "wav")
        self.codec = self.settings.get("codec", "pcm_s16le")
        logger.info(f"Initialized AudioExtractor with settings: {self.settings}")

    @timing_decorator
    def extract_audio(
            self,
            video_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None,
            output_filename: Optional[str] = None,
    ) -> Path:
        """Extract audio from a video file.

        Args:
            video_path: Path to the video file.
            output_dir: Directory to save the extracted audio. If None, use default audio directory.
            output_filename: Filename for the extracted audio. If None, generate from video filename.

        Returns:
            Path to the extracted audio file.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or AUDIO_DIR)

        # Generate output filename if not provided
        if output_filename is None:
            output_path = generate_output_filename(
                video_path, output_dir, suffix="audio", extension=f".{self.format}"
            )
        else:
            output_path = output_dir / output_filename
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{self.format}")

        logger.info(f"Extracting audio from {video_path} to {output_path}")

        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(output_path),
                    acodec=self.codec,
                    ac=self.channels,
                    ar=self.sample_rate,
                    format=self.format,
                )
                .global_args("-loglevel", "error")
                .global_args("-y")  # Overwrite if exists
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Successfully extracted audio to {output_path}")
            return output_path

        except ffmpeg.Error as e:
            logger.error(f"Error extracting audio: {e.stderr.decode() if e.stderr else str(e)}")
            raise

    def extract_audio_batch(
            self,
            video_paths: list[Union[str, Path]],
            output_dir: Optional[Union[str, Path]] = None,
    ) -> list[Path]:
        """Extract audio from multiple video files.

        Args:
            video_paths: List of paths to video files.
            output_dir: Directory to save the extracted audio. If None, use default audio directory.

        Returns:
            List of paths to the extracted audio files.
        """
        output_dir = ensure_dir(output_dir or AUDIO_DIR)
        audio_paths = []

        for video_path in video_paths:
            try:
                audio_path = self.extract_audio(video_path, output_dir)
                audio_paths.append(audio_path)
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")

        return audio_paths

    @staticmethod
    def check_ffmpeg_installed() -> bool:
        """Check if ffmpeg is installed and available in PATH."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("ffmpeg is not installed or not available in PATH")
            return False