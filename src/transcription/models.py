"""Module for managing transcription models."""
import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from src.config import STT_SETTINGS
from src.utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class ModelManager:
    """Manager for downloading and setting up transcription models."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the model manager with settings."""
        self.settings = settings or STT_SETTINGS
        self.model_type = self.settings.get("model_type", "faster-whisper")
        self.model_size = self.settings.get("model_size", "base")
        self.models_dir = Path.home() / ".cache" / "lecture_summarizer" / "models"
        ensure_dir(self.models_dir)

    def get_model_path(self) -> Path:
        """Get the path to the model directory."""
        if self.model_type == "faster-whisper":
            return Path.home() / ".cache" / "faster_whisper"
        elif self.model_type == "vosk":
            return self.models_dir / "vosk"
        else:
            return self.models_dir / self.model_type

    def download_model(self, force: bool = False) -> Path:
        """Download the model if it doesn't exist."""
        model_path = self.get_model_path()

        if model_path.exists() and not force:
            logger.info(f"Model already exists at {model_path}")
            return model_path

        ensure_dir(model_path)

        if self.model_type == "faster-whisper":
            logger.info("Faster Whisper models are downloaded automatically when first used")
            return model_path

        elif self.model_type == "vosk":
            # Vosk models need manual download
            logger.info("Downloading Vosk model...")
            model_url = self._get_vosk_model_url()

            # Download the model using Python requests
            try:
                import requests
                import tqdm
                import zipfile

                # Download the model
                response = requests.get(model_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                # Get the filename from the URL
                filename = os.path.basename(model_url)
                zip_path = self.models_dir / filename

                # Download with progress bar
                with open(zip_path, 'wb') as f, tqdm.tqdm(
                        desc=filename,
                        total=total_size,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = f.write(data)
                        bar.update(size)

                # Extract the model
                logger.info(f"Extracting model to {model_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.models_dir)

                # Rename the extracted directory if needed
                extracted_dir = self.models_dir / zip_path.stem.split("-")[0]
                if extracted_dir.exists() and extracted_dir != model_path:
                    os.rename(extracted_dir, model_path)

                # Remove the zip file
                os.remove(zip_path)

                logger.info(f"Successfully downloaded and extracted Vosk model to {model_path}")
                return model_path

            except Exception as e:
                logger.error(f"Error downloading Vosk model: {str(e)}")
                raise

        else:
            logger.error(f"Downloading for model type {self.model_type} not implemented")
            raise NotImplementedError(f"Downloading for model type {self.model_type} not implemented")

    def _get_vosk_model_url(self) -> str:
        """Get the URL for downloading the Vosk model."""
        # Map of available Vosk models
        models = {
            "small": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
            "base": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
            "large": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.42.zip",
        }

        model_size = self.model_size
        if model_size not in models:
            logger.warning(f"Vosk model size {model_size} not available, using 'base' instead")
            model_size = "base"

        return models[model_size]