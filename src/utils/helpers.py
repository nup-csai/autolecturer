"""Utility functions for the lecture summarizer."""
import os
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def timing_decorator(func):
    """Decorator to measure the execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result

    return wrapper


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure a directory exists and return its Path object."""
    dir_path = Path(directory)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Save data as a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, filepath: Union[str, Path]) -> None:
    """Save text to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def load_text(filepath: Union[str, Path]) -> str:
    """Load text from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def generate_output_filename(
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        suffix: str = "",
        extension: Optional[str] = None
) -> Path:
    """Generate an output filename based on the input filename."""
    input_path = Path(input_path)
    output_dir = ensure_dir(output_dir)

    # Use the original extension if not specified
    if extension is None:
        extension = input_path.suffix

    # Add the suffix to the stem of the filename
    stem = input_path.stem
    if suffix:
        stem = f"{stem}_{suffix}"

    # Ensure the extension starts with a dot
    if not extension.startswith("."):
        extension = f".{extension}"

    return output_dir / f"{stem}{extension}"