from setuptools import setup, find_packages

setup(
    name="lecture-summarizer",
    version="0.1.0",
    description="A system for summarizing video lectures using local processing",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "ffmpeg-python",
        "pydub",
        "faster-whisper",
        "vosk",
        "spacy",
        "nltk",
        "regex",
        "symspellpy",
        "moviepy",
    ],
    python_requires=">=3.11",
)