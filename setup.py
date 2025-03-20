from setuptools import setup, find_packages

setup(
    name="autolecturer",
    version="0.1.0",
    description="A system for processing and summarizing video lectures using local processing",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "ffmpeg-python>=0.2.0",
        "pydub>=0.25.1",
        "faster-whisper>=0.9.0",
        "vosk>=0.3.45",
        "spacy>=3.6.0",
        "nltk>=3.8.1",
        "regex>=2023.0.0",
        "symspellpy>=6.7.7",
        "moviepy>=1.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "pylint>=2.17.5",
        ],
        "advanced": [
            "transformers>=4.30.0",
            "llama-cpp-python>=0.2.0",
            "gpt4all>=2.0.0",
        ]
    },
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "autolecturer=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
)