# AutoLecturer Pipeline

## 0. Set Up the Environment and Prepare Data

### Tools and Dependencies
#### Programming Language:
- Use Python 3.11.

#### IDE:
- Install and configure PyCharm.

#### Required Libraries:
- Install `ffmpeg` for audio extraction.
- Use `nltk`, `spaCy`, `regex` for text processing.
- Deploy `llama.cpp`, `GPT4All` for local LLMs.
- Utilize `MoviePy`, `FFmpeg` for video assembly.
- Implement `Coqui TTS` for speech synthesis.
- Apply `SadTalker`, `Wav2Lip` for deepfake lecturer.
- (Optional) Use `Selenium` for browser automation.

#### Hardware Considerations:
- Ensure CPU/GPU support for optimized execution (NVIDIA CUDA, Metal for Mac).

---

## 1. Extract Audio from the Video (Audio Transcription)

### 1.1 Extracting and Processing Text from Video
- Use `ffmpeg` for high-speed, high-quality audio extraction.
- Convert video to WAV format (16kHz, mono, PCM S16LE) for compatibility with STT models.
- (Optional) Implement batch-processing to handle multiple videos automatically.
- (TODO: возможно, улучшить звук от шума).

### 1.2 Speech-to-Text (STT) Processing (Fully Local)
#### Best Local STT Models:
- **Faster-Whisper** (Whisper Large-v2, local execution) – Best accuracy for general English transcription.
- **NVIDIA NeMo** (Conformer-CTC / Hybrid Transformer-Transducer) – GPU-optimized speech-to-text.
- **Vosk** (offline STT engine, lightweight, supports weak hardware) – Suitable for low-resource environments.

#### STT Process:
- Convert speech into highly accurate transcripts with timestamps.
- Ensure full local execution (CPU/GPU) with no cloud dependencies.
- Enable batch processing for multiple lectures.

### 1.3 Text Cleaning & Normalization (Local NLP Processing)
- Remove filler words, repetitions, and irrelevant noise.
- Normalize abbreviations and correct spelling mistakes.
- Apply lemmatization and casing correction for readability.
- Use local NLP libraries (`spaCy`, `nltk`) for text processing.

#### Tools:
- **spaCy** – Offline NLP for tokenization and lemmatization.
- **SymSpell** – Local spell checking (no external API required).
- **Regex-based processing** for unwanted characters and formatting issues.

### 1.4 Chunking the Transcript (Efficient Processing for Local LLMs)
#### Chunking Strategies:
- Sentence-based segmentation (`nltk.sent_tokenize()`).
- Fixed-token chunking (e.g., 512 tokens per chunk for LLM compatibility).
- Timestamp-based segmentation (maintaining coherence with the lecture structure).

#### Why Chunking?
- Local LLMs have limited context windows, and chunking improves performance.
- Preserves logical consistency when processing large transcripts.

---

## 2. Post-Processing with Local LLM Models (Summarization, Key Topic Extraction, Refinement)

### 2.1 Best Local LLMs (No External APIs)
- **Mistral 7B** (GGUF, llama.cpp) – High-performance summarization.
- **Llama 2** (Meta, GGUF format) – Balanced accuracy and efficiency.
- **Gemma 7B** (Google, llama.cpp) – Good for structured text generation.
- **GPT4All** (Multiple GGUF models) – Lightweight local processing.
- **RWKV** (RNN alternative to transformers) – Minimal resource usage.

### 2.2 Post-Processing Tasks
- **Summarization** – Generate concise, high-quality lecture summaries.
- **Topic Extraction** – Identify key discussion points.
- **Text Refinement** – Improve grammar, punctuation, and structure.
- **Question Generation** – Create quiz questions based on lecture content.

#### Execution Approach:
- Run LLM models fully offline using `llama.cpp` or `GPT4All`.
- Optimize performance with CPU/GPU acceleration (Metal for Mac, CUDA for NVIDIA GPUs).
- Process chunks sequentially to maintain coherence.

---

## 3. Avatar Generation (Deepfake Lecturer)

### 3.1 Choosing a Deepfake Method
For generating a talking-head lecturer:
- **DeepFaceLive + FaceSwap** – Real-time deepface with a reference model.
- **SadTalker** – Motion-driven talking head generation from an image + audio.
- **Wav2Lip** – Lip-syncing an existing video with generated speech.

### 3.2 Generating the Talking-Head Video
- Choose a lecturer image (or generate one using AI tools).
- Generate a talking-head video with `SadTalker`.
- The output is a video of the lecturer speaking with realistic facial movements.

---

## 4. Text-to-Speech (TTS) for Narration

### 4.1 TTS Options
- **Coqui TTS** – Open-source, supports many voices.
- **Mozilla TTS** – Good offline voice models.
- **XTTS / Bark** – More expressive voices.
- **Voice Cloning** – If samples are available.

### 4.2 Generating the TTS Narration
- Install `Coqui TTS`.
- List available voices.
- Generate narration.
- **Final Output:** A realistic audio file of the lecturer speaking each summarized thought.

---

## 5. Video Assembly with MoviePy

### 5.1 Finding a Classroom Image
- High-resolution empty classroom background.
- A method to overlay images dynamically on the blackboard.

#### Generating or Finding an Image:
- Use real stock images.
- Generate an AI image (e.g., Stable Diffusion).
- Ensure the blackboard is empty for dynamic overlays.

### 5.2 Video Assembly Steps
#### Collect Assets:
- Summaries (with generated audio).
- Slide images.
- (Optional) Talking head clips.

#### Automate Video Editing:
- Use Python's `MoviePy` or `FFmpeg`.
- **Subtitles (Optional):**
  - Generate `.srt` or `.vtt` subtitles.
  - Save summary text with timestamps matching audio clips.
- **Post-Processing:**
  - Add intro/outro.
  - Overlay text, watermarks.
  - Normalize audio levels.

- **TODO:** Добавить возможность создания видео по ссылке на видеолекцию.

---

## 7. Future Extensions

### 7.1 Interactive Web UI
- Build a local interface (`Flask`, `Django`):
  - Upload lectures/slides.
  - Generate and edit summaries.
  - Automate final video creation.

### 7.2 Q&A or Chatbot
- Store transcripts in a vector DB (`FAISS`, `Milvus`).
- Let a local LLM answer questions about the lecture.

### 7.3 Multi-Lingual Support
- Use local translation models for bilingual versions.

### 7.4 LMS Integration
- Automate quiz generation.
- Provide knowledge checks.

### 7.5 Advanced Slide Recognition
- Use `OpenCV` / `YOLO` for precise slide change detection.

---

## Usage Instructions

### Basic Usage
1. Place your lecture video in the `data/input` directory (name it `lecture.mp4` or use the `--video` option)
2. Run the pipeline:
```
python main.py
```

### Command Line Options
- `--video PATH` - Specify the path to the input video file
- `--output DIR` - Specify the output directory for processed files
- `--skip-existing` - Skip processing steps if output files already exist
- `--no-extract-clips` - Skip extracting video clips
- `--no-combine-clips` - Skip combining clips into a final video

### Testing Individual Components
You can test individual components of the pipeline using the test scripts:
```
python tests/test_audio_extraction.py  # Test audio extraction
python tests/test_transcription.py     # Test speech-to-text
python tests/test_text_cleaning.py     # Test text cleaning
python tests/test_text_chunking.py     # Test text chunking
python tests/test_video_clipper.py     # Test video clip extraction
```

## Pipeline Workflow

### Full Pipeline
1. **Audio Extraction**: Extract high-quality audio from the input video
2. **Speech-to-Text**: Convert audio to text with timestamps using local STT models
3. **Text Cleaning**: Clean and normalize the transcribed text
4. **Text Chunking**: Divide text into logical paragraphs with main ideas
5. **Video Clipping**: Extract video clips based on paragraph timestamps
6. **Video Assembly**: Combine clips into a summarized video

### Supported Features
- Advanced text cleaning and normalization
- Intelligent paragraph division based on content
- Preservation of timestamp information between text and video
- Automatic extraction of video clips for each logical paragraph
- Combining clips into a summarized video under 2 minutes

## Final Thoughts
### Advantages of Local Processing
- Ensures data privacy (no external API calls).
- Requires sufficient CPU/GPU resources.
- Offers fully automated lecture summarization.

### Typical Workflow:
Audio Extraction → Local STT → Text Cleaning → Logical Paragraph Creation → Video Clip Extraction → Video Assembly → Final Output.