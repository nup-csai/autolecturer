# YouTube Trailer Generator

This project automatically creates a trailer from a YouTube video. It downloads a video, transcribes the audio, summarizes the content, and creates a trailer by combining relevant video clips, generated images, and text-to-speech narration.

## How It Works

1. **Download**: Downloads a YouTube video using the provided URL
2. **Transcription**: Extracts and transcribes the audio from the video
3. **Summarization**: Summarizes the transcribed text to create a concise script for the trailer
4. **Trailer Creation**: 
   - Matches phrases from the summary with segments in the original video
   - Generates images related to key points in the summary
   - Creates a text-to-speech narration of the summary
   - Combines video clips, images, and narration into a final trailer

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script with optional parameters:

```
python run_avatar.py [--youtube-url URL]
```

Options:
- `--youtube-url URL`: Specify a custom YouTube URL (default is a preset educational video)

The script will:
1. Download the YouTube video
2. Transcribe the audio to `sample_text.txt`
3. Summarize the text and save it to `text_to_speak.txt`
4. Create a trailer by combining video clips, generated images, and narration
5. Automatically add subtitles to the trailer (if the subtitles module is available)
6. Save the final trailer as `trailer.mp4` and a version with subtitles as `trailer_with_subtitles.mp4`

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster image generation and video processing)
- Internet connection for downloading models and YouTube videos
- ffmpeg (required for video processing and adding subtitles)

## Project Structure

- `run_avatar.py`: Main script that orchestrates the entire process
- `video_processor.py`: Handles downloading YouTube videos
- `audio_transcriber.py`: Transcribes audio to text using Whisper
- `bart_summarizer.py`: Summarizes text using BART model
- `video_clipper.py`: Creates the trailer by combining video clips, images, and narration
- `subtitles.py`: Adds subtitles to videos from text files
- `example_subtitles.py`: Example script demonstrating how to use the subtitles module

## Models Used

- **Whisper**: For audio transcription
- **BART**: For text summarization
- **Sentence Transformers**: For semantic matching between summary and video segments
- **Stable Diffusion**: For generating images based on text prompts
- **Coqui TTS**: Primary TTS engine for creating high-quality narration audio locally
- **gTTS (Google Text-to-Speech)**: Fallback TTS service if Coqui TTS is not available (requires internet)

## Text-to-Speech

The project uses a high-quality local text-to-speech solution:

1. **Coqui TTS**: A deep learning toolkit for Text-to-Speech that provides:
   - High-quality, natural-sounding speech synthesis
   - Multiple pre-trained models (default: "tts_models/en/ljspeech/tacotron2-DDC")
   - Completely local operation with no API key or internet required
   - Adjustable speech parameters like speed

   The first time you run the system, it will download the selected TTS model automatically.

2. If Coqui TTS is not available or encounters an error, the system will automatically fall back to using Google Text-to-Speech (gTTS), which requires an internet connection but no API key.


## Notes

- The first run may take longer as it downloads the required models
- Processing time depends on the length of the video and your hardware
- For best results, use videos with clear speech and good audio quality

## Subtitles

The project includes a dedicated subtitles module that can:

1. **Add subtitles to any video**: Convert plain text to timed subtitles and overlay them on videos
2. **Create SRT files**: Generate standard SRT subtitle files for use with other video players
3. **Customize appearance**: Adjust font, size, color, and background opacity

### Using Subtitles with the Main Script

Add the `--with-subtitles` flag when running the main script:

```
python run_avatar.py --with-subtitles
```

### Using the Subtitles Module Separately

You can use the subtitles module separately with any video and text file:

```
python subtitles.py video_file.mp4 text_file.txt --output output_video.mp4
```

Or use the example script:

```
python example_subtitles.py
```

This will add subtitles to the trailer using the summarized text from `text_to_speak.txt`.
