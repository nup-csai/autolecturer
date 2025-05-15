import os
import numpy as np
import torch
import subprocess
import sys
from transformers import pipeline, AutoTokenizer, AutoModel
# Try to import StableDiffusionPipeline, but handle import errors gracefully
try:
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers package not available or incompatible. Image generation will be disabled.")
import whisper
# Add ANTIALIAS constant to PIL.Image for compatibility with newer Pillow versions
from PIL import Image
if not hasattr(Image, 'ANTIALIAS'):
    # In newer versions of Pillow, ANTIALIAS was replaced with LANCZOS
    Image.ANTIALIAS = Image.LANCZOS
import moviepy.editor as mp
from gtts import gTTS
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import re
# Import Coqui TTS for local TTS
try:
    from TTS.api import TTS
    COQUI_TTS_AVAILABLE = True
except ImportError:
    COQUI_TTS_AVAILABLE = False

# Note: ImageMagick is not required for this project
# MoviePy can use ImageMagick for some features like TextClip and write_gif,
# but we're not using those features in this project.
try:
    from moviepy.config import get_setting
    # Just check if ImageMagick is available, but don't try to install it
    imagemagick_binary = get_setting("IMAGEMAGICK_BINARY")
    IMAGEMAGICK_AVAILABLE = imagemagick_binary != "unset" and imagemagick_binary is not None and os.path.exists(imagemagick_binary)
    if not IMAGEMAGICK_AVAILABLE:
        print("ImageMagick not found, but it's not required for this project.")
except Exception:
    IMAGEMAGICK_AVAILABLE = False

# Function to get embeddings for text using a sentence transformer
def get_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get embeddings for a list of texts using a sentence transformer.

    Args:
        texts (list): List of text strings
        model_name (str): Name of the model to use

    Returns:
        numpy.ndarray: Array of embeddings
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling to get sentence embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)

    return np.array(embeddings)

# Function to transcribe video with timestamps
def transcribe_with_timestamps(video_path, model_name="base"):
    """
    Transcribe video with timestamps for each segment.

    Args:
        video_path (str): Path to the video file
        model_name (str): Whisper model to use

    Returns:
        list: List of dictionaries with text and timestamps
    """
    print(f"Loading Whisper model for timestamp transcription: {model_name}")
    model = whisper.load_model(model_name)

    # Extract audio from video
    audio_path = tempfile.mktemp(suffix=".mp3")
    video_clip = mp.VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # Transcribe with timestamps
    print("Transcribing with timestamps...")
    result = model.transcribe(audio_path)

    # Clean up
    os.remove(audio_path)
    video_clip.close()

    # Extract segments with timestamps
    segments = []
    for segment in result["segments"]:
        segments.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"]
        })

    return segments

# Function to find matching video segments for each line in the summary
def find_matching_segments(summary_lines, video_segments):
    """
    Find matching video segments for each line in the summary using semantic similarity.

    Args:
        summary_lines (list): List of text lines from the summary
        video_segments (list): List of dictionaries with text and timestamps

    Returns:
        list: List of dictionaries with summary text and matching video segments
    """
    print("Finding matching video segments for each summary line...")

    # Get embeddings for summary lines and video segments
    summary_embeddings = get_embeddings(summary_lines)
    segment_texts = [segment["text"] for segment in video_segments]
    segment_embeddings = get_embeddings(segment_texts)

    matches = []
    for i, summary_line in enumerate(summary_lines):
        # Calculate similarity between this summary line and all video segments
        similarities = cosine_similarity([summary_embeddings[i]], segment_embeddings)[0]

        # Get top 3 matching segments
        top_indices = np.argsort(similarities)[-3:][::-1]

        matching_segments = [video_segments[idx] for idx in top_indices if similarities[idx] > 0.3]

        matches.append({
            "summary_text": summary_line,
            "matching_segments": matching_segments
        })

    return matches

# Function to generate images using Stable Diffusion
def generate_images(prompts, output_dir="generated_images"):
    """
    Generate images using Stable Diffusion based on text prompts.
    (Commented out to speed up testing)

    Args:
        prompts (list): List of text prompts
        output_dir (str): Directory to save generated images

    Returns:
        list: List of paths to generated images (empty list when commented out)
    """
    # Commented out to speed up testing
    print("Image generation is commented out to speed up testing")
    return []


# Function to create text-to-speech audio
def create_tts(text, output_path="narration.mp3", use_local_tts=True, model_name="tts_models/en/ljspeech/tacotron2-DDC"):
    """
    Create text-to-speech audio from text using Coqui TTS (local TTS) or gTTS (online).

    Args:
        text (str): Text to convert to speech
        output_path (str): Path to save the audio file
        use_local_tts (bool): Whether to try using local TTS first
        model_name (str): Coqui TTS model to use

    Returns:
        str: Path to the audio file
    """
    print("Creating text-to-speech narration...")

    # Try to use Coqui TTS if available and requested
    if use_local_tts and COQUI_TTS_AVAILABLE:
        try:
            print(f"Using Coqui TTS engine for narration with model: {model_name}")

            # Initialize the TTS engine with the selected model
            tts = TTS(model_name=model_name)

            # Generate temporary WAV file
            temp_wav = tempfile.mktemp(suffix=".wav")

            # Generate speech
            tts.tts_to_file(text=text, file_path=temp_wav, speed=1.0)

            # Convert WAV to MP3 using moviepy if output is not WAV
            if not output_path.endswith('.wav'):
                audio_clip = mp.AudioFileClip(temp_wav)
                audio_clip.write_audiofile(output_path, verbose=False, logger=None)
                audio_clip.close()

                # Clean up temporary file
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            else:
                # If output is WAV, just copy the file
                import shutil
                shutil.copy(temp_wav, output_path)
                os.remove(temp_wav)

            print(f"Successfully created narration using Coqui TTS: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error using Coqui TTS: {e}. Falling back to gTTS.")

    # Fallback to gTTS (requires internet)
    print("Using gTTS for narration (requires internet)")
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_path)
    return output_path

# Main function to create the trailer
def create_trailer(video_path, summary_path, output_path="trailer.mp4"):
    """
    Create a trailer from a video and summary text.

    Args:
        video_path (str): Path to the video file
        summary_path (str): Path to the summary text file
        output_path (str): Path to save the trailer

    Returns:
        str: Path to the created trailer
    """
    try:
        # Load the summary text
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_text = f.read()

        # Split summary into sentences for better matching
        summary_lines = re.split(r'(?<=[.!?])\s+', summary_text)
        summary_lines = [line.strip() for line in summary_lines if line.strip()]

        # Transcribe video with timestamps
        video_segments = transcribe_with_timestamps(video_path)

        # Find matching video segments for each summary line
        matches = find_matching_segments(summary_lines, video_segments)

        # Create TTS narration
        narration_path = create_tts(summary_text)
        narration_audio = mp.AudioFileClip(narration_path)

        # Generate images for a few key sentences (select 3 evenly spaced sentences)
        # Commented out to speed up testing
        # num_images = 3
        # image_indices = [i * len(summary_lines) // (num_images + 1) for i in range(1, num_images + 1)]
        # image_prompts = [summary_lines[i] for i in image_indices]
        image_paths = generate_images([])

        # Create video clips
        clips = []
        current_time = 0

        # Get narration duration to limit trailer length
        narration_duration = narration_audio.duration
        print(f"Narration duration: {narration_duration:.2f} seconds")

        # Target trailer duration (in seconds) - aim for the same as narration
        target_duration = narration_duration
        print(f"Target trailer duration: {target_duration:.2f} seconds")

        # Track total duration of selected clips
        total_clip_duration = 0

        # Calculate average time per match to distribute evenly
        avg_time_per_match = target_duration / len(matches) if matches else 0
        print(f"Average time per segment: {avg_time_per_match:.2f} seconds")

        for match in matches:
            if match["matching_segments"] and total_clip_duration < target_duration:
                # Get the best matching segment
                segment = match["matching_segments"][0]

                # Create clip from the video
                start_time = max(0, segment["start"] - 0.5)  # Add a small buffer
                end_time = min(segment["end"] + 0.5, mp.VideoFileClip(video_path).duration)

                # Calculate segment duration
                segment_duration = end_time - start_time

                # Limit segment duration to not exceed average time per match by too much
                # This ensures more even distribution of time across segments
                if segment_duration > avg_time_per_match * 1.5:
                    # If segment is too long, trim it to a reasonable length
                    middle_point = (start_time + end_time) / 2
                    half_duration = min(avg_time_per_match * 0.75, segment_duration / 2)
                    start_time = middle_point - half_duration
                    end_time = middle_point + half_duration
                    segment_duration = end_time - start_time

                # Check if adding this clip would exceed our target duration
                if total_clip_duration + segment_duration > target_duration:
                    # If it would exceed, trim the clip to fit
                    available_time = target_duration - total_clip_duration
                    if available_time >= 1.0:  # Only add if we have at least 1 second available
                        end_time = start_time + available_time
                        segment_duration = available_time
                    else:
                        # Skip this clip if we don't have enough time left
                        continue

                video_clip = (mp.VideoFileClip(video_path)
                             .subclip(start_time, end_time)
                             .without_audio())

                # Update total duration
                total_clip_duration += segment_duration
                print(f"Added clip: {segment_duration:.2f}s, Total: {total_clip_duration:.2f}s / {target_duration:.2f}s")

                # Add clip to the list
                clips.append(video_clip)

        # Add image slides (2 seconds each)
        # Commented out to speed up testing
        # for image_path in image_paths:
        #     image_clip = mp.ImageClip(image_path, duration=2)
        #     clips.append(image_clip)

        # Concatenate all clips
        final_clip = mp.concatenate_videoclips(clips, method="compose")

        # Add narration audio
        final_clip = final_clip.set_audio(narration_audio)


        # Resize to a standard resolution if needed
        final_clip = final_clip.resize(height=720)

        # Write the final trailer
        print(f"Writing trailer to {output_path}...")
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)

        # Clean up
        for clip in clips:
            clip.close()
        narration_audio.close()

        print(f"Trailer created successfully at {output_path}")
        return output_path

    except Exception as e:
        print(f"Error creating trailer: {e}")
        raise
