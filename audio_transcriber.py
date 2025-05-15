import os
import whisper
import moviepy.editor as mp

def extract_audio(video_path, audio_path="temp_audio.mp3"):
    """
    Extract audio from a video file.
    
    Args:
        video_path (str): Path to the video file
        audio_path (str): Path to save the extracted audio
        
    Returns:
        str: Path to the extracted audio file
    """
    try:
        video_clip = mp.VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise

def transcribe_audio(video_path, output_text_file="sample_text.txt", model_name="base"):
    """
    Transcribe audio from a video file using Whisper.
    
    Args:
        video_path (str): Path to the video file
        output_text_file (str): Path to save the transcribed text
        model_name (str): Whisper model to use (tiny, base, small, medium, large)
        
    Returns:
        str: Path to the transcribed text file
    """
    try:
        print("Extracting audio from video...")
        audio_path = extract_audio(video_path)
        
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        print("Transcribing audio...")
        result = model.transcribe(audio_path)
        
        # Write transcription to file
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        print(f"Transcription completed and saved to {output_text_file}")
        return output_text_file
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise