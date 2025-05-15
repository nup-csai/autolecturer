import os
from video_processor import download_youtube_video
from audio_transcriber import transcribe_audio
from bart_summarizer import summarize_text
from video_clipper import create_trailer

def main():
    # YouTube URL to process
    youtube_url = "https://www.youtube.com/watch?v=cHLaI6YN48I&list=PLBwY-w1dQOq5bVKA67x3RjpJASD9ONjoM&index=3"

    print("Starting the trailer creation process...")

    # Step 1: Download the YouTube video
    print("Downloading YouTube video...")
    video_path = download_youtube_video(youtube_url)

    # Step 2: Transcribe the audio to text
    print("Transcribing audio to text...")
    transcribe_audio(video_path, "sample_text.txt")

    # Step 3: Summarize the text
    print("Summarizing text...")
    summarize_text("sample_text.txt", "text_to_speak.txt")

    # Step 4: Create the trailer
    print("Creating trailer...")
    trailer_path = create_trailer(video_path, "text_to_speak.txt")

    print(f"Trailer created successfully! Saved at: {trailer_path}")

if __name__ == "__main__":
    main()
