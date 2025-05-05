#!/usr/bin/env python
"""
Script creates a talking avatar based on YouTube video.
Process:
1. Extracting text from the video
2. Summarizing text using Claude API
3. Creating an avatar using D-ID API

Usage: simply run python run_avatar.py
"""

import os
import sys
import subprocess
import argparse
from gtts import gTTS
from src.video_processor import VideoProcessor
import summarize_text
import create_avatar_video
from src.bart_summarizer import Summarizer

# Configuration (constant paths)
TEXT_FILE = 'sample_text.txt'           # File with the original text
SUMMARIZED_TEXT_FILE = 'text_to_speak.txt'  # File with summarized text
OUTPUT_VIDEO = 'generated_avatar_video.mp4'  # Output avatar video
LANGUAGE = 'en'  # Text language (ru - Russian, en - English)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Creating a talking avatar and extracting text from video')
    parser.add_argument('--extract', action='store_true', help='Extract text from video in Wav2Lip/inputs/audios directory')
    parser.add_argument('--avatar', action='store_true', help='Create an avatar based on the text')
    parser.add_argument('--language', type=str, default='en', choices=['en', 'ru'], help='Text language (en - English, ru - Russian)')
    parser.add_argument('--youtube', type=str, default="https://www.youtube.com/watch?v=r4n79ZRepi8", help='YouTube video URL to download')
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn', help='BART model to use for summarization')
    parser.add_argument('--skip-summarize', action='store_true', help='Skip summarization (use existing text in text_to_speak.txt)')
    args = parser.parse_args()
    
    # Set language from arguments
    global LANGUAGE
    LANGUAGE = args.language
    
    # Language mapping for different libraries
    language_map = {
        'en': 'en-US',  # For Google Speech API and Vosk
        'ru': 'ru-RU'   # For Google Speech API and Vosk
    }
    
    # If no flags are specified, enable both modes by default
    if not args.extract and not args.avatar:
        args.extract = True
        args.avatar = True
    
    # Create necessary directories
    os.makedirs('Wav2Lip/inputs', exist_ok=True)
    os.makedirs('Wav2Lip/inputs/audios', exist_ok=True)
    os.makedirs('Wav2Lip/temp', exist_ok=True)
    
    # Step 1: Extract text from video
    if args.extract:
        print("\n=== EXTRACTING TEXT FROM VIDEO ===")
        processor = VideoProcessor(language=language_map[LANGUAGE], youtube_url=args.youtube)
        result = processor.run()
        
        if not result:
            print("Error extracting text from video. Check the video file.")
            if not os.path.exists(TEXT_FILE):
                return

    if args.avatar and not args.skip_summarize:
        print("\n=== SUMMARIZING TEXT USING LOCAL BART MODEL ===")

        # Check if the source text file exists
        if not os.path.isfile(TEXT_FILE):
            print(f"Error: File {TEXT_FILE} not found")
            return

        try:
            # Read source text
            with open(TEXT_FILE, 'r', encoding='utf-8') as file:
                text = file.read()
                if not text.strip():
                    print(f"Error: File {TEXT_FILE} is empty")
                    return
                print(f"Read text of length {len(text)} characters")

            # Initialize BART summarizer
            print(f"Initializing BART summarizer with model: {args.model}")
            summarizer = Summarizer(model_name=args.model)

            # Summarize text
            print("Summarizing text... This may take a while for longer texts")
            summary = summarizer.summarize(text)

            # Save summarized text
            with open(SUMMARIZED_TEXT_FILE, 'w', encoding='utf-8') as file:
                file.write(summary)
            print(f"Summary saved to file: {SUMMARIZED_TEXT_FILE}")
            print(f"Summary length: {len(summary)} characters")

        except Exception as e:
            print(f"Error during summarization: {e}")
            return

        # Check that the summarized text file was created
        if not os.path.exists(SUMMARIZED_TEXT_FILE):
            print(f"Error: Failed to create summarized text in {SUMMARIZED_TEXT_FILE}")
            return
    
    # Step 3: Create avatar using D-ID API
    if args.avatar:
        print("\n=== CREATING AVATAR USING D-ID API ===")
        # Check if the summarized text file exists
        if not os.path.exists(SUMMARIZED_TEXT_FILE):
            print(f"Error: File {SUMMARIZED_TEXT_FILE} not found")
            return
            
        # Call main() function from AKOOL_avatar module
        create_avatar_video.main()
        
        print(f"\nDone! Avatar video saved to {OUTPUT_VIDEO}")
        print(f"Open the file in a video player to view it")

if __name__ == "__main__":
    main()