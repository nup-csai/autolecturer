#!/usr/bin/env python
"""
Script for processing text from sample_text.txt using Claude API,
extracting main ideas and saving a concise summary to text_to_speak.txt.
"""

import os
import argparse
import requests
import json
from dotenv import load_dotenv


def summarize_with_claude(text, api_key, max_length=300):
    """
    Processes text using Claude API (Messages API).
    
    Args:
        text: Source text to process
        api_key: Claude API key
        max_length: Maximum response length
        
    Returns:
        Processed text
    """

    if not api_key:
        raise ValueError("Claude API key must be provided")
        
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        # Create request for Claude using new Messages API
        system_prompt = "You are an expert in text processing and summarization."
        
        user_message = (
            "Read the transcript of the lecture and do the following\n\n"
            "Your task is to summarize the given text without losing the structure and sense of it. "
            "Make sure to capture all the key points and maintain the overall logical flow."
            f"Text for the processing:\n{text}"
        )
        
        data = {
            "model": "claude-3-haiku-20240307",
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "max_tokens": max_length,
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["content"][0]["text"]
        else:
            print(f"Claude API Error: {response.status_code}, {response.text}")
            return None
            
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Text processing using Claude API')
    parser.add_argument('--input', type=str, default='sample_text.txt', help='Path to source text file')
    parser.add_argument('--output', type=str, default='text_to_speak.txt', help='Path to file for saving results')
    parser.add_argument('--api-key', type=str, help='Claude API key')
    parser.add_argument('--max-length', type=int, default=500, help='Maximum response length')
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv('ANTHROPIC_API_KEY')

    # Read source text
    try:
        with open(args.input, 'r', encoding='utf-8') as file:
            text = file.read()
            if not text.strip():
                print(f"Error: File {args.input} is empty")
                return
            print(f"Read text of length {len(text)} characters")
    except Exception as e:
        print(f"Error reading file {args.input}: {e}")
        return
    
    # Process text
    print("Sending request to Claude API...")
    summary = summarize_with_claude(text, api_key, args.max_length)
    
    if not summary:
        print("Failed to get response from Claude API")
        return
    
    # Save result
    try:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(summary)
        print(f"Result saved to file: {args.output}")
        print(f"Text length: {len(summary)} characters")
        preview = summary[:100] + "..." if len(summary) > 100 else summary
        print(f"Preview: {preview}")
    except Exception as e:
        print(f"Error saving file {args.output}: {e}")
        return

if __name__ == "__main__":
    main()