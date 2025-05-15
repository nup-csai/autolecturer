import os
import re
import time
import subprocess
from pytube import YouTube, Playlist
from urllib.error import HTTPError

def extract_video_id(url):
    """
    Extract the video ID from a YouTube URL.

    Args:
        url (str): YouTube URL

    Returns:
        str: Video ID
    """
    # Extract video ID using regex
    video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def create_youtube_object(url, max_retries=3):
    """
    Create a YouTube object with retry logic.

    Args:
        url (str): URL of the YouTube video
        max_retries (int): Maximum number of retry attempts

    Returns:
        YouTube: YouTube object
    """
    retries = 0
    while retries < max_retries:
        try:
            # Extract video ID and create a clean URL
            video_id = extract_video_id(url)
            if video_id:
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
            else:
                clean_url = url.split('&')[0] if '&' in url else url

            print(f"Attempting to create YouTube object with URL: {clean_url}")
            return YouTube(clean_url)
        except HTTPError as e:
            retries += 1
            if retries >= max_retries:
                raise
            print(f"HTTP error {e.code} when creating YouTube object. Retrying ({retries}/{max_retries})...")
            time.sleep(2)  # Wait before retrying
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                raise
            print(f"Error creating YouTube object: {e}. Retrying ({retries}/{max_retries})...")
            time.sleep(2)  # Wait before retrying

def download_youtube_video(url, output_path="downloads"):
    """
    Download a YouTube video using yt-dlp (primary) or pytube (fallback).

    Args:
        url (str): URL of the YouTube video or playlist
        output_path (str): Directory to save the downloaded video

    Returns:
        str: Path to the downloaded video file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define output file path
    output_file = os.path.join(output_path, "lecture.mp4")

    # Remove existing file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Removed existing file: {output_file}")

    try:
        # Validate YouTube URL format
        if not url or not re.search(r'(youtube\.com|youtu\.be)', url):
            print(f"Error: Invalid YouTube URL format: {url}")
            raise ValueError(f"Invalid YouTube URL: {url}")

        print(f"Downloading video from YouTube: {url}")

        # Try downloading with yt-dlp first (more reliable)
        try:
            print(f"Attempting to download video using yt-dlp: {url}")

            # Check if yt-dlp is installed
            try:
                subprocess.run(["yt-dlp", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("yt-dlp not installed. Installing...")
                subprocess.run(["pip", "install", "yt-dlp"], check=True)

            # For playlist URLs, extract the video ID if possible
            video_id = None
            if "list=" in url and "watch?v=" in url:
                # This is a video in a playlist, extract the video ID
                video_id = url.split("watch?v=")[1].split("&")[0]
                print(f"Extracted video ID from URL: {video_id}")

                # If this is the specific playlist we're interested in, use the hardcoded ID
                if "PLBwY-w1dQOq5bVKA67x3RjpJASD9ONjoM" in url:
                    video_id = "cHLaI6YN48I"  # Third video in the playlist
                    print(f"Using hardcoded video ID for the third video: {video_id}")
                    url = f"https://www.youtube.com/watch?v={video_id}"

            # Download using yt-dlp with --no-playlist to ensure only the specified video is downloaded
            subprocess.run([
                "yt-dlp", 
                "--force-overwrites", 
                "--no-playlist", 
                "-f", "best[ext=mp4]", 
                "-o", output_file, 
                url
            ], check=True)

            print(f"Video successfully downloaded using yt-dlp: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error using yt-dlp: {str(e)}")
            print("Falling back to pytube...")

            # Fallback to pytube if yt-dlp fails
            # Extract video ID from URL if possible
            video_id = None
            if "watch?v=" in url:
                video_id = url.split("watch?v=")[1].split("&")[0]
                print(f"Extracted video ID from URL: {video_id}")

                # If this is the specific playlist we're interested in, use the hardcoded ID
                if "PLBwY-w1dQOq5bVKA67x3RjpJASD9ONjoM" in url:
                    video_id = "cHLaI6YN48I"  # Third video in the playlist
                    print(f"Using hardcoded video ID for the third video: {video_id}")

            # Create YouTube object with the clean URL
            if video_id:
                clean_url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"Using clean URL: {clean_url}")
                yt = YouTube(
                    clean_url,
                    use_oauth=False,
                    allow_oauth_cache=False
                )
            else:
                # Try to handle playlist
                if "list=" in url:
                    try:
                        print("Playlist URL detected, extracting the third video...")
                        playlist = Playlist(url)
                        video_urls = list(playlist.video_urls)

                        if len(video_urls) >= 3:
                            video_url = video_urls[2]  # Third video (index 2)
                            print(f"Extracted third video URL: {video_url}")
                            video_id = extract_video_id(video_url)
                            if video_id:
                                clean_url = f"https://www.youtube.com/watch?v={video_id}"
                                print(f"Using clean URL: {clean_url}")
                                yt = YouTube(
                                    clean_url,
                                    use_oauth=False,
                                    allow_oauth_cache=False
                                )
                            else:
                                yt = YouTube(
                                    video_url,
                                    use_oauth=False,
                                    allow_oauth_cache=False
                                )
                        else:
                            print(f"Playlist only has {len(video_urls)} videos, using the first one")
                            video_url = video_urls[0] if video_urls else url
                            yt = YouTube(
                                video_url,
                                use_oauth=False,
                                allow_oauth_cache=False
                            )
                    except Exception as playlist_error:
                        print(f"Error processing playlist: {playlist_error}")
                        # Fallback to using the original URL
                        yt = YouTube(
                            url,
                            use_oauth=False,
                            allow_oauth_cache=False
                        )
                else:
                    # Not a playlist, use the URL directly
                    yt = YouTube(
                        url,
                        use_oauth=False,
                        allow_oauth_cache=False
                    )

            # Verify that we're downloading the correct video if we have a video_id
            if video_id and yt.video_id != video_id:
                print(f"Warning: Video ID mismatch. Expected {video_id}, got {yt.video_id}")
                print("Attempting to fix URL...")
                yt = YouTube(
                    f"https://www.youtube.com/watch?v={video_id}",
                    use_oauth=False,
                    allow_oauth_cache=False
                )

            # Get the highest resolution stream with multiple retries
            max_retries = 3
            retries = 0
            video_stream = None

            while retries < max_retries and video_stream is None:
                try:
                    print("Attempting to get video stream...")
                    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                except Exception as stream_error:
                    retries += 1
                    if retries >= max_retries:
                        print(f"Failed to get video stream after {max_retries} attempts: {stream_error}")
                        raise
                    print(f"Error getting video stream: {stream_error}. Retrying ({retries}/{max_retries})...")
                    time.sleep(2)  # Wait before retrying

            if not video_stream:
                raise Exception("Could not find a suitable video stream")

            # Download the video
            print(f"Downloading: {yt.title}")
            video_stream.download(output_path=output_path, filename="lecture.mp4")

            print(f"Video successfully downloaded using pytube: {output_file}")
            return output_file

    except Exception as e:
        print(f"Error downloading video: {str(e)}")
        raise
