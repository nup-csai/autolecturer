import os
import requests
import base64
import time
import json
from dotenv import load_dotenv
import os


class DIDVideoGenerator:
    def __init__(self):
        """
        Initialize client for working with D-ID API

        :param api_key: Your D-ID API key
        """
        load_dotenv()

        self.api_key = os.getenv("D_ID_API_KEY")
        self.base_url = "https://api.d-id.com"

        # Proper Base64 encoding
        encoded_key = base64.b64encode(f"{self.api_key}:".encode('utf-8')).decode('utf-8')
        self.headers = {
            "Authorization": f"Basic {encoded_key}",
            "Content-Type": "application/json"
        }

    def read_text_file(self, file_path):
        """
        Read text from file

        :param file_path: Path to text file
        :return: File contents
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                # Limit text length
                return text[:1000]  # Maximum 1000 characters
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def create_talking_photo(self, text):
        """
        Create video with talking photo

        :param text: Text to speak
        :return: ID of created presentation
        """
        create_url = f"{self.base_url}/talks"

        payload = {
            "script": {
                "type": "text",
                "input": text,
                "provider": {
                    "type": "microsoft",
                    "voice_id": "en-US-JennyNeural"
                }
            },
            "presenter_id": "amy_45",  # Built-in avatar ID
            "config": {
                "result_format": "mp4"
            }
        }

        try:
            # Detailed debug output
            print("Sending request with payload:")
            print(json.dumps(payload, indent=2))

            response = requests.post(create_url,
                                     headers=self.headers,
                                     json=payload)

            # Full response information
            print("Response status:", response.status_code)
            print("Full response text:", response.text)

            response.raise_for_status()

            result = response.json()

            if 'id' in result:
                print(f"Created Talk ID: {result['id']}")
                return result['id']
            else:
                print("Failed to get presentation ID")
                return None

        except requests.exceptions.RequestException as e:
            print(f"FULL REQUEST ERROR: {e}")
            # Output server response if available
            if hasattr(e, 'response'):
                print(f"Server response: {e.response.text}")
            return None

    def get_talk_result(self, talk_id, max_attempts=30, delay=10):
        """
        Get result of video creation

        :param talk_id: Presentation ID
        :param max_attempts: Maximum number of attempts
        :param delay: Delay between attempts in seconds
        :return: Video URL or None
        """
        for _ in range(max_attempts):
            try:
                result_url = f"{self.base_url}/talks/{talk_id}"
                response = requests.get(result_url, headers=self.headers)
                response.raise_for_status()

                status = response.json()

                if status.get('status') == 'done':
                    return status.get('result_url')
                elif status.get('status') == 'failed':
                    print("Video creation failed")
                    return None

                # Wait before next check
                time.sleep(delay)

            except requests.exceptions.RequestException as e:
                print(f"Error checking status: {e}")
                return None

        print("Video generation timeout exceeded")
        return None

    def download_video(self, video_url, output_path):
        """
        Download generated video

        :param video_url: Video URL
        :param output_path: Path to save file
        """
        try:
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Video saved: {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading video: {e}")


def main():
    # IMPORTANT: Replace with your actual API key
    # Path to text file
    TEXT_FILE_PATH = "text_to_speak.txt"

    # Path to save video
    OUTPUT_VIDEO_PATH = "generated_avatar_video.mp4"

    # Create generator instance
    generator = DIDVideoGenerator()

    # Read text
    text = generator.read_text_file(TEXT_FILE_PATH)

    if text:
        # Create video with talking photo
        talk_id = generator.create_talking_photo(text)

        if talk_id:
            # Get video URL
            video_url = generator.get_talk_result(talk_id)

            if video_url:
                # Download video
                generator.download_video(video_url, OUTPUT_VIDEO_PATH)
            else:
                print("Failed to get video URL.")
        else:
            print("Failed to create talking photo.")
    else:
        print("Failed to read text for speaking.")


if __name__ == "__main__":
    main()