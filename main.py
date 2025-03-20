"""Main script for the AutoLecturer pipeline."""
import os
import sys
import logging
import argparse
from pathlib import Path

from src.audio.extractor import AudioExtractor
from src.transcription.speech_to_text import TranscriberFactory
from src.text.cleaner import TextCleaner
from src.text.chunker import TextChunker
from src.video.clipper import VideoClipper
from src.config import INPUT_DIR, PROCESSED_DIR, OUTPUT_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_lecture(
        video_path,
        output_dir=None,
        skip_existing=False,
        extract_clips=True,
        combine_clips=True
):
    """Process a lecture video through the pipeline.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the output files. If None, use default output directory.
        skip_existing: Skip steps if output files already exist.
        extract_clips: Whether to extract video clips.
        combine_clips: Whether to combine the extracted clips into a single video.

    Returns:
        Dictionary with paths to the output files.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Set output directory
    if output_dir is None:
        output_dir = PROCESSED_DIR / video_path.stem
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_paths = {}
    
    # Step 1: Extract audio from the video
    logger.info("Step 1: Extracting audio from video")
    audio_extractor = AudioExtractor()
    audio_path = audio_extractor.extract_audio(video_path)
    output_paths["audio"] = audio_path
    
    # Step 2: Transcribe the audio
    logger.info("Step 2: Transcribing audio")
    transcriber = TranscriberFactory.create_transcriber()
    transcript_text, transcript_metadata = transcriber.transcribe(audio_path)
    output_paths["transcript"] = {"text": transcript_text, "metadata": transcript_metadata}
    
    # Find the JSON metadata file
    transcript_dir = Path(audio_path).parent.parent / "transcripts"
    transcript_json_files = list(transcript_dir.glob(f"{Path(audio_path).stem}*_transcript.json"))
    if transcript_json_files:
        transcript_json_path = transcript_json_files[0]
        logger.info(f"Found transcript metadata: {transcript_json_path}")
    else:
        transcript_json_path = None
        logger.warning("No transcript metadata file found. Timestamps will not be available.")
    
    # Step 3: Clean the transcript
    logger.info("Step 3: Cleaning transcript text")
    cleaner = TextCleaner()
    cleaned_text_path = cleaner.process_file(
        next(transcript_dir.glob(f"{Path(audio_path).stem}*_transcript.txt")),
        output_dir=output_dir
    )
    output_paths["cleaned_text"] = cleaned_text_path
    
    # Step 4: Chunk the cleaned text into paragraphs
    logger.info("Step 4: Chunking text into paragraphs")
    chunker = TextChunker()
    paragraphs_text_path, paragraphs_json_path = chunker.process_file(
        cleaned_text_path,
        metadata_path=transcript_json_path,
        output_dir=output_dir
    )
    output_paths["paragraphs"] = {
        "text": paragraphs_text_path,
        "json": paragraphs_json_path
    }
    
    # Step 5: Extract video clips based on timestamps (if requested)
    if extract_clips and transcript_json_path:
        logger.info("Step 5: Extracting video clips based on timestamps")
        clipper = VideoClipper()
        clip_paths = clipper.extract_clips(
            video_path,
            paragraphs_json_path,
            output_dir=output_dir / "clips"
        )
        output_paths["clips"] = clip_paths
        
        # Step 6: Combine clips into a single video (if requested)
        if combine_clips and clip_paths:
            logger.info("Step 6: Combining clips into a final video")
            combined_video_path = clipper.concatenate_clips(
                clip_paths,
                output_path=output_dir / f"{video_path.stem}_summarized.mp4"
            )
            output_paths["combined_video"] = combined_video_path
    
    logger.info(f"Pipeline completed. Output files saved to {output_dir}")
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="AutoLecturer: Process lecture videos into summarized content")
    parser.add_argument("--video", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, help="Directory to save output files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip steps if output files exist")
    parser.add_argument("--no-extract-clips", action="store_true", help="Skip extracting video clips")
    parser.add_argument("--no-combine-clips", action="store_true", help="Skip combining clips into final video")
    
    args = parser.parse_args()
    
    # If no video path provided, look for video in the input directory
    if not args.video:
        input_videos = list(INPUT_DIR.glob("*.mp4"))
        if not input_videos:
            logger.error(f"No video files found in {INPUT_DIR}. Please provide a video path with --video")
            sys.exit(1)
        video_path = input_videos[0]
        logger.info(f"Using video file: {video_path}")
    else:
        video_path = args.video
    
    try:
        process_lecture(
            video_path,
            output_dir=args.output,
            skip_existing=args.skip_existing,
            extract_clips=not args.no_extract_clips,
            combine_clips=not args.no_combine_clips
        )
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()