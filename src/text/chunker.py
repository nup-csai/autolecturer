"""Module for chunking and organizing text into logical paragraphs."""
import re
import logging
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import math

import nltk
# Custom sentence tokenization using regex
import re

def sent_tokenize(text):
    """Simple sentence tokenizer using regex."""
    return re.split(r'(?<=[.!?])\s+', text)

from src.config import CHUNKING, PROCESSED_DIR
from src.utils.helpers import timing_decorator, ensure_dir, load_text, save_text, save_json, generate_output_filename

# Initialize logger
logger = logging.getLogger(__name__)

# Use regex for sentence tokenization instead of NLTK
logger.info("Using regex-based sentence tokenization instead of NLTK")


class TextChunker:
    """Class for chunking text into logical paragraphs with main ideas."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the text chunker with settings.

        Args:
            settings: Chunking settings. If None, use default settings from config.
        """
        self.settings = settings or CHUNKING
        self.method = self.settings.get("method", "sentence")
        self.max_tokens = self.settings.get("max_tokens", 512)
        self.overlap_tokens = self.settings.get("overlap_tokens", 50)
        self.min_chunk_size = self.settings.get("min_chunk_size", 100)
        logger.info(f"Initialized TextChunker with settings: {self.settings}")

    @timing_decorator
    def chunk_text(
            self,
            text: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text into logical paragraphs with main ideas.

        Args:
            text: Input text to chunk.
            metadata: Optional metadata with timestamp information.

        Returns:
            List of chunks with text, timestamps, and main ideas.
        """
        logger.info("Chunking text into logical paragraphs...")

        # Choose chunking method based on settings
        if self.method == "sentence" or metadata is None:
            chunks = self._chunk_by_sentences(text)
        elif self.method == "timestamp" and metadata:
            chunks = self._chunk_by_timestamps(text, metadata)
        else:
            chunks = self._chunk_by_tokens(text)

        # Extract main ideas for each chunk
        chunks_with_ideas = self._extract_main_ideas(chunks)

        logger.info(f"Created {len(chunks_with_ideas)} logical paragraphs")
        return chunks_with_ideas

    def _chunk_by_sentences(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by sentences into logical paragraphs.

        Args:
            text: Input text.

        Returns:
            List of chunks with text content.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            # Estimate token count (roughly words + punctuation)
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max tokens and we have enough content
            if current_token_count + sentence_tokens > self.max_tokens and current_token_count > self.min_chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_token_count
                })
                
                # Start new chunk with some overlap
                overlap_point = max(0, len(current_chunk) - math.ceil(self.overlap_tokens / 5))
                current_chunk = current_chunk[overlap_point:]
                current_token_count = sum(len(s.split()) for s in current_chunk)
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_token_count += sentence_tokens
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "token_count": current_token_count
            })
        
        return chunks

    def _chunk_by_timestamps(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text based on timestamps from transcription metadata.

        Args:
            text: Input text.
            metadata: Transcription metadata with timestamp information.

        Returns:
            List of chunks with text and timestamp information.
        """
        chunks = []
        
        # Check if segments exist in metadata
        segments = metadata.get("segments", [])
        if not segments:
            logger.warning("No segments found in metadata, falling back to sentence chunking")
            return self._chunk_by_sentences(text)
        
        # Group segments into chunks based on max_tokens
        current_chunk = []
        current_text = []
        current_token_count = 0
        chunk_start_time = segments[0].get("start", 0) if segments else 0
        
        for segment in segments:
            segment_text = segment.get("text", "").strip()
            if not segment_text:
                continue
                
            # Estimate token count
            segment_tokens = len(segment_text.split())
            
            # If adding this segment would exceed max tokens and we have enough content
            if current_token_count + segment_tokens > self.max_tokens and current_token_count > self.min_chunk_size:
                # Save current chunk
                chunk_text = " ".join(current_text)
                chunk_end_time = current_chunk[-1].get("end", 0) if current_chunk else 0
                
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_token_count,
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time
                })
                
                # Start new chunk
                chunk_start_time = segment.get("start", 0)
                current_chunk = []
                current_text = []
                current_token_count = 0
            
            # Add segment to current chunk
            current_chunk.append(segment)
            current_text.append(segment_text)
            current_token_count += segment_tokens
        
        # Add the last chunk if not empty
        if current_chunk:
            chunk_text = " ".join(current_text)
            chunk_end_time = current_chunk[-1].get("end", 0) if current_chunk else 0
            
            chunks.append({
                "text": chunk_text,
                "token_count": current_token_count,
                "start_time": chunk_start_time,
                "end_time": chunk_end_time
            })
        
        return chunks

    def _chunk_by_tokens(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by token count with overlap.

        Args:
            text: Input text.

        Returns:
            List of chunks with text content.
        """
        # Split text into words (tokens)
        words = text.split()
        
        chunks = []
        
        # If text is shorter than max_tokens, return as single chunk
        if len(words) <= self.max_tokens:
            return [{
                "text": text,
                "token_count": len(words)
            }]
        
        # Create chunks with overlap
        for i in range(0, len(words), self.max_tokens - self.overlap_tokens):
            # Calculate end index for current chunk
            end_idx = min(i + self.max_tokens, len(words))
            
            # Extract chunk
            chunk_words = words[i:end_idx]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "token_count": len(chunk_words)
            })
            
            # If we've reached the end of the text, break
            if end_idx == len(words):
                break
        
        return chunks

    def _extract_main_ideas(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract main ideas from each chunk of text.

        Args:
            chunks: List of text chunks.

        Returns:
            List of chunks with main ideas added.
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            
            # Extract first sentence as main idea (simple approach)
            sentences = sent_tokenize(text)
            main_idea = sentences[0] if sentences else ""
            
            # Create enhanced chunk with main idea
            enhanced_chunk = {
                **chunk,
                "main_idea": main_idea,
                "chunk_index": i,
                "paragraph_heading": self._generate_paragraph_heading(text)
            }
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks

    def _generate_paragraph_heading(self, text: str) -> str:
        """Generate a concise heading for a paragraph based on its content.

        Args:
            text: Paragraph text.

        Returns:
            A concise heading summarizing the main topic.
        """
        # Extract first sentence and clean it
        sentences = sent_tokenize(text)
        if not sentences:
            return "Unnamed Section"
            
        first_sentence = sentences[0]
        
        # Remove common filler phrases from beginning
        fillers = [
            "so,", "now,", "and,", "but,", "then,", "well,", "you know,", 
            "i mean,", "like,", "basically,", "actually,", "essentially,"
        ]
        
        cleaned_sentence = first_sentence.lower()
        for filler in fillers:
            if cleaned_sentence.startswith(filler):
                cleaned_sentence = cleaned_sentence[len(filler):].strip()
        
        # Limit length to create concise heading
        words = cleaned_sentence.split()
        if len(words) > 10:
            heading = " ".join(words[:10]) + "..."
        else:
            heading = cleaned_sentence
            
        # Capitalize first letter
        heading = heading[0].upper() + heading[1:]
        
        return heading

    @timing_decorator
    def process_file(
            self,
            file_path: Union[str, Path],
            metadata_path: Optional[Union[str, Path]] = None,
            output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """Process a text file and apply chunking.

        Args:
            file_path: Path to the input text file.
            metadata_path: Path to the metadata JSON file with timestamps.
            output_dir: Directory to save the chunked text. If None, use default processed directory.

        Returns:
            Tuple of paths to the chunked text file and JSON metadata.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or PROCESSED_DIR)

        # Load text from file
        text = load_text(file_path)

        # Load metadata if provided
        metadata = None
        if metadata_path:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load metadata file: {str(e)}")

        # Chunk the text
        chunks = self.chunk_text(text, metadata)

        # Generate output filenames
        output_text_path = generate_output_filename(
            file_path, output_dir, suffix="paragraphs", extension=".txt"
        )
        output_json_path = generate_output_filename(
            file_path, output_dir, suffix="paragraphs", extension=".json"
        )

        # Format text output with paragraph headings
        formatted_text = self._format_paragraphs_with_headings(chunks)

        # Save the chunked text and metadata
        save_text(formatted_text, output_text_path)
        save_json({"chunks": chunks}, output_json_path)

        logger.info(f"Chunked text saved to {output_text_path}")
        logger.info(f"Chunk metadata saved to {output_json_path}")

        return output_text_path, output_json_path
    
    def _format_paragraphs_with_headings(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into a readable text with paragraph headings.

        Args:
            chunks: List of text chunks with main ideas.

        Returns:
            Formatted text with paragraph headings.
        """
        formatted_text = ""
        
        for chunk in chunks:
            # Add paragraph heading
            formatted_text += f"## {chunk['paragraph_heading']}\n\n"
            
            # Add timestamp information if available
            if 'start_time' in chunk and 'end_time' in chunk:
                start_time = self._format_timestamp(chunk['start_time'])
                end_time = self._format_timestamp(chunk['end_time'])
                formatted_text += f"[{start_time} - {end_time}]\n\n"
            
            # Add paragraph text
            formatted_text += f"{chunk['text']}\n\n"
            formatted_text += "-" * 80 + "\n\n"
            
        return formatted_text
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds into MM:SS format.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted time string as MM:SS.
        """
        minutes = int(seconds / 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes:02d}:{remaining_seconds:02d}"