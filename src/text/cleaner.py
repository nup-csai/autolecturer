"""Module for cleaning and normalizing text transcripts."""
import re
import string
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Set

# Import spaCy conditionally
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

import nltk
# Custom tokenization using regex instead of NLTK
import re

def word_tokenize(text):
    """Simple word tokenizer using regex."""
    return re.findall(r'\b\w+\b|[^\w\s]', text)

def sent_tokenize(text):
    """Simple sentence tokenizer using regex."""
    return re.split(r'(?<=[.!?])\s+', text)
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

from src.config import TEXT_PROCESSING, PROCESSED_DIR
from src.utils.helpers import timing_decorator, ensure_dir, load_text, save_text, generate_output_filename

# Initialize logger
logger = logging.getLogger(__name__)

# Use regex for tokenization instead of NLTK
logger.info("Using regex-based tokenization instead of NLTK")


class TextCleaner:
    """Class for cleaning and normalizing text transcripts."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        """Initialize the text cleaner with settings.

        Args:
            settings: Text processing settings. If None, use default settings from config.
        """
        self.settings = settings or TEXT_PROCESSING
        self.remove_filler_words = self.settings.get("remove_filler_words", True)
        self.filler_words = set(self.settings.get("filler_words", []))
        self.spell_check = self.settings.get("spell_check", True) and SYMSPELL_AVAILABLE
        self.normalize_case = self.settings.get("normalize_case", True)
        self.lemmatize = self.settings.get("lemmatize", False) and SPACY_AVAILABLE
        self.enhance_readability = self.settings.get("enhance_readability", True)
        self.fix_punctuation = self.settings.get("fix_punctuation", True)

        # Initialize NLP components as needed
        self.nlp = None
        self.symspell = None

        logger.info(f"Initialized TextCleaner with settings: {self.settings}")

    def _initialize_spacy(self):
        """Initialize spaCy NLP model if not already loaded."""
        if self.nlp is None and SPACY_AVAILABLE:
            try:
                # Load English model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model 'en_core_web_sm'")
            except Exception as e:
                logger.error(f"Error initializing spaCy: {str(e)}")
                self.nlp = None
        elif not SPACY_AVAILABLE:
            logger.warning("spaCy is not available. Install it with 'pip install spacy' and 'python -m spacy download en_core_web_sm'")
            self.nlp = None

    def _initialize_symspell(self):
        """Initialize SymSpell spell checker if not already loaded."""
        if self.symspell is None and self.spell_check and SYMSPELL_AVAILABLE:
            try:
                self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = Path(__file__).parent / "data" / "frequency_dictionary_en_82_765.txt"

                # If dictionary doesn't exist, create the data directory and download it
                if not dictionary_path.exists():
                    import requests

                    # Create data directory
                    ensure_dir(dictionary_path.parent)

                    # URL for the English frequency dictionary
                    url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"

                    # Download the dictionary
                    logger.info(f"Downloading SymSpell dictionary from {url}")
                    response = requests.get(url)
                    with open(dictionary_path, "wb") as f:
                        f.write(response.content)

                # Load the dictionary
                self.symspell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
                logger.info("SymSpell dictionary loaded successfully")

            except Exception as e:
                logger.error(f"Error initializing SymSpell: {str(e)}")
                self.spell_check = False
        elif not SYMSPELL_AVAILABLE:
            logger.warning("SymSpell is not available. Install it with 'pip install symspellpy'")
            self.spell_check = False

    @timing_decorator
    def clean_text(
            self,
            text: str,
            remove_filler: Optional[bool] = None,
            spell_check: Optional[bool] = None,
            normalize_case: Optional[bool] = None,
            lemmatize: Optional[bool] = None,
            enhance_readability: Optional[bool] = None,
            fix_punctuation: Optional[bool] = None
    ) -> str:
        """Clean and normalize text.

        Args:
            text: Input text to clean.
            remove_filler: Whether to remove filler words.
            spell_check: Whether to apply spell checking.
            normalize_case: Whether to normalize case.
            lemmatize: Whether to apply lemmatization.
            enhance_readability: Whether to enhance readability.
            fix_punctuation: Whether to fix punctuation.

        Returns:
            Cleaned text.
        """
        # Use provided parameters or fall back to instance settings
        remove_filler = remove_filler if remove_filler is not None else self.remove_filler_words
        spell_check = spell_check if spell_check is not None else self.spell_check
        normalize_case = normalize_case if normalize_case is not None else self.normalize_case
        lemmatize = lemmatize if lemmatize is not None else self.lemmatize
        enhance_readability = enhance_readability if enhance_readability is not None else self.enhance_readability
        fix_punctuation = fix_punctuation if fix_punctuation is not None else self.fix_punctuation

        logger.info("Cleaning text...")

        # Step 1: Basic text normalization
        text = self._normalize_whitespace(text)

        # Step 2: Remove filler words if enabled
        if remove_filler:
            text = self._remove_filler_words(text)

        # Step 3: Fix punctuation if enabled
        if fix_punctuation:
            text = self._fix_punctuation(text)

        # Step 4: Apply spell checking if enabled
        if spell_check:
            text = self._apply_spell_checking(text)

        # Step 5: Case normalization if enabled
        if normalize_case:
            text = self._normalize_case(text)

        # Step 6: Lemmatization if enabled
        if lemmatize:
            text = self._apply_lemmatization(text)

        # Step 7: Enhance readability if enabled
        if enhance_readability:
            text = self._enhance_readability(text)

        logger.info("Text cleaning completed")
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace, remove extra spaces, etc."""
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)

        # Normalize newlines
        text = re.sub(r'\n+', '\n', text)

        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)

        # Add space after punctuation if not followed by a space
        text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)

        return text.strip()

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from text."""
        logger.info("Removing filler words...")

        # Tokenize the text
        words = word_tokenize(text)

        # Filter out filler words
        filtered_words = [word for word in words if word.lower() not in self.filler_words]

        # Reconstruct the text
        filtered_text = ' '.join(filtered_words)

        return self._normalize_whitespace(filtered_text)

    def _fix_punctuation(self, text: str) -> str:
        """Fix common punctuation issues."""
        logger.info("Fixing punctuation...")

        # Fix repeated punctuation
        text = re.sub(r'([.,;:!?]){2,}', r'\1', text)

        # Add periods at the end of sentences if missing
        sentences = sent_tokenize(text)
        fixed_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
            fixed_sentences.append(sentence)

        # Ensure proper spacing between sentences
        text = ' '.join(fixed_sentences)

        # Fix quotes
        text = re.sub(r'``', '"', text)
        text = re.sub(r"''", '"', text)

        return text

    def _apply_spell_checking(self, text: str) -> str:
        """Apply spell checking to correct misspelled words."""
        logger.info("Applying spell checking...")

        # Initialize SymSpell if not already initialized
        self._initialize_symspell()

        if self.symspell is None:
            logger.warning("SymSpell not initialized, skipping spell checking")
            return text

        # Tokenize the text
        words = word_tokenize(text)
        corrected_words = []

        for word in words:
            # Skip punctuation and numeric tokens
            if word in string.punctuation or word.isdigit():
                corrected_words.append(word)
                continue

            # Skip short words (likely correct or abbreviations)
            if len(word) <= 2:
                corrected_words.append(word)
                continue

            # Apply spell checking
            suggestions = self.symspell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)

            if suggestions:
                # Get the most likely correction
                corrected_word = suggestions[0].term

                # Preserve original capitalization
                if word.isupper():
                    corrected_word = corrected_word.upper()
                elif word[0].isupper():
                    corrected_word = corrected_word.capitalize()

                corrected_words.append(corrected_word)
            else:
                # If no suggestions, keep the original word
                corrected_words.append(word)

        # Reconstruct the text
        corrected_text = ' '.join(corrected_words)

        return self._normalize_whitespace(corrected_text)

    def _normalize_case(self, text: str) -> str:
        """Normalize case for sentence beginnings and proper nouns."""
        logger.info("Normalizing case...")

        # Initialize spaCy if not already initialized
        self._initialize_spacy()

        if self.nlp and SPACY_AVAILABLE:
            try:
                # Use spaCy for advanced case normalization
                doc = self.nlp(text)
                
                # Get token information
                tokens_with_case = []
                for token in doc:
                    # Keep proper nouns capitalized
                    if token.pos_ == "PROPN":
                        tokens_with_case.append(token.text.capitalize())
                    # Keep acronyms in all caps
                    elif token.text.isupper() and len(token.text) > 1:
                        tokens_with_case.append(token.text)
                    else:
                        tokens_with_case.append(token.text.lower())
                
                # Reconstruct the text
                text = ' '.join(tokens_with_case)
                
                # Capitalize first letter of sentences
                sentences = sent_tokenize(text)
                normalized_sentences = [s.capitalize() for s in sentences]
                return ' '.join(normalized_sentences)
            except Exception as e:
                logger.error(f"Error in spaCy case normalization: {str(e)}")
        
        # Fallback: Basic case normalization
        logger.warning("Using basic case normalization")
        # Basic case normalization: capitalize first letter of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        normalized_sentences = [s.capitalize() for s in sentences]
        return ' '.join(normalized_sentences)

    def _apply_lemmatization(self, text: str) -> str:
        """Apply lemmatization to reduce words to their base forms."""
        logger.info("Applying lemmatization...")

        # Initialize spaCy if not already initialized
        self._initialize_spacy()

        if self.nlp and SPACY_AVAILABLE:
            try:
                # Use spaCy for lemmatization
                doc = self.nlp(text)
                
                # Preserve sentence structure but lemmatize content words
                lemmatized_tokens = []
                for token in doc:
                    # Don't lemmatize proper nouns, determiners, pronouns, etc.
                    if token.pos_ in ["PROPN", "DET", "PRON", "ADP", "PUNCT", "AUX", "PART", "INTJ"]:
                        lemmatized_tokens.append(token.text)
                    else:
                        lemmatized_tokens.append(token.lemma_)
                
                return ' '.join(lemmatized_tokens)
            except Exception as e:
                logger.error(f"Error in spaCy lemmatization: {str(e)}")
                return text
        else:
            logger.warning("spaCy not available, skipping lemmatization")
            return text

    def _enhance_readability(self, text: str) -> str:
        """Enhance readability by fixing sentence structure and formatting."""
        logger.info("Enhancing readability...")
        
        # Split into sentences
        sentences = sent_tokenize(text)
        enhanced_sentences = []
        
        for sentence in sentences:
            # Fix common readability issues
            
            # Remove unnecessary repetitions of words
            words = sentence.split()
            unique_words = []
            for i, word in enumerate(words):
                # Skip if it's a repeated word (allow common repeats like "very very")
                if i > 0 and word.lower() == words[i-1].lower() and word.lower() not in ["very", "really"]:
                    continue
                unique_words.append(word)
            
            # Reconstruct sentence
            enhanced_sentence = ' '.join(unique_words)
            
            # Ensure proper capitalization at the beginning of sentences
            if enhanced_sentence and enhanced_sentence[0].islower():
                enhanced_sentence = enhanced_sentence[0].upper() + enhanced_sentence[1:]
                
            enhanced_sentences.append(enhanced_sentence)
        
        # Join sentences with proper spacing
        enhanced_text = ' '.join(enhanced_sentences)
        
        # Fix common issues with joined sentences
        enhanced_text = re.sub(r'(\w)\.(\w)', r'\1. \2', enhanced_text)
        
        return enhanced_text

    @timing_decorator
    def process_file(
            self,
            file_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """Process a text file and apply cleaning.

        Args:
            file_path: Path to the input text file.
            output_dir: Directory to save the cleaned text. If None, use default processed directory.

        Returns:
            Path to the cleaned text file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        # Set output directory
        output_dir = ensure_dir(output_dir or PROCESSED_DIR)

        # Load text from file
        text = load_text(file_path)

        # Clean the text
        cleaned_text = self.clean_text(text)

        # Generate output filename
        output_path = generate_output_filename(
            file_path, output_dir, suffix="cleaned", extension=".txt"
        )

        # Save the cleaned text
        save_text(cleaned_text, output_path)

        logger.info(f"Cleaned text saved to {output_path}")

        return output_path