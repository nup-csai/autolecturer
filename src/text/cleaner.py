"""Module for cleaning and normalizing text transcripts."""
import re
import string
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Set

# Temporarily comment out spacy import
# import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from symspellpy import SymSpell, Verbosity

from src.config import TEXT_PROCESSING, PROCESSED_DIR
from src.utils.helpers import timing_decorator, ensure_dir, load_text, save_text, generate_output_filename

# Initialize logger
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')


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
        self.spell_check = self.settings.get("spell_check", True)
        self.normalize_case = self.settings.get("normalize_case", True)
        self.lemmatize = self.settings.get("lemmatize", False)

        # Initialize NLP components as needed
        self.nlp = None
        self.symspell = None

        logger.info(f"Initialized TextCleaner with settings: {self.settings}")

    def _initialize_spacy(self):
        """Initialize spaCy NLP model if not already loaded."""
        if self.nlp is None:
            try:
                # Temporarily disabled spaCy
                # Load English model
                # self.nlp = spacy.load("en_core_web_sm")
                # logger.info("Loaded spaCy model 'en_core_web_sm'")
                logger.warning("spaCy is temporarily disabled")
                self.nlp = None
            except Exception as e:
                logger.error(f"Error initializing spaCy: {str(e)}")
                self.nlp = None

    def _initialize_symspell(self):
        """Initialize SymSpell spell checker if not already loaded."""
        if self.symspell is None and self.spell_check:
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

    @timing_decorator
    def clean_text(
            self,
            text: str,
            remove_filler: Optional[bool] = None,
            spell_check: Optional[bool] = None,
            normalize_case: Optional[bool] = None,
            lemmatize: Optional[bool] = None
    ) -> str:
        """Clean and normalize text.

        Args:
            text: Input text to clean.
            remove_filler: Whether to remove filler words.
            spell_check: Whether to apply spell checking.
            normalize_case: Whether to normalize case.
            lemmatize: Whether to apply lemmatization.

        Returns:
            Cleaned text.
        """
        # Use provided parameters or fall back to instance settings
        remove_filler = remove_filler if remove_filler is not None else self.remove_filler_words
        spell_check = spell_check if spell_check is not None else self.spell_check
        normalize_case = normalize_case if normalize_case is not None else self.normalize_case
        lemmatize = lemmatize if lemmatize is not None else self.lemmatize

        logger.info("Cleaning text...")

        # Step 1: Basic text normalization
        text = self._normalize_whitespace(text)

        # Step 2: Remove filler words if enabled
        if remove_filler:
            text = self._remove_filler_words(text)

        # Step 3: Apply spell checking if enabled
        if spell_check:
            text = self._apply_spell_checking(text)

        # Step 4: Case normalization if enabled
        if normalize_case:
            text = self._normalize_case(text)

        # Step 5: Lemmatization if enabled
        if lemmatize:
            text = self._apply_lemmatization(text)

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

        # Use basic case normalization since spaCy is disabled
        logger.warning("Using basic case normalization (spaCy disabled)")
        # Basic case normalization: capitalize first letter of sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        normalized_sentences = [s.capitalize() for s in sentences]
        return ' '.join(normalized_sentences)

    def _apply_lemmatization(self, text: str) -> str:
        """Apply lemmatization to reduce words to their base forms."""
        logger.info("Applying lemmatization...")

        # Initialize spaCy if not already initialized
        self._initialize_spacy()

        # spaCy is disabled, skip lemmatization
        logger.warning("spaCy is disabled, skipping lemmatization")
        return text

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