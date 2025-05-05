"""
Module for abstractive text summarization using T5 model with overlapping chunks.
"""

import ssl
import logging
import os
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Disable SSL verification (if needed)
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Default model for summarization
DEFAULT_MODEL = "t5-large"

class Summarizer:
    """Abstractive summarization of English text using T5 with overlapping chunks."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading `{model_name}` to {self.device}")

        logger.info("Loading tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        logger.info("Tokenizer loaded successfully")

        logger.info("Loading model... This may take a few minutes")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        logger.info("Model loaded successfully")

        self.model.eval()
        self.max_tokens = 512  # T5 has a fixed input size

    def chunk(self, text: str, overlap: int = 100) -> list[str]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_len = self.max_tokens - 2
        stride = max_len - overlap
        chunks = []
        for start in range(0, len(tokens), stride):
            window = tokens[start:start + max_len]
            chunk_text = self.tokenizer.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            chunks.append(chunk_text)
            if start + max_len >= len(tokens):
                break
        return chunks

    def summarize_chunk(self, chunk: str) -> str:
        prompt = "summarize: " + chunk.strip().replace("\n", " ")
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
        outputs = self.model.generate(
            inputs,
            num_beams=4,
            length_penalty=1.0,
            max_length=250,
            min_length=100,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            early_stopping=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize(self, text: str) -> str:
        logger.info("Creating overlapping chunks")
        chunks = self.chunk(text)
        partial_summaries = []
        for i, ch in tqdm(enumerate(chunks, 1), total=len(chunks), desc="Processing chunks"):
            logger.info(f"Summarizing chunk {i}/{len(chunks)}")
            partial_summaries.append(self.summarize_chunk(ch))
        logger.info("Combining chunk summaries")
        return self.summarize_chunk(" ".join(partial_summaries))


if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "summary.txt"

    if not os.path.exists(input_file):
        logger.error(f"Input file `{input_file}` not found.")
        exit(1)

    with open(input_file, "r", encoding="utf-8") as f:
        input_text = f.read()

    summarizer = Summarizer()
    summary = summarizer.summarize(input_text)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"Summary saved to `{output_file}`")
