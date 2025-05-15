import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import math

def load_bart_model():
    """
    Load the BART model and tokenizer for text summarization.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading BART model for summarization...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def chunk_text(text, max_length=1024):
    """
    Split text into chunks that can be processed by the model.
    
    Args:
        text (str): The text to chunk
        max_length (int): Maximum token length for each chunk
        
    Returns:
        list: List of text chunks
    """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_tokens = tokenizer.tokenize(word)
        word_token_length = len(word_tokens)
        
        if current_length + word_token_length <= max_length:
            current_chunk.append(word)
            current_length += word_token_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_token_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_chunk(model, tokenizer, chunk, max_length=150, min_length=50):
    """
    Summarize a single chunk of text.
    
    Args:
        model: BART model
        tokenizer: BART tokenizer
        chunk (str): Text chunk to summarize
        max_length (int): Maximum length of the summary
        min_length (int): Minimum length of the summary
        
    Returns:
        str: Summarized text
    """
    inputs = tokenizer([chunk], max_length=1024, return_tensors='pt', truncation=True)
    
    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        num_beams=4,
        min_length=min_length,
        max_length=max_length,
        early_stopping=True
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def summarize_text(input_file, output_file, target_duration_minutes=2.5):
    """
    Summarize text from input file and write to output file.
    The summary is designed to be a coherent script for a 2-3 minute trailer.
    
    Args:
        input_file (str): Path to the input text file
        output_file (str): Path to save the summarized text
        target_duration_minutes (float): Target duration of the spoken summary in minutes
        
    Returns:
        str: Path to the summarized text file
    """
    try:
        # Load the text
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Load model
        model, tokenizer = load_bart_model()
        
        # Split text into chunks
        chunks = chunk_text(text)
        print(f"Split text into {len(chunks)} chunks for processing")
        
        # Estimate words needed for target duration (average speaking rate is ~150 words per minute)
        target_word_count = int(target_duration_minutes * 150)
        
        # Calculate how much to summarize each chunk based on total text length and target length
        total_words = len(text.split())
        compression_ratio = target_word_count / total_words
        
        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            chunk_word_count = len(chunk.split())
            target_chunk_length = max(50, min(150, int(chunk_word_count * compression_ratio * 1.5)))
            
            summary = summarize_chunk(
                model, 
                tokenizer, 
                chunk, 
                max_length=target_chunk_length, 
                min_length=min(30, target_chunk_length // 2)
            )
            summaries.append(summary)
        
        # Combine summaries into a coherent text
        combined_summary = " ".join(summaries)
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_summary)
        
        print(f"Summary completed and saved to {output_file}")
        print(f"Summary contains {len(combined_summary.split())} words, targeting ~{target_duration_minutes} minutes of speech")
        
        return output_file
        
    except Exception as e:
        print(f"Error summarizing text: {e}")
        raise