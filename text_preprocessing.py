import re
import nltk
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from config import MAX_CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special control characters but keep punctuation
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def remove_stopwords(text: str) -> str:
    """Remove stop words from text (optional, for some preprocessing steps)."""
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        filtered_words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(filtered_words)
    except:
        # If stopwords not available, return original text
        return text


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    try:
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]
    except:
        # Fallback to simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def chunk_text(text: str, max_chunk_size: int = MAX_CHUNK_SIZE, 
               chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with overlap."""
    sentences = split_into_sentences(text)
    
    if not sentences:
        return [text] if text else []
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        
        # If adding this sentence would exceed limit, save current chunk
        if current_size + sentence_tokens > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if chunk_overlap > 0:
                # Take last few sentences for overlap
                overlap_sentences = []
                overlap_size = 0
                for s in reversed(current_chunk):
                    s_tokens = estimate_tokens(s)
                    if overlap_size + s_tokens <= chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_size += s_tokens
                    else:
                        break
                current_chunk = overlap_sentences
                current_size = overlap_size
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks if chunks else [text]


def preprocess_text(text: str, clean: bool = True, 
                    chunk: bool = True) -> Tuple[str, List[str]]:
    """Preprocess text: clean and optionally chunk."""
    # Clean text
    if clean:
        cleaned = clean_text(text)
    else:
        cleaned = text
    
    # Chunk text
    if chunk:
        chunks = chunk_text(cleaned)
    else:
        chunks = [cleaned]
    
    return cleaned, chunks


def create_source_mapping(chunks: List[str], 
                          original_text: str) -> dict:
    """Create mapping from chunks to original text positions."""
    mapping = {}
    for idx, chunk in enumerate(chunks):
        # Find chunk position in original text
        start_pos = original_text.find(chunk[:50])  # Use first 50 chars for matching
        if start_pos != -1:
            mapping[idx] = {
                "start": start_pos,
                "end": start_pos + len(chunk),
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
        else:
            mapping[idx] = {
                "start": -1,
                "end": -1,
                "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
            }
    
    return mapping


def save_preprocessing_output(cleaned_text: str, chunks: List[str], 
                              output_dir: str, document_name: str):
    """Save preprocessing results (cleaned text and chunks) to output files."""
    import os
    from datetime import datetime
    
    # Create preprocessing subdirectory
    preprocess_dir = os.path.join(output_dir, "preprocessing")
    os.makedirs(preprocess_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save cleaned text
    cleaned_file = os.path.join(preprocess_dir, f"{document_name}_cleaned_{timestamp}.txt")
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    print(f"Saved cleaned text to {cleaned_file}")
    
    # Save chunks
    chunks_file = os.path.join(preprocess_dir, f"{document_name}_chunks_{timestamp}.txt")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        f.write(f"Total chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")
        for idx, chunk in enumerate(chunks):
            f.write(f"CHUNK {idx + 1}/{len(chunks)}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Token estimate: {estimate_tokens(chunk)}\n")
            f.write("-" * 80 + "\n")
            f.write(chunk)
            f.write("\n\n" + "=" * 80 + "\n\n")
    print(f"Saved chunks to {chunks_file}")
    
    # Save chunk summary as JSON
    import json
    chunks_summary = {
        "document_name": document_name,
        "timestamp": timestamp,
        "total_chunks": len(chunks),
        "total_tokens_estimate": sum(estimate_tokens(chunk) for chunk in chunks),
        "chunks": [
            {
                "chunk_id": idx + 1,
                "token_estimate": estimate_tokens(chunk),
                "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk
            }
            for idx, chunk in enumerate(chunks)
        ]
    }
    summary_file = os.path.join(preprocess_dir, f"{document_name}_chunks_summary_{timestamp}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved chunks summary to {summary_file}")
    
    return cleaned_file, chunks_file, summary_file

