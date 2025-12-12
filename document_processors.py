import fitz  # PyMuPDF
import os
from typing import Tuple, Optional
from pathlib import Path


def extract_text_from_pdf(file_path: str) -> Tuple[str, dict]:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(file_path)
        text_parts = []
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "source": file_path
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(f"[Page {page_num + 1}]\n{text}")
        
        doc.close()
        full_text = "\n\n".join(text_parts)
        
        return full_text, metadata
    except Exception as e:
        raise Exception(f"Error extracting text from PDF {file_path}: {str(e)}")


def extract_text_from_txt(file_path: str) -> Tuple[str, dict]:
    """Extract text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        metadata = {
            "source": file_path,
            "file_size": os.path.getsize(file_path)
        }
        
        return text, metadata
    except Exception as e:
        raise Exception(f"Error extracting text from TXT {file_path}: {str(e)}")


def load_document(file_path: str) -> Tuple[str, dict]:
    """Load and extract text from a document (PDF or TXT)."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(str(file_path))
    elif file_ext in ['.txt', '.text']:
        return extract_text_from_txt(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: .pdf, .txt")


def load_multiple_documents(file_paths: list) -> list:
    """Load multiple documents and return list of (text, metadata) tuples."""
    documents = []
    for file_path in file_paths:
        try:
            text, metadata = load_document(file_path)
            documents.append((text, metadata))
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")
            continue
    
    return documents

