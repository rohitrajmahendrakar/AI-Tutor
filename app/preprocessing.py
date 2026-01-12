"""
Unified preprocessing, chunking, and indexing pipeline for all data sources.
This module ensures consistent processing across datasets, uploaded files, and embeddings.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

try:
    from .constants import DATASET_DIR
except ImportError:
    from constants import DATASET_DIR

# Consistent naming conventions
PROCESSED_DIR = DATASET_DIR / "processed"
INDEX_DIR = DATASET_DIR / "indices"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def preprocess_text(text: str) -> str:
    """
    Unified text preprocessing function used across all modules.
    
    Args:
        text: Raw text input
        
    Returns:
        Cleaned and normalized text
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Strip whitespace
    text = text.strip()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
               chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[:break_point + 1]
                end = start + break_point + 1
        
        chunks.append(chunk.strip())
        start = end - chunk_overlap
    
    return [c for c in chunks if c]  # Remove empty chunks


def create_index(chunks: List[str], index_name: str) -> Tuple[TfidfVectorizer, any]:
    """
    Create a TF-IDF index from text chunks.
    
    Args:
        chunks: List of text chunks to index
        index_name: Name for the index file (without extension)
        
    Returns:
        Tuple of (vectorizer, document vectors)
    """
    if not chunks:
        return None, None
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    doc_vectors = vectorizer.fit_transform(chunks)
    
    # Save index with consistent naming
    index_path = INDEX_DIR / f"{index_name}.index"
    with open(index_path, 'wb') as f:
        pickle.dump((vectorizer, doc_vectors), f)
    
    return vectorizer, doc_vectors


def load_index(index_name: str) -> Tuple[Optional[TfidfVectorizer], Optional[any]]:
    """
    Load a saved index.
    
    Args:
        index_name: Name of the index file (without extension)
        
    Returns:
        Tuple of (vectorizer, document vectors) or (None, None) if not found
    """
    index_path = INDEX_DIR / f"{index_name}.index"
    if not index_path.exists():
        return None, None
    
    try:
        with open(index_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading index {index_name}: {e}")
        return None, None


def process_dataset_csv(csv_path: Path, output_name: str) -> pd.DataFrame:
    """
    Process a CSV dataset using the unified pipeline.
    
    Args:
        csv_path: Path to the CSV file
        output_name: Base name for output files (e.g., "main_dataset")
        
    Returns:
        Processed DataFrame
    """
    # Load raw dataset
    try:
        df = pd.read_csv(csv_path, encoding="latin-1")
    except Exception:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
    
    # Clean: remove duplicates and nulls
    df.drop_duplicates(subset="Question", inplace=True)
    df.dropna(subset=["Question", "Answer"], inplace=True)
    
    # Preprocess: apply unified preprocessing
    df["Question"] = df["Question"].astype(str).apply(preprocess_text).str.lower()
    df["Answer"] = df["Answer"].astype(str).apply(preprocess_text)
    
    # Save processed dataset with consistent naming
    processed_path = PROCESSED_DIR / f"{output_name}_processed.csv"
    df.to_csv(processed_path, index=False)
    
    # Create chunks from answers
    all_chunks = []
    chunk_metadata = []
    
    for idx, row in df.iterrows():
        answer_chunks = chunk_text(row["Answer"])
        all_chunks.extend(answer_chunks)
        for chunk_idx, chunk in enumerate(answer_chunks):
            chunk_metadata.append({
                "dataset": output_name,
                "question_idx": idx,
                "chunk_idx": chunk_idx,
                "question": row["Question"],
                "answer": row["Answer"]
            })
    
    # Create and save index
    vectorizer, doc_vectors = create_index(all_chunks, output_name)
    
    # Save chunk metadata as JSONL
    jsonl_path = PROCESSED_DIR / f"{output_name}_chunks.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for meta in chunk_metadata:
            f.write(json.dumps(meta, ensure_ascii=False) + '\n')
    
    print(f"Processed dataset '{output_name}': {len(df)} Q&A pairs, {len(all_chunks)} chunks")
    
    return df


def process_uploaded_file(file_path: Path, file_name: str) -> Tuple[List[str], List[Dict]]:
    """
    Process an uploaded file using the unified pipeline.
    
    Args:
        file_path: Path to the uploaded file
        file_name: Original filename
        
    Returns:
        Tuple of (chunks, metadata list)
    """
    # Determine file type and extract text
    text_content = ""
    
    if file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    elif file_path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            # Convert CSV to text representation
            text_content = df.to_string()
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
    elif file_path.suffix.lower() in ['.md', '.markdown']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text_content = f.read()
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    # Preprocess
    cleaned_text = preprocess_text(text_content)
    
    # Chunk
    chunks = chunk_text(cleaned_text)
    
    # Create metadata
    metadata = []
    for idx, chunk in enumerate(chunks):
        metadata.append({
            "source": "uploaded_file",
            "filename": file_name,
            "chunk_idx": idx,
            "total_chunks": len(chunks)
        })
    
    return chunks, metadata


