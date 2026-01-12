"""
Modular RAG (Retrieval-Augmented Generation) pipeline components.
Each function is small, testable, and focused on a single responsibility.
"""
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from .preprocessing import preprocess_text, load_index, PROCESSED_DIR, INDEX_DIR
    from .constants import SIMILARITY_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD, UNCERTAIN_RESPONSE, PROMPT_PREFIX
except ImportError:
    from preprocessing import preprocess_text, load_index, PROCESSED_DIR, INDEX_DIR
    from constants import SIMILARITY_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD, UNCERTAIN_RESPONSE, PROMPT_PREFIX

# Global state for datasets and indices
_main_dataset = None
_main_vectorizer = None
_main_doc_vectors = None
_main_chunks = None
_main_chunk_metadata = None

_uploaded_chunks = []
_uploaded_metadata = []
_uploaded_vectorizer = None
_uploaded_doc_vectors = None


def initialize_main_dataset(dataset_name: str = "main_dataset"):
    """
    Initialize the main curated dataset for retrieval.
    
    Args:
        dataset_name: Name of the processed dataset to load
    """
    global _main_dataset, _main_vectorizer, _main_doc_vectors, _main_chunks, _main_chunk_metadata
    
    # Load processed dataset
    processed_path = PROCESSED_DIR / f"{dataset_name}_processed.csv"
    if processed_path.exists():
        _main_dataset = pd.read_csv(processed_path)
    else:
        # Try to load from cleaned_dataset.csv (legacy format)
        from .constants import DATASET_DIR
        legacy_path = DATASET_DIR / "cleaned_dataset.csv"
        if legacy_path.exists():
            _main_dataset = pd.read_csv(legacy_path)
        else:
            _main_dataset = pd.DataFrame({"Question": [], "Answer": []})
            return
    
    # Load index
    _main_vectorizer, _main_doc_vectors = load_index(dataset_name)
    
    # Load chunks metadata
    import json
    jsonl_path = PROCESSED_DIR / f"{dataset_name}_chunks.jsonl"
    _main_chunks = []
    _main_chunk_metadata = []
    
    if jsonl_path.exists():
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    meta = json.loads(line.strip())
                    _main_chunks.append(meta.get("answer", ""))
                    _main_chunk_metadata.append(meta)
        except Exception as e:
            print(f"Warning: Could not load chunks metadata: {e}")
    
    if _main_vectorizer is None or _main_doc_vectors is None:
        # Fallback: create index from questions if chunks not available
        if not _main_dataset.empty:
            _main_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            _main_doc_vectors = _main_vectorizer.fit_transform(
                _main_dataset["Question"].values.astype("U")
            )
            # Use answers as chunks if chunks not available
            if not _main_chunks:
                _main_chunks = _main_dataset["Answer"].astype(str).tolist()
                _main_chunk_metadata = [
                    {"question_idx": i, "chunk_idx": 0, "question": q, "answer": a}
                    for i, (q, a) in enumerate(zip(_main_dataset["Question"], _main_dataset["Answer"]))
                ]


def initialize_uploaded_files(chunks: List[str], metadata: List[Dict], index_name: str = "uploaded_files"):
    """
    Initialize uploaded files for retrieval.
    
    Args:
        chunks: List of text chunks from uploaded files
        metadata: Metadata for each chunk
        index_name: Name for the uploaded files index
    """
    global _uploaded_chunks, _uploaded_metadata, _uploaded_vectorizer, _uploaded_doc_vectors
    
    if not chunks:
        _uploaded_chunks = []
        _uploaded_metadata = []
        _uploaded_vectorizer = None
        _uploaded_doc_vectors = None
        return
    
    _uploaded_chunks = chunks
    _uploaded_metadata = metadata
    
    # Create index for uploaded files
    _uploaded_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    _uploaded_doc_vectors = _uploaded_vectorizer.fit_transform(chunks)
    
    # Save index
    import pickle
    index_path = INDEX_DIR / f"{index_name}.index"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, 'wb') as f:
        pickle.dump((_uploaded_vectorizer, _uploaded_doc_vectors), f)


def retrieve_chunks(query: str, source: str = "main", top_k: int = 3, 
                   threshold: float = SIMILARITY_THRESHOLD) -> List[Tuple[str, float, Dict]]:
    """
    Retrieve top-k most relevant chunks for a query.
    
    Args:
        query: User query
        source: "main" for main dataset, "uploaded" for uploaded files
        top_k: Number of chunks to retrieve
        threshold: Minimum similarity threshold
        
    Returns:
        List of tuples: (chunk_text, similarity_score, metadata)
    """
    # Preprocess query
    normalized_query = preprocess_text(query).lower()
    
    if source == "main":
        vectorizer = _main_vectorizer
        doc_vectors = _main_doc_vectors
        chunks = _main_chunks if _main_chunks else []
        metadata = _main_chunk_metadata if _main_chunk_metadata else []
        # Fallback: use dataset directly if chunks not available
        if not chunks and _main_dataset is not None and not _main_dataset.empty:
            chunks = _main_dataset["Answer"].astype(str).tolist()
            metadata = [
                {"question_idx": i, "chunk_idx": 0, "question": q, "answer": a}
                for i, (q, a) in enumerate(zip(_main_dataset["Question"], _main_dataset["Answer"]))
            ]
    elif source == "uploaded":
        vectorizer = _uploaded_vectorizer
        doc_vectors = _uploaded_doc_vectors
        chunks = _uploaded_chunks
        metadata = _uploaded_metadata
    else:
        return []
    
    if vectorizer is None or doc_vectors is None:
        return []
    
    if not chunks:
        return []
    
    # Transform query
    query_vec = vectorizer.transform([normalized_query])
    
    # Compute similarities
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    
    # Get top-k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Filter by threshold and return
    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score >= threshold:
            chunk_text = chunks[idx] if idx < len(chunks) else ""
            chunk_meta = metadata[idx] if idx < len(metadata) else {}
            results.append((chunk_text, score, chunk_meta))
    
    return results


def rank_chunks(chunks_with_scores: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
    """
    Rank chunks by similarity score (already sorted, but can add additional ranking logic).
    
    Args:
        chunks_with_scores: List of (chunk_text, score, metadata) tuples
        
    Returns:
        Ranked list of chunks
    """
    # Already sorted by similarity, but we can add diversity or other ranking
    return sorted(chunks_with_scores, key=lambda x: x[1], reverse=True)


def clean_context_chunk(chunk: str) -> str:
    """
    Clean a context chunk to remove formatting artifacts.
    
    Args:
        chunk: Raw context chunk
        
    Returns:
        Cleaned chunk text
    """
    if not isinstance(chunk, str):
        chunk = str(chunk)
    
    # Remove list-like string representations
    import re
    if chunk.startswith('[') and chunk.endswith(']'):
        try:
            import json
            parsed = json.loads(chunk)
            if isinstance(parsed, list):
                # Join list items into readable text
                return '\n'.join([str(item).strip() for item in parsed if item])
        except (json.JSONDecodeError, ValueError):
            try:
                import ast
                parsed = ast.literal_eval(chunk)
                if isinstance(parsed, list):
                    return '\n'.join([str(item).strip() for item in parsed if item])
            except (ValueError, SyntaxError):
                # Remove brackets if parsing fails
                chunk = chunk.strip('[]').strip()
    
    # Clean up the chunk
    chunk = chunk.strip()
    
    # Remove excessive whitespace
    chunk = re.sub(r'\s+', ' ', chunk)
    chunk = re.sub(r'\n{3,}', '\n\n', chunk)
    
    return chunk


def assemble_prompt(query: str, context_chunks: List[str], use_llm: bool = True) -> str:
    """
    Assemble a prompt with context for LLM generation.
    
    Args:
        query: User query
        context_chunks: Retrieved context chunks
        use_llm: Whether this is for LLM (True) or dataset retrieval (False)
        
    Returns:
        Assembled prompt string
    """
    if use_llm and context_chunks:
        # Clean and format context chunks
        cleaned_chunks = []
        for chunk in context_chunks:
            # Remove any list-like structures from chunks
            cleaned_chunk = clean_context_chunk(chunk)
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)
        
        if cleaned_chunks:
            # Format context more naturally
            context = "\n\n".join([f"Relevant information:\n{chunk}" for chunk in cleaned_chunks[:2]])  # Limit to top 2
            prompt = f"{PROMPT_PREFIX}\n\n{context}\n\nBased on the above information, answer this question:\n{query}\n\nAnswer:"
        else:
            prompt = f"{PROMPT_PREFIX}\n\n{query}\n\nAnswer:"
    elif use_llm:
        prompt = f"{PROMPT_PREFIX}\n\n{query}\n\nAnswer:"
    else:
        # For dataset retrieval, return query as-is
        prompt = query
    
    return prompt


def generate_answer(prompt: str, model=None, tokenizer=None, max_new_tokens: int = 200) -> str:
    """
    Generate an answer using the LLM.
    
    Args:
        prompt: Input prompt
        model: LLM model
        tokenizer: Tokenizer
        max_new_tokens: Maximum tokens to generate (increased from 150 to 200)
        
    Returns:
        Generated answer text (only the newly generated part, not the prompt)
    """
    if not transformers_available or model is None or tokenizer is None:
        return UNCERTAIN_RESPONSE
    
    try:
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_length = inputs['input_ids'].shape[1]
        
        # Generate response with better parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,  # Nucleus sampling for better quality
            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # Reduce repetition
        )
        
        # Decode only the newly generated tokens (exclude the input prompt)
        generated_tokens = outputs[0][input_length:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up the answer
        answer = clean_llm_response(answer)
        
        # Check if answer is too short or empty (likely a generation failure)
        if not answer or len(answer.strip()) < 10:
            print(f"Warning: Generated answer too short or empty: '{answer[:50]}'")
            return ""  # Return empty string, let caller handle it
        
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""


def clean_llm_response(text: str) -> str:
    """
    Clean LLM response by removing common artifacts and formatting issues.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Cleaned response text
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove common prompt artifacts
    text = text.strip()
    
    # Remove "Context 1:", "Context 2:" etc. markers
    import re
    text = re.sub(r'Context\s+\d+:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove "Question:" markers
    text = re.sub(r'Question:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove "Answer:" markers
    text = re.sub(r'^Answer:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Fix list-like structures that are strings (e.g., '["item1", "item2"]')
    # Try to parse and format as proper text
    if text.startswith('[') and text.endswith(']'):
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # Convert list to readable format
                formatted = []
                for item in parsed:
                    if isinstance(item, str):
                        formatted.append(item.strip())
                    else:
                        formatted.append(str(item))
                text = '\n'.join(formatted)
        except (json.JSONDecodeError, ValueError):
            # If not valid JSON, try ast.literal_eval
            try:
                import ast
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    formatted = []
                    for item in parsed:
                        if isinstance(item, str):
                            formatted.append(item.strip())
                        else:
                            formatted.append(str(item))
                    text = '\n'.join(formatted)
            except (ValueError, SyntaxError):
                # If parsing fails, just remove the brackets and clean up
                text = text.strip('[]').strip()
    
    # Remove duplicate newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def check_answer_relevance(answer: str, retrieved_chunks: List[Tuple[str, float, Dict]], 
                          threshold: float = 0.3) -> Tuple[bool, float]:
    """
    Check if generated answer is relevant to retrieved context (hallucination prevention).
    
    Args:
        answer: Generated answer text
        retrieved_chunks: List of (chunk_text, score, metadata) tuples
        threshold: Minimum similarity threshold for relevance
        
    Returns:
        Tuple of (is_relevant, max_similarity_score)
    """
    if not retrieved_chunks:
        return False, 0.0
    
    # Preprocess answer
    answer_processed = preprocess_text(answer).lower()
    
    # Use TF-IDF to compare answer with retrieved chunks
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    chunk_texts = [chunk[0] for chunk in retrieved_chunks]
    if not chunk_texts:
        return False, 0.0
    
    # Create vectorizer and compute similarities
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    try:
        all_texts = [answer_processed] + [preprocess_text(chunk).lower() for chunk in chunk_texts]
        vectors = vectorizer.fit_transform(all_texts)
        
        # Compare answer with each chunk
        answer_vec = vectors[0:1]
        chunk_vecs = vectors[1:]
        
        similarities = cosine_similarity(answer_vec, chunk_vecs).flatten()
        max_similarity = float(similarities.max()) if len(similarities) > 0 else 0.0
        
        is_relevant = max_similarity >= threshold
        return is_relevant, max_similarity
    except Exception:
        return False, 0.0

