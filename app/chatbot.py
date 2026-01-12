"""
Chatbot module using unified preprocessing and modular RAG pipeline.
"""
import os
from pathlib import Path

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    transformers_available = True
except ImportError:
    print("Transformers not found. Using retrieval-only mode.")
    transformers_available = False

try:
    from .preprocessing import process_dataset_csv, PROCESSED_DIR
    from .rag_pipeline import (
        initialize_main_dataset, initialize_uploaded_files,
        retrieve_chunks, rank_chunks, assemble_prompt, generate_answer,
        check_answer_relevance
    )
    from .constants import (
        DATASET_DIR, SIMILARITY_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD,
        UNCERTAIN_RESPONSE, SUPPORTED_FILE_TYPES
    )
except ImportError:
    from preprocessing import process_dataset_csv, PROCESSED_DIR
    from rag_pipeline import (
        initialize_main_dataset, initialize_uploaded_files,
        retrieve_chunks, rank_chunks, assemble_prompt, generate_answer,
        check_answer_relevance
    )
    from constants import (
        DATASET_DIR, SIMILARITY_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD,
        UNCERTAIN_RESPONSE, SUPPORTED_FILE_TYPES
    )

RAW_DATASET_PATH = DATASET_DIR / "Dataset_Python_Question_Answer.csv"
MAIN_DATASET_NAME = "main_dataset"

# Global LLM state
model, tokenizer = None, None
model_name_used = "Retrieval-only"


def ensure_main_dataset_processed():
    """Ensure the main dataset is processed using the unified pipeline."""
    processed_path = PROCESSED_DIR / f"{MAIN_DATASET_NAME}_processed.csv"
    
    if processed_path.exists():
        print(f"Main dataset already processed: {processed_path.name}")
        return
    
    if not RAW_DATASET_PATH.exists():
        print("Raw dataset not found. Chatbot will run without retrieval data.")
        return
    
    try:
        print("Processing main dataset with unified pipeline...")
        process_dataset_csv(RAW_DATASET_PATH, MAIN_DATASET_NAME)
        print("Main dataset processing complete.")
    except Exception as exc:
        print(f"Failed to process main dataset: {exc}")


def load_llm():
    """Load the LLM model."""
    global model, tokenizer, model_name_used
    if not transformers_available:
        return
    
    fine_tuned_path = os.getenv("FINE_TUNED_MODEL_PATH")
    if fine_tuned_path and Path(fine_tuned_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
        model_name_used = Path(fine_tuned_path).name
        print(f"Chatbot running with fine-tuned model '{model_name_used}'.")
        return
    
    try:
        base_model = os.getenv("BASE_MODEL_NAME", "google/gemma-2b")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model)
        model_name_used = base_model
    except Exception as exc:
        print(f"Gemma model access failed: {exc}")
        fallback_model = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(fallback_model)
        model_name_used = fallback_model
    
    print(f"Chatbot running in {model_name_used} mode.")


def is_chatbot_ready() -> dict:
    """
    Check if chatbot is ready to answer questions.
    
    Returns:
        Dictionary with status information
    """
    try:
        from .rag_pipeline import _main_dataset, _main_vectorizer, _main_chunks, _uploaded_chunks
    except ImportError:
        from rag_pipeline import _main_dataset, _main_vectorizer, _main_chunks, _uploaded_chunks
    
    has_dataset = (
        _main_dataset is not None and 
        not _main_dataset.empty and 
        _main_vectorizer is not None
    )
    
    has_llm = model is not None and tokenizer is not None
    
    has_uploaded_files = _uploaded_chunks is not None and len(_uploaded_chunks) > 0
    
    dataset_size = len(_main_dataset) if _main_dataset is not None and not _main_dataset.empty else 0
    
    # Chatbot is ready if it has dataset, LLM, or uploaded files
    ready = has_dataset or has_llm or has_uploaded_files
    
    return {
        "ready": ready,
        "has_dataset": has_dataset,
        "has_llm": has_llm,
        "has_uploaded_files": has_uploaded_files,
        "dataset_size": dataset_size,
        "model_name": model_name_used
    }


# Initialize on module load
print("=" * 60)
print("Initializing AI Tutor Chatbot...")
print("=" * 60)

ensure_main_dataset_processed()
initialize_main_dataset(MAIN_DATASET_NAME)
load_llm()

# Print initialization status
status = is_chatbot_ready()
print(f"\nChatbot initialization complete:")
print(f"  - Dataset available: {status['has_dataset']} ({status['dataset_size']} Q&A pairs)")
print(f"  - LLM available: {status['has_llm']} ({status['model_name']})")
print(f"  - Uploaded files: {status['has_uploaded_files']}")
print(f"  - Ready: {status['ready']}")
print("=" * 60 + "\n")


def get_chatbot_response(user_input: str, use_uploaded_files: bool = False) -> tuple:
    """
    Get chatbot response with hallucination prevention.
    
    Args:
        user_input: User query
        use_uploaded_files: Whether to search uploaded files first
        
    Returns:
        Tuple of (response_text, source_label, metadata_dict)
        source_label can be: "dataset", "uploaded", "llm", "uncertain"
        metadata_dict contains: similarity_score, dataset_source, is_grounded
    """
    metadata = {
        "similarity_score": 0.0,
        "dataset_source": None,
        "is_grounded": False
    }
    
    # Try uploaded files first if requested
    if use_uploaded_files:
        uploaded_results = retrieve_chunks(user_input, source="uploaded", top_k=3)
        if uploaded_results:
            ranked_chunks = rank_chunks(uploaded_results)
            best_chunk, best_score, best_meta = ranked_chunks[0]
            
            metadata["similarity_score"] = best_score
            metadata["dataset_source"] = "uploaded_files"
            metadata["is_grounded"] = True
            
            print(f"Retrieved from uploaded files (score {best_score:.2f}) for '{user_input}'.")
            return best_chunk, "uploaded", metadata
    
    # Try main dataset
    main_results = retrieve_chunks(user_input, source="main", top_k=3)
    if main_results:
        ranked_chunks = rank_chunks(main_results)
        best_chunk, best_score, best_meta = ranked_chunks[0]
        
        metadata["similarity_score"] = best_score
        metadata["dataset_source"] = "main_dataset"
        metadata["is_grounded"] = True
        
        print(f"Retrieved from main dataset (score {best_score:.2f}) for '{user_input}'.")
        return best_chunk, "dataset", metadata
    
    # Fallback to LLM generation
    if model and tokenizer:
        # Try to retrieve any context for RAG (use lower threshold for context gathering)
        all_results = []
        if use_uploaded_files:
            all_results.extend(retrieve_chunks(user_input, source="uploaded", top_k=2, threshold=0.2))
        all_results.extend(retrieve_chunks(user_input, source="main", top_k=2, threshold=0.2))
        
        context_chunks = [chunk[0] for chunk in all_results[:2]]  # Top 2 chunks
        
        prompt = assemble_prompt(user_input, context_chunks, use_llm=True)
        llm_answer = generate_answer(prompt, model, tokenizer)
        
        # Check if answer is empty or invalid
        if not llm_answer or len(llm_answer.strip()) < 5:
            metadata["dataset_source"] = "none"
            print(f"LLM generated empty response for '{user_input}'. Answer: '{llm_answer[:50] if llm_answer else 'None'}'")
            return UNCERTAIN_RESPONSE, "uncertain", metadata
        
        # Clean the answer - remove any UNCERTAIN_RESPONSE text if it leaked in
        llm_answer = llm_answer.strip()
        if UNCERTAIN_RESPONSE.lower() in llm_answer.lower() and len(llm_answer) < len(UNCERTAIN_RESPONSE) + 20:
            # Answer is just the uncertain response, reject it
            metadata["dataset_source"] = "none"
            print(f"LLM generated uncertain response text for '{user_input}'.")
            return UNCERTAIN_RESPONSE, "uncertain", metadata
        
        # Hallucination prevention: only check relevance if we have good context
        # If no context available, allow LLM to answer (that's the point of fallback)
        if all_results and len(all_results) > 0:
            # Check relevance, but use a more lenient threshold
            # Only reject if we have good context AND answer is clearly irrelevant
            best_context_score = max([chunk[1] for chunk in all_results])
            
            # Only apply strict check if we have reasonably good context (score > 0.3)
            if best_context_score > 0.3:
                is_relevant, relevance_score = check_answer_relevance(
                    llm_answer, all_results, threshold=ANSWER_RELEVANCE_THRESHOLD
                )
                metadata["similarity_score"] = relevance_score
                metadata["is_grounded"] = is_relevant
                
                # Only reject if relevance is very low (< 0.2) when we have good context
                if not is_relevant and relevance_score < 0.2:
                    print(f"LLM answer not grounded (relevance {relevance_score:.2f}, context {best_context_score:.2f}) for '{user_input}'.")
                    return UNCERTAIN_RESPONSE, "uncertain", metadata
            else:
                # Weak context, be more lenient - allow the answer
                metadata["similarity_score"] = best_context_score
                metadata["is_grounded"] = False
        else:
            # No context available - this is a pure LLM fallback, allow it
            metadata["similarity_score"] = 0.0
            metadata["is_grounded"] = False
        
        metadata["dataset_source"] = "llm_fallback"
        print(f"Generated response via {model_name_used} for '{user_input}'.")
        return llm_answer, "llm", metadata
    
    # No LLM available and no retrieval results
    metadata["dataset_source"] = "none"
    print(f"Returning uncertainty notice for '{user_input}'.")
    return UNCERTAIN_RESPONSE, "uncertain", metadata


def add_uploaded_file(file_path: Path, file_name: str) -> bool:
    """
    Process and index an uploaded file.
    
    Args:
        file_path: Path to the uploaded file
        file_name: Original filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from .preprocessing import process_uploaded_file
    except ImportError:
        from preprocessing import process_uploaded_file
    
    try:
        chunks, metadata = process_uploaded_file(file_path, file_name)
        if chunks:
            initialize_uploaded_files(chunks, metadata)
            print(f"Successfully indexed uploaded file: {file_name} ({len(chunks)} chunks)")
            return True
        return False
    except Exception as e:
        print(f"Error processing uploaded file {file_name}: {e}")
        return False
