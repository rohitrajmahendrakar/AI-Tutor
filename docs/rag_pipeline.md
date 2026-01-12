## RAG Pipeline Documentation

This document describes the **Retrieval‑Augmented Generation (RAG)** architecture implemented in the project and how it integrates dataset‑based retrieval, optional LLMs, and hallucination control.

- **Core module**: `rag_pipeline.py`
- **Supporting modules**: `preprocessing.py`, `chatbot.py`, `constants.py`
- **Key capabilities**:
  - Dual‑source retrieval (main curated dataset + user‑uploaded files).
  - Context‑aware prompt assembly for LLMs.
  - Relevance‑based hallucination prevention.

### Global State and Data Structures

`rag_pipeline.py` maintains a set of module‑level variables to cache dataset and index state:

- **Main dataset**:
  - `_main_dataset`: Pandas `DataFrame` of processed Q&A pairs.
  - `_main_vectorizer`: TF‑IDF vectorizer for main chunks/questions.
  - `_main_doc_vectors`: TF‑IDF matrix of main chunks/questions.
  - `_main_chunks`: list of answer chunks (or full answers as a fallback).
  - `_main_chunk_metadata`: metadata aligned with `_main_chunks`.

- **Uploaded files**:
  - `_uploaded_chunks`: list of chunks derived from uploaded documents.
  - `_uploaded_metadata`: metadata per uploaded chunk.
  - `_uploaded_vectorizer`: TF‑IDF vectorizer for uploaded chunks.
  - `_uploaded_doc_vectors`: TF‑IDF matrix for uploaded chunks.

These caches allow low‑latency retrieval without reloading from disk on every request.

### Main Dataset Initialization

- **Function**: `initialize_main_dataset(dataset_name: str = "main_dataset")`

- **Steps**:
  1. **Load processed dataset** from `datasets/processed/{dataset_name}_processed.csv`.
     - If not found, falls back to a legacy `cleaned_dataset.csv` under `datasets/`.
     - If both are missing, initializes an empty `DataFrame` with `Question` and `Answer` columns.
  2. **Load index** using `load_index(dataset_name)`:
     - Attempts to load a pickled `(vectorizer, doc_vectors)` from `datasets/indices/{dataset_name}.index`.
  3. **Load chunks metadata** from `datasets/processed/{dataset_name}_chunks.jsonl`:
     - For each JSONL record, appends `meta["answer"]` to `_main_chunks` and `meta` itself to `_main_chunk_metadata`.
  4. **Fallback behavior**:
     - If `_main_vectorizer` or `_main_doc_vectors` is `None`, builds a **fresh TF‑IDF index over questions**:
       - `TfidfVectorizer(stop_words="english", max_features=5000)` fit on `Question` column.
     - If no chunk metadata is available, uses **answers as chunks**, building metadata on the fly per (question, answer) pair.

This design separates disk‑level artifacts (CSV, JSONL, `.index`) from in‑memory structures used at runtime.

### Uploaded Files Initialization

- **Function**: `initialize_uploaded_files(chunks: List[str], metadata: List[Dict], index_name: str = "uploaded_files")`

- **Behavior**:
  - If `chunks` is empty:
    - Clears uploaded‑files state (`_uploaded_chunks`, `_uploaded_metadata`, `_uploaded_vectorizer`, `_uploaded_doc_vectors`).
  - Otherwise:
    - Caches `chunks` and `metadata`.
    - Trains a new TF‑IDF vectorizer over the chunks and computes `doc_vectors`.
    - Saves the uploaded index to `datasets/indices/{index_name}.index`.

Uploaded documents thus become a **first‑class retrieval source** with their own independent index.

### Retrieval

- **Function**: `retrieve_chunks(query: str, source: str = "main", top_k: int = 3, threshold: float = SIMILARITY_THRESHOLD)`

- **Steps**:
  1. **Preprocess query**:
     - Calls `preprocess_text(query)` to normalize whitespace and ensure consistent input.
     - Converts to lowercase for vectorization.
  2. **Select source**:
     - For `source="main"`:
       - Uses `_main_vectorizer`, `_main_doc_vectors`, `_main_chunks`, `_main_chunk_metadata`.
       - If no chunks, may fallback to building chunks directly from `_main_dataset["Answer"]`.
     - For `source="uploaded"`:
       - Uses `_uploaded_vectorizer`, `_uploaded_doc_vectors`, `_uploaded_chunks`, `_uploaded_metadata`.
  3. **Guard rails**:
     - If no vectorizer, no doc vectors, or no chunks are available, returns an empty list.
  4. **Compute similarities**:
     - Transforms the query into TF‑IDF space.
     - Computes cosine similarity against document vectors.
     - Sorts indices by descending similarity and selects `top_k`.
  5. **Apply threshold**:
     - For each candidate, includes it only if `score >= threshold`.

- **Return format**:
  - List of `(chunk_text: str, similarity_score: float, metadata: Dict)` tuples, sorted by similarity.

### Ranking

- **Function**: `rank_chunks(chunks_with_scores: List[Tuple[str, float, Dict]])`

- **Behavior**:
  - Sorts the input list by similarity score in **descending order**.
  - Designed to be easily extended with additional criteria (e.g., diversity, recency).

### Context Cleaning and Prompt Assembly

- **Context cleaning (`clean_context_chunk`)**:
  - Handles chunks that might be list‑like strings (e.g., `'["item1", "item2"]'`).
  - Attempts to parse them as JSON or Python lists and join items with newlines.
  - Removes excessive whitespace and normalizes formatting.

- **Prompt assembly (`assemble_prompt`)**:
  - Signature: `assemble_prompt(query: str, context_chunks: List[str], use_llm: bool = True) -> str`
  - **LLM mode** (`use_llm=True`):
    - Cleans each context chunk.
    - Formats up to the top 2 chunks as blocks of the form:
      - `"Relevant information:\n{chunk}"`
    - Prepares a final prompt:
      - Combines `PROMPT_PREFIX` from `constants.py`, context blocks, and the user query.
      - Appends `"Answer:"` to steer the LLM toward concise answers.
  - **Dataset mode** (`use_llm=False`):
    - Returns the raw query unchanged (used by pure retrieval pipelines).

### Answer Generation

- **Function**: `generate_answer(prompt: str, model=None, tokenizer=None, max_new_tokens: int = 200) -> str`

- **Behavior**:
  - If `transformers` is unavailable or model/tokenizer is `None`, returns `UNCERTAIN_RESPONSE`.
  - Otherwise:
    - Tokenizes the prompt with truncation at 512 tokens.
    - Generates output using:
      - `max_new_tokens` up to 200.
      - Sampling parameters:
        - `do_sample=True`, `temperature=0.7`, `top_p=0.9`.
      - `pad_token_id` and `eos_token_id` configured from tokenizer.
    - Extracts **only the newly generated tokens** (excluding the prompt).
    - Decodes and cleans the answer using `clean_llm_response`:
      - Removes prompt artifacts (`Context 1:`, `Question:`, `Answer:`).
      - Attempts to normalize list‑like responses.
      - Collapses excessive newlines and trims whitespace.
    - If the answer is empty or extremely short, returns an empty string so callers can handle fallback logic.

### Hallucination Prevention

- **Function**: `check_answer_relevance(answer: str, retrieved_chunks: List[Tuple[str, float, Dict]], threshold: float = 0.3) -> (bool, float)`

- **Design**:
  - Uses a fresh TF‑IDF vectorizer to compare the **generated answer** with the **retrieved context chunks**.
  - Preprocesses both answer and chunks with `preprocess_text`.
  - Computes cosine similarity between answer vector and each chunk vector.
  - Returns:
    - `is_relevant`: whether the maximum similarity over all chunks exceeds the given threshold.
    - `max_similarity`: the numeric maximum similarity score.

- **Interaction with `chatbot.py`**:
  - When context exists and the best retrieval score is reasonably high (e.g., >0.3):
    - `chatbot.py` calls `check_answer_relevance` with `ANSWER_RELEVANCE_THRESHOLD`.
    - If `max_similarity` is very low (e.g., `< 0.2`), the answer is treated as hallucinated and replaced by `UNCERTAIN_RESPONSE`.
  - If context is weak or missing:
    - The system becomes **more lenient**, allowing LLM answers but labeling them as ungrounded (`is_grounded=False`).

### RAG Flow in Context

In the full system:

1. **Dataset preparation**:
   - `preprocessing.py` transforms raw CSV into processed CSV, chunks, and indices.
2. **Initialization**:
   - `chatbot.py` calls `initialize_main_dataset` once at startup.
3. **Query handling**:
   - User queries arrive at `/ask` and are forwarded to `get_chatbot_response`.
4. **Retrieval**:
   - `retrieve_chunks` is used for both main dataset and uploaded files.
5. **Generation**:
   - If retrieval fails or is insufficient, `assemble_prompt` + `generate_answer` produce an LLM answer.
6. **Relevance checking**:
   - `check_answer_relevance` validates that the answer is consistent with retrieved context when context is strong.
7. **Response**:
   - The final response is labeled with a `source` and `is_grounded` flag for transparency.

### Research and Extension Points

The RAG implementation exposes several **axes of experimentation**:

- **Indexing**:
  - Alternative vector representations (e.g., dense embedding models instead of TF‑IDF).
  - Different chunk sizes and overlap strategies.
- **Retrieval logic**:
  - Multi‑stage retrieval (e.g., BM25 followed by embedding reranking).
  - Hybrid scoring that combines lexical and semantic similarity.
- **Prompt design**:
  - Alternative `PROMPT_PREFIX` formulations targeting different pedagogical styles.
  - Explicit citation of supporting chunks in the prompt and final answer.
- **Hallucination control**:
  - More nuanced thresholds based on topic, question type, or user confidence preferences.
  - Integration of **answerability classifiers** or **consistency checks** across multiple retrieved chunks.

These components and design decisions form a solid foundation for a research paper focused on **transparent, grounded AI tutoring** with retrieval‑augmented generation.




