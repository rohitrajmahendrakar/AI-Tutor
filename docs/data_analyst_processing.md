## Data Analyst & Processing

This system is centered around a curated **Python / data‑science Q&A dataset** and a unified preprocessing pipeline that prepares data for retrieval‑augmented generation (RAG).

- **Primary modules**: `preprocessing.py`, `data_preprocess.py`, `constants.py`
- **Primary assets**: `datasets/Dataset_Python_Question_Answer.csv`, `datasets/processed/main_dataset_processed.csv`, `datasets/processed/main_dataset_chunks.jsonl`, `datasets/indices/main_dataset.index`

### Objectives

- **Normalize heterogeneous raw Q&A data** into a clean, deduplicated corpus.
- **Chunk long answers** into semantically coherent segments suitable for retrieval.
- **Build TF‑IDF indices** to support fast, similarity‑based retrieval during question answering.
- **Ensure reproducibility** so that the same raw dataset always yields the same processed artifacts.

### Data Pipeline Overview

- **Raw input**
  - **Source**: `datasets/Dataset_Python_Question_Answer.csv`
  - **Expected columns**: at minimum `Question`, `Answer`.

- **Preprocessing & cleaning** (in `preprocessing.py`, `process_dataset_csv`)
  - **Loading with robust encoding fallback**:
    - Try `latin-1`, fall back to `utf-8` with error‑ignore for noisy files.
  - **De‑duplication**:
    - Remove duplicated rows based on `Question` (`drop_duplicates(subset="Question")`).
  - **Null handling**:
    - Drop rows where **either** `Question` or `Answer` is missing (`dropna(subset=["Question", "Answer"])`).
  - **Text normalization** (via `preprocess_text`):
    - Coerce non‑string values to string.
    - Strip leading/trailing whitespace.
    - Normalize internal whitespace with a regex (`\s+ -> " "`).
    - Convert questions to lowercase to stabilize vectorization; answers remain case‑preserving but cleaned.

- **Processed dataset output**
  - **File**: `datasets/processed/main_dataset_processed.csv`
  - **Properties**:
    - Cleaned, unique questions.
    - Normalized answers with consistent formatting.
    - Ready for downstream chunking and indexing.

### Answer Chunking

Long answers are split into overlapping chunks to improve semantic retrieval granularity.

- **Function**: `chunk_text` in `preprocessing.py`
- **Default parameters**:
  - **`DEFAULT_CHUNK_SIZE = 500` characters**
  - **`DEFAULT_CHUNK_OVERLAP = 50` characters**

- **Algorithm sketch**:
  - Iterate with a sliding window over the answer text.
  - Prefer to end chunks at **sentence boundaries**:
    - Within each window, search backward for the last `"."` or newline.
    - If a boundary occurs after at least 50% of the chunk, cut there.
  - Apply a fixed **overlap of 50 characters** between successive chunks.
  - Drop empty chunks at the end.

- **Rationale**:
  - Overlap helps preserve context across chunk boundaries.
  - Sentence‑aware breaking reduces mid‑sentence truncation artifacts.
  - Character‑based chunking is simple, deterministic, and fast for CSV‑sourced text.

### Index Construction

The processed and chunked dataset is turned into TF‑IDF vectors for cosine‑similarity retrieval.

- **Function**: `create_index` in `preprocessing.py`
- **Index type**: `sklearn.feature_extraction.text.TfidfVectorizer`
- **Key parameters**:
  - `stop_words="english"`
  - `max_features=5000`

- **Outputs**:
  - **Vectorizer**: learned vocabulary and IDF weights.
  - **Document vectors**: sparse TF‑IDF matrix of shape \\(N \times V\\).
  - **Persistence**:
    - Saved to `datasets/indices/main_dataset.index` via `pickle`.
    - Reloaded by `load_index` when the application starts (`initialize_main_dataset`).

### Chunk Metadata

- **File**: `datasets/processed/main_dataset_chunks.jsonl`
- **Producer**: `process_dataset_csv`
- **Record schema** (one JSON object per chunk):
  - **`dataset`**: dataset logical name (e.g., `"main_dataset"`).
  - **`question_idx`**: row index of the original question in the processed CSV.
  - **`chunk_idx`**: index of the chunk within that answer.
  - **`question`**: normalized question text.
  - **`answer`**: full normalized answer text (not just the chunk).

These metadata are used by the RAG pipeline to:

- Reconstruct the relationship between chunks and original Q&A pairs.
- Provide **traceability** from retrieved chunks back to the curated dataset.
- Enable future analysis such as **coverage**, **overlap statistics**, and **per‑topic retrieval performance**.

### Orchestration Script

- **File**: `data_preprocess.py`
- **Entry point**: `preprocess()`
- **Responsibilities**:
  - Validate the existence of the raw CSV.
  - Call `process_dataset_csv` to run the unified pipeline.
  - Log the number of processed Q&A pairs.

This script can be invoked manually (e.g., from the command line) to regenerate all processed artifacts, supporting reproducible experiments.

### Role of the Data Analyst

From a **data analyst / data engineer** perspective, the key contributions are:

- **Curation and cleaning** of Q&A pairs to ensure high‑quality ground‑truth for retrieval.
- **Design of chunking strategy** (size, overlap, sentence awareness) to balance recall and precision.
- **Design of vectorization scheme** (TF‑IDF with English stopwords, 5k features) to support fast, interpretable similarity search.
- **Creation of durable artifacts** (processed CSV, JSONL, `.index`) that can be versioned and compared across model/rule changes.

These design choices directly influence:

- Retrieval quality (which chunks are considered “similar” to a query).
- Latency (size of the vocabulary and index).
- Downstream **hallucination control**, since the RAG layer relies on similarity scores derived from this pipeline.




