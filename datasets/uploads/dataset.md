## Dataset Documentation

This document describes the **datasets** used by the AI Tutor Chatbot, including their structure, processing artifacts, and how they support the RAG pipeline.

- **Root dataset directory**: `datasets/`
  - `Dataset_Python_Question_Answer.csv`
  - `processed/`
    - `main_dataset_processed.csv`
    - `main_dataset_chunks.jsonl`
  - `indices/`
    - `main_dataset.index`
  - `uploads/`

### Core Q&A Dataset

- **File**: `datasets/Dataset_Python_Question_Answer.csv`
- **Semantic domain**:
  - Python programming fundamentals.
  - Data science and machine learning concepts.
  - Practical tasks such as using pandas, linear regression, and related workflows.

- **Structure**:
  - Minimum required columns:
    - **`Question`**: natural‑language question text.
    - **`Answer`**: explanatory or procedural answer text.
  - Additional columns (if present) are ignored by the current preprocessing pipeline.

### Processed Dataset

- **File**: `datasets/processed/main_dataset_processed.csv`
- **Producer**:
  - `preprocessing.process_dataset_csv` called either from:
    - `data_preprocess.py` (manual/CLI preprocessing), or
    - `chatbot.ensure_main_dataset_processed` during application startup.

- **Transformations**:
  - **Encoding‑robust loading**:
    - Attempts to read with `latin-1`, falling back to `utf-8` with error suppression.
  - **Cleaning**:
    - Row‑level deduplication based on `Question`.
    - Dropping rows where either `Question` or `Answer` is null.
  - **Text normalization**:
    - `Question`:
      - Cast to string, run through `preprocess_text`, then lowercased.
    - `Answer`:
      - Cast to string and run through `preprocess_text` (whitespace normalization).

- **Intended use**:
  - Serves as the canonical source for:
    - Retrieval training (TF‑IDF index).
    - Chunk generation.
    - Evaluation of dataset‑sourced answers.

### Chunk Metadata

- **File**: `datasets/processed/main_dataset_chunks.jsonl`
- **Content**:
  - Each line is a JSON object describing a **single chunk** of an answer.
  - Schema:
    - `dataset`: logical name (e.g., `"main_dataset"`).
    - `question_idx`: row index in the processed CSV.
    - `chunk_idx`: index of this chunk within its answer.
    - `question`: normalized question text.
    - `answer`: full normalized answer text (not just the chunk snippet).

- **Generation**:
  - From `preprocessing.process_dataset_csv`:
    - For each row, answers are segmented with `chunk_text` into overlapping chunks.
    - For every chunk, a metadata record is appended and later serialized as a JSONL line.

- **Usage**:
  - Loaded by `rag_pipeline.initialize_main_dataset` to construct:
    - `_main_chunks`: list of chunk texts used in retrieval.
    - `_main_chunk_metadata`: enriched records used for interpretability and tracing.

### TF‑IDF Index

- **File**: `datasets/indices/main_dataset.index`
- **Type**:
  - Pickled tuple: `(TfidfVectorizer, sparse_matrix_of_document_vectors)`.
- **Producer**:
  - `preprocessing.create_index` invoked from `process_dataset_csv`.

- **Role in the system**:
  - Used by `rag_pipeline.retrieve_chunks` to:
    - Map normalized queries into TF‑IDF space.
    - Compute cosine similarity against answer chunks or questions.
  - Provides a **lexical retrieval baseline** sufficient for educational Q&A tasks without requiring heavy embedding models.

### Uploaded Datasets

- **Directory**: `datasets/uploads/`
- **Population**:
  - When users upload files through the `/upload` endpoint in the web UI:
    - Flask saves the raw file under `datasets/uploads/`.
    - `chatbot.add_uploaded_file` and `preprocessing.process_uploaded_file`:
      - Extract text contents (for `.txt`, `.csv`, `.md/.markdown`).
      - Preprocess and chunk them.
      - Construct metadata and call `initialize_uploaded_files`.

- **Indexing & metadata**:
  - A separate TF‑IDF index is created for uploaded chunks and stored under `datasets/indices/` (e.g., `uploaded_files.index`).
  - In‑memory state:
    - `_uploaded_chunks`, `_uploaded_metadata`, `_uploaded_vectorizer`, `_uploaded_doc_vectors` in `rag_pipeline.py`.

- **Retrieval behavior**:
  - If the user explicitly enables uploaded‑file retrieval (`use_uploaded=true`), the system:
    - Queries the uploaded index first.
    - Returns top chunks as answers when similarity is above threshold.

### Evaluation Datasets

- **Files** (under `tests/`):
  - `test_set.json`, `evaluation_set.json`.

- **Usage**:
  - Provide structured test questions with:
    - `id`, `question`, `topic`, `category`, `expected_source`.
  - Consumed by:
    - `tests/evaluate_system.py` to generate quantitative evaluation reports.

### Dataset Considerations for Research

From a research perspective, the dataset layer provides:

- A **curated, domain‑specific Q&A corpus** enabling focused evaluation on Python/data‑science topics.
- Ground truth for analyzing:
  - Retrieval quality (which chunks are selected for which queries).
  - Answer grounding and hallucination.
  - The trade‑off between dataset coverage and LLM fallback behavior.

Potential extensions suitable for a paper include:

- Adding **difficulty labels**, **topic tags**, or **pedagogical intents** (e.g., concept explanation vs procedural guidance).
- Evaluating **coverage gaps** where the system falls back to LLM generation due to missing dataset entries.
- Measuring the impact of **dataset size and composition** on retrieval accuracy and student learning outcomes.




