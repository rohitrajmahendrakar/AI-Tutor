## Python / Backend Developer Documentation

This document describes the **Python backend architecture** of the AI Tutor Chatbot, focusing on Flask endpoints, the chatbot orchestration logic, and integration with the RAG pipeline and database.

- **Core modules**: `app.py`, `chatbot.py`, `rag_pipeline.py`, `preprocessing.py`, `constants.py`, `database.py`
- **Supporting scripts**: `data_preprocess.py`, `fine_tune_gemma.py` (if used), `test_endpoints.py`

### High‑Level Architecture

- **Flask application (`app.py`)**
  - Hosts all HTTP routes (UI pages and API endpoints).
  - Handles user authentication via a MySQL/MariaDB backend.
  - Exposes the `/ask` endpoint for chat queries and `/upload` for document ingestion.

- **Chatbot orchestration (`chatbot.py`)**
  - Initializes and coordinates:
    - Preprocessed dataset and retrieval index (`initialize_main_dataset`).
    - Optional LLM (Gemma or DistilGPT‑2) via `transformers`.
  - Implements **hallucination‑aware response generation** through the RAG pipeline.

- **RAG pipeline (`rag_pipeline.py`)**
  - Provides modular building blocks:
    - Retrieval (`retrieve_chunks`), ranking (`rank_chunks`).
    - Prompt assembly (`assemble_prompt`).
    - LLM answer generation (`generate_answer`) and relevance checking (`check_answer_relevance`).

- **Preprocessing & indexing (`preprocessing.py`)**
  - Shared text preprocessing, chunking, and TF‑IDF index management.

### Flask Application (`app.py`)

#### App Initialization

- Loads environment variables via `python-dotenv` (`load_dotenv()`).
- Imports application components with both **package‑relative** and **standalone** fallbacks:
  - `get_chatbot_response`, `model_name_used`, `add_uploaded_file` from `chatbot.py`.
  - Database configuration and initializer from `database.py` (`init_db`, `DB_CONFIG`).
  - Path and configuration constants from `constants.py` (templates, static, dataset dirs, prompt prefix, file type support).
- Configures Flask:
  - `template_folder` and `static_folder` from `TEMPLATE_DIR` and `STATIC_DIR`.
  - `secret_key` from `FLASK_SECRET_KEY` (with a default fallback).
  - `UPLOAD_FOLDER` as `datasets/uploads`, auto‑creating the directory.
  - `MAX_CONTENT_LENGTH = 16MB` to limit upload size.

#### Database Access

- Wrapper: `get_connection()` returns a `pymysql.connect(**DB_CONFIG)` connection.
- `init_db()` is invoked on startup to ensure user tables exist.

#### Core Routes

- **`/` (home)**:
  - Renders `index.html` with:
    - `username` from `session['user']` if present.
    - `logged_in` boolean for frontend conditional rendering.

- **`/register`**:
  - `GET`: render registration form.
  - `POST`:
    - Validate **name, email, password**.
    - Hash password with `werkzeug.security.generate_password_hash`.
    - Insert into `users` table; handle:
      - `IntegrityError` (duplicate email) with user‑friendly error.
      - `OperationalError` (DB connectivity) with clear message.

- **`/login`**:
  - `GET`: render login form.
  - `POST`:
    - Fetch `name, password` from `users` table by email.
    - Verify password with `check_password_hash`.
    - Set `session['user']` and redirect to `/chat` on success.

- **`/logout`**:
  - Clear `session['user']` and redirect to `/login`.

- **`/chat`**:
  - Requires `session['user']`.
  - Renders `chat.html` with username.

- **`/ask`** (chat API):
  - Method: `POST`.
  - Parameters: `message` (required), `use_uploaded` (optional).
  - Behavior:
    - If message is empty, returns a JSON object with an error‑style message and `source="uncertain"`.
    - Otherwise calls `get_chatbot_response(user_input, use_uploaded_files=use_uploaded)` from `chatbot.py`.
    - Normalizes output via `format_chatbot_response`, which:
      - Strips technical prompt prefixes and echoed question text.
      - Handles list‑like responses (`["a", "b"]`) and deduplicates repeated sentences while preserving formatting.
    - Returns JSON with fields:
      - `response`, `source`, `dataset_source`, `similarity_score`, `is_grounded`.

- **`/upload`** (document ingestion):
  - Method: `POST`.
  - Validates presence of a file and non‑empty filename.
  - Enforces `SUPPORTED_FILE_TYPES` from `constants.py` (`.txt`, `.csv`, `.md`, `.markdown`).
  - Saves the file under `UPLOAD_FOLDER`, then calls `add_uploaded_file(file_path, filename)`:
    - On success, returns a success JSON message and file metadata.
    - On failure, returns a descriptive error.

- **`/status`**:
  - Returns a machine‑readable JSON status snapshot via `is_chatbot_ready()`:
    - `ready`, `has_dataset`, `has_llm`, `has_uploaded_files`, `dataset_size`, `mode`.

- **`/debug`**:
  - Provides deeper internal state:
    - Dataset emptiness.
    - Whether vectorizer and chunks are loaded for main and uploaded sources.

### Chatbot Orchestration (`chatbot.py`)

#### Initialization

At module import, the chatbot performs the following:

- `ensure_main_dataset_processed()`:
  - Checks for `main_dataset_processed.csv` under `datasets/processed/`.
  - If missing and raw CSV exists, runs `process_dataset_csv` to generate processed artifacts.
- `initialize_main_dataset(MAIN_DATASET_NAME)`:
  - Loads processed dataset, TF‑IDF index, and chunk metadata into in‑memory globals in `rag_pipeline.py`.
- `load_llm()`:
  - If `transformers` is available:
    - Prefer a `FINE_TUNED_MODEL_PATH` environment variable (fine‑tuned Gemma or compatible model).
    - Fallback to base `BASE_MODEL_NAME` (default `"google/gemma-2b"`).
    - Secondary fallback to lightweight `"distilgpt2"` if Gemma fails.
  - Sets `model_name_used` to describe which model is active.
- Logs a summary of initialization state (dataset size, LLM status, uploaded files).

#### Readiness Check

- `is_chatbot_ready() -> dict`:
  - Inspects global state in `rag_pipeline.py` (`_main_dataset`, `_main_vectorizer`, `_main_chunks`, `_uploaded_chunks`).
  - Computes:
    - `has_dataset`: dataset loaded + vectorizer not `None`.
    - `has_llm`: both `model` and `tokenizer` instantiated.
    - `has_uploaded_files`: non‑empty `_uploaded_chunks`.
    - `dataset_size`: number of rows in main dataset.
  - `ready` is `True` if **any** of dataset, LLM, or uploaded files is available.

#### Response Generation

- `get_chatbot_response(user_input: str, use_uploaded_files: bool = False) -> (response, source, metadata)`:
  - **Step 1 – Optional uploaded retrieval**:
    - If `use_uploaded_files=True`, calls `retrieve_chunks(query, source="uploaded", top_k=3)`.
    - If results exist:
      - Ranks them (`rank_chunks`) and takes the top chunk.
      - Returns the best chunk directly with `source="uploaded"` and `metadata` populated (`similarity_score`, `dataset_source="uploaded_files"`, `is_grounded=True`).
  - **Step 2 – Main dataset retrieval**:
    - Calls `retrieve_chunks(query, source="main", top_k=3)`.
    - If results exist:
      - Returns best chunk with `source="dataset"` and metadata (`dataset_source="main_dataset"`, `is_grounded=True`).
  - **Step 3 – LLM fallback**:
    - If `model` and `tokenizer` are available:
      - Gathers additional context with a **lower threshold**:
        - `retrieve_chunks(..., source="uploaded", threshold=0.2)` and `source="main"` similarly.
      - Assembles a prompt with `assemble_prompt(user_input, context_chunks, use_llm=True)`.
      - Calls `generate_answer(prompt, model, tokenizer)` to generate text.
      - Rejects empty or extremely short answers.
      - Applies **hallucination prevention** only if some context exists:
        - Compute best retrieval score across all context chunks.
        - If `best_context_score > 0.3`, call `check_answer_relevance` with a configurable threshold (`ANSWER_RELEVANCE_THRESHOLD`).
        - If relevance is very low (e.g., `< 0.2`), return the standardized `UNCERTAIN_RESPONSE` with `source="uncertain"`.
      - Otherwise, accept the LLM answer with `source="llm"` and `dataset_source="llm_fallback"`.
  - **Step 4 – No LLM fallback**:
    - If no LLM is available and retrieval returns nothing, returns `UNCERTAIN_RESPONSE` with `source="uncertain"`.

#### Uploaded File Integration

- `add_uploaded_file(file_path: Path, file_name: str) -> bool`:
  - Delegates to `process_uploaded_file` in `preprocessing.py`:
    - Performs text extraction/cleaning and chunking.
  - Calls `initialize_uploaded_files(chunks, metadata)` in `rag_pipeline.py` to:
    - Create/refresh a TF‑IDF index for uploaded document chunks.
  - Returns `True` on successful indexing, `False` otherwise.

### Constants and Configuration (`constants.py`)

- **Paths** (relative to project root):
  - `TEMPLATE_DIR`, `STATIC_DIR`, `DATASET_DIR`.

- **RAG/LLM behavior**:
  - `PROMPT_PREFIX`: textual preamble used in `assemble_prompt`.
  - `SIMILARITY_THRESHOLD`: base similarity threshold for retrieval (`default 0.55`).
  - `ANSWER_RELEVANCE_THRESHOLD`: relevance threshold for hallucination check (default `0.2`).
  - `UNCERTAIN_RESPONSE`: human‑readable fallback message; can be overridden via environment variable.

- **Upload support**:
  - `SUPPORTED_FILE_TYPES = ['.txt', '.csv', '.md', '.markdown']`.

### End‑to‑End Request Flow

1. **User submits a question** via the `/chat` UI.
2. **Flask `/ask` route** receives the request and forwards it to `get_chatbot_response`.
3. **RAG pipeline**:
   - Attempts retrieval from uploaded and main dataset sources.
   - If needed, assembles a context‑aware prompt and calls the LLM.
   - Applies hallucination checks based on similarity and relevance.
4. **Response formatting** in `app.py`:
   - Cleans and deduplicates the response text.
   - Returns a JSON object to the frontend.
5. **Frontend updates** the chat transcript with answer text and optional metadata.

This modular design allows Python developers to:

- Swap LLMs or retrieval strategies while preserving the HTTP contract.
- Adjust thresholds and chunking behavior for different experimental conditions.
- Extend the system with additional routes (e.g., analytics, admin, feedback collection) without disturbing the core tutor flow.




