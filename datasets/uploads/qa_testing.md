## QA & Testing Documentation

Quality assurance for this AI tutor spans **unit tests, integration tests, and system‑level evaluation scripts**, all designed to validate both correctness and behavior of the RAG pipeline and Flask backend.

- **Test suite**: `tests/` directory
  - `test_rag_pipeline.py`
  - `test_chatbot.py`
  - `test_chatbot_integration.py`
  - `evaluate_system.py`
  - `evaluation_set.json`, `test_set.json`
- **API tests**: `test_endpoints.py`

### Testing Strategy Overview

The testing approach covers:

- **Low‑level correctness**: preprocessing, chunking, retrieval, ranking, prompt assembly, hallucination checks.
- **Integration behavior**: end‑to‑end chatbot responses, uploaded file indexing, source labeling semantics.
- **Web/API surface**: Flask routes, authentication flows, and `/ask` contract.
- **System performance & reliability**: latency statistics, source usage patterns, grounding rates.

### RAG Pipeline Tests (`tests/test_rag_pipeline.py`)

#### Preprocessing Tests

- **`test_preprocess_text_basic`**:
  - Verifies that whitespace trimming and normalization behave as expected (e.g., `"  Hello   World  "` → `"Hello World"`).
- **`test_preprocess_text_normalize_whitespace`**:
  - Ensures newlines and tabs are collapsed into single spaces while preserving token order.

#### Chunking Tests

- **`test_chunk_text_small`**:
  - Guarantees that small text below the chunk size remains a single chunk with no modifications.
- **`test_chunk_text_large`**:
  - Confirms that large text is split into multiple chunks.
  - Validates that chunks are non‑empty and overlap is applied.

#### Retrieval and Ranking Tests

- **Setup**:
  - Uses artificial test chunks such as:
    - `"Python is a programming language"`
    - `"Data science uses Python for analysis"`
  - Initializes the uploaded‑files index via `initialize_uploaded_files`.

- **`test_retrieve_chunks_non_empty`**:
  - Ensures that a relevant query (e.g., `"Python programming"`) returns at least one result.
  - Checks the structure of results `(chunk_text, similarity_score, metadata)`.

- **`test_retrieve_chunks_threshold`**:
  - Tests behavior under a **high similarity threshold**:
    - For an unrelated query with `threshold=0.8`, any returned result must meet the threshold.

- **`test_retrieve_chunks_empty_query`**:
  - Ensures the system handles empty queries gracefully without crashing.

- **`test_rank_chunks`**:
  - Confirms that `rank_chunks` sorts by similarity score descending.

#### Prompt Assembly Tests

- **`test_assemble_prompt_with_context`**:
  - Verifies that both query and context strings appear in the constructed prompt when `use_llm=True`.
- **`test_assemble_prompt_without_context`**:
  - Confirms the query is included even when no context is provided.
- **`test_assemble_prompt_dataset_mode`**:
  - Ensures dataset retrieval mode (`use_llm=False`) returns the query unchanged.

#### Hallucination Prevention Tests

- **`test_check_answer_relevance_relevant`**:
  - Uses an answer about Python and data science with retrieved Python‑related chunks.
  - Checks that relevance scoring returns a numeric similarity within [0, 1].

- **`test_check_answer_relevance_irrelevant`**:
  - Uses a cooking‑related answer against Python chunks.
  - Ensures the function still returns well‑formed outputs and can detect low similarity when using a stricter threshold.

- **`test_check_answer_relevance_empty_chunks`**:
  - Validates behavior when `retrieved_chunks` is empty (should return `(False, 0.0)`).

#### Paraphrase Robustness & Latency

- **`test_paraphrased_question_handling`**:
  - Checks that multiple paraphrases of “What is Python?” still retrieve reasonable chunks from the uploaded index.

- **`test_no_context_handling`**:
  - Verifies system behavior when no chunk meets a high similarity threshold.

- **`test_retrieval_speed`**:
  - Empirically ensures retrieval completes within a **sub‑second** budget (e.g., <1 second), providing a basic latency bound.

### Chatbot & Integration Tests

#### Integration Tests (`tests/test_chatbot_integration.py`)

- **`test_get_response_structure`**:
  - Calls `get_chatbot_response("What is Python?")`.
  - Validates:
    - Response is a string.
    - `source` is one of `"dataset"`, `"uploaded"`, `"llm"`, `"uncertain"`.
    - Metadata dictionary contains `similarity_score`, `dataset_source`, `is_grounded`.

- **`test_uploaded_file_indexing`**:
  - Creates a temporary `.txt` file describing Python and data science.
  - Passes the file to `add_uploaded_file`.
  - Ensures indexing succeeds and cleans up the file afterwards.

- **`test_unsupported_file_type`**:
  - Asserts that unsupported uploads (e.g., `.pdf`) raise a `ValueError` in `process_uploaded_file`.

- **`test_empty_upload_handling`**:
  - Tests ingestion of an empty text file to make sure the pipeline returns valid (possibly empty) `chunks` and `metadata` lists.

- **`test_correct_source_labeling`**:
  - Ensures `get_chatbot_response`’s `source` aligns with `metadata["dataset_source"]` semantics and similarity thresholds when `source=="uncertain"`.

#### Flask Endpoint & Flow Tests (`test_endpoints.py`)

- **Environment setup**:
  - Uses `.env` to configure DB connection and sets `FLASK_SECRET_KEY` for session handling.
  - Connects to a **test database** (`test_ai_tutor_db`) and initializes a `users` table dedicated to tests.

- **Route tests**:
  - `/`:
    - For guests: expects landing content (e.g., “Transparent learning workflow”).
    - For logged‑in users: expects UI elements such as “Try the tutor”.
  - `/register`:
    - `GET`: renders registration page.
    - `POST` success: creates user and redirects to `/login`.
    - `POST` duplicate email: stays on page and shows an error.
    - `POST` missing fields: stays on page and shows “All fields are required”.
  - `/login`:
    - `GET`: loads login page.
    - `POST` success: redirects to `/chat`.
    - `POST` with invalid credentials or wrong password: shows “Invalid credentials”.
  - `/logout`:
    - Redirects to `/login` and clears session.
  - `/chat`:
    - Requires login; unauthenticated users are redirected to `/login`.
    - Authenticated users see their username in the page.
  - `/ask`:
    - Without `message`: returns a JSON response with `source="uncertain"`.
    - With `message`: returns JSON with `response` and a valid `source`.
  - `/status`:
    - Returns readiness JSON containing `status` and `mode` fields.
  - **End‑to‑end user flow**:
    - Register → login → access `/chat` → post an `/ask` query → logout → verify access control.

### Evaluation Scripts

#### Quick API Sanity Check (`tests/evaluate_system.py` at repo root)

- Sends a small set of predefined questions to the running Flask server (`/ask` at `http://127.0.0.1:5001/ask`).
- Measures latency and compares observed `source` with expected behavior (e.g., dataset vs LLM).
- Prints a short **accuracy** summary for dataset‑backed questions and basic latency statistics.

#### Comprehensive Evaluation (`tests/evaluate_system.py` in tests/)

- Loads `evaluation_set.json`, which contains a structured evaluation set with:
  - `id`, `question`, `topic`, `category`, `expected_source`.
- For each question:
  - Sends a request to `/ask`, capturing:
    - `source_used`, `similarity_score`, `is_grounded`, `dataset_source`.
    - `latency_ms`, `response_length`, and a response preview.
  - Annotates whether context was retrieved and whether the actual source matches `expected_source`.

- **Report generation**:
  - Builds a Pandas `DataFrame` with per‑question metrics.
  - Outputs:
    - Overall success/error rates.
    - Distribution of `source_used` (dataset, uploaded, llm, uncertain).
    - Context retrieval rate and similarity score statistics.
    - Source accuracy where expectations are specified.
    - Grounded vs fallback answer rates.
    - Latency statistics (overall and per source).
    - Topic‑level breakdown.
  - Saves:
    - `evaluation_results.csv`: per‑question metrics.
    - `evaluation_summary.json`: aggregated summary metrics.

### QA Role Perspective

From a QA engineering perspective, this project supports:

- **Deterministic unit tests** verifying core text and retrieval logic.
- **Integration tests** that confirm the chatbot behaves correctly across module boundaries (preprocessing → RAG → Flask).
- **User‑flow tests** that mimic realistic usage, including registration, login, and tutoring sessions.
- **Quantitative evaluation tooling** to analyze retrieval quality, grounding behavior, and system latency under realistic loads.

These elements, taken together, provide rich material for describing **experimental methodology**, **evaluation metrics**, and **limitations** in a research paper.




