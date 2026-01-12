import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DATASET_DIR = BASE_DIR / "datasets"

PROMPT_PREFIX = "Answer this Python/Data Science question clearly:"
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.55"))
ANSWER_RELEVANCE_THRESHOLD = float(os.getenv("ANSWER_RELEVANCE_THRESHOLD", "0.2"))  # Lowered from 0.3 to be more lenient
UNCERTAIN_RESPONSE = os.getenv(
    "UNCERTAIN_RESPONSE",
    "I am not confident enough to answer that yet. Please try another wording or consult additional resources.",
)

# Supported file types for uploads
SUPPORTED_FILE_TYPES = ['.txt', '.csv', '.md', '.markdown']

