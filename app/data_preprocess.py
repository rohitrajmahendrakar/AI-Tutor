"""
Dataset preprocessing script using the unified preprocessing pipeline.
"""
import pandas as pd

try:
    from .preprocessing import process_dataset_csv
    from .constants import DATASET_DIR
except ImportError:
    from preprocessing import process_dataset_csv
    from constants import DATASET_DIR

RAW_DATASET = DATASET_DIR / "Dataset_Python_Question_Answer.csv"
MAIN_DATASET_NAME = "main_dataset"


def preprocess():
    """
    Process the main dataset using the unified preprocessing pipeline.
    This ensures consistency with other data processing modules.
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    if not RAW_DATASET.exists():
        print(f"Raw dataset not found: {RAW_DATASET}")
        return
    
    try:
        print("Processing dataset with unified pipeline...")
        df = process_dataset_csv(RAW_DATASET, MAIN_DATASET_NAME)
        print(f"Dataset processing complete. Processed {len(df)} Q&A pairs.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    preprocess()
