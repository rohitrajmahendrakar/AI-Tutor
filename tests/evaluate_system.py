"""
Evaluation script for the AI Tutor Chatbot system.
Generates a comprehensive evaluation report with metrics.
"""
import json
import time
import statistics
import requests
from pathlib import Path
from typing import Dict, List
import pandas as pd

API_URL = "http://127.0.0.1:5001/ask"

# Load evaluation set
EVAL_SET_PATH = Path(__file__).parent / "evaluation_set.json"


def load_evaluation_set() -> List[Dict]:
    """Load evaluation questions from JSON file."""
    with open(EVAL_SET_PATH, 'r') as f:
        data = json.load(f)
    return data["evaluation_questions"]


def evaluate_question(question_data: Dict) -> Dict:
    """
    Evaluate a single question and return metrics.
    
    Returns:
        Dictionary with evaluation metrics
    """
    question = question_data["question"]
    expected_source = question_data.get("expected_source")
    
    start_time = time.perf_counter()
    
    try:
        response = requests.post(
            API_URL,
            data={"message": question},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        actual_source = data.get("source", "unknown")
        similarity_score = data.get("similarity_score", 0.0)
        is_grounded = data.get("is_grounded", False)
        dataset_source = data.get("dataset_source", "unknown")
        response_text = data.get("response", "")
        
        # Check if context was retrieved
        context_retrieved = similarity_score > 0 or actual_source in ["dataset", "uploaded"]
        
        # Determine if answer was grounded or fallback
        answer_type = "grounded" if is_grounded else "fallback"
        
        # Check if source matches expectation
        source_match = (expected_source is None or actual_source == expected_source)
        
        return {
            "question_id": question_data["id"],
            "question": question,
            "topic": question_data.get("topic", "Unknown"),
            "category": question_data.get("category", "Unknown"),
            "context_retrieved": context_retrieved,
            "source_used": actual_source,
            "expected_source": expected_source,
            "source_match": source_match,
            "similarity_score": similarity_score,
            "answer_type": answer_type,
            "is_grounded": is_grounded,
            "dataset_source": dataset_source,
            "latency_ms": round(latency_ms, 2),
            "response_length": len(response_text),
            "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "question_id": question_data["id"],
            "question": question,
            "topic": question_data.get("topic", "Unknown"),
            "category": question_data.get("category", "Unknown"),
            "error": str(e),
            "context_retrieved": False,
            "source_used": "error",
            "latency_ms": None
        }
    except Exception as e:
        return {
            "question_id": question_data["id"],
            "question": question,
            "error": str(e),
            "context_retrieved": False,
            "source_used": "error",
            "latency_ms": None
        }


def generate_evaluation_report(results: List[Dict]) -> pd.DataFrame:
    """
    Generate a comprehensive evaluation report.
    
    Returns:
        DataFrame with evaluation results
    """
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("AI TUTOR CHATBOT EVALUATION REPORT")
    print("="*80)
    
    # Overall statistics
    print("\n--- OVERALL STATISTICS ---")
    total_questions = len(df)
    successful_questions = len(df[df["source_used"] != "error"])
    print(f"Total questions: {total_questions}")
    print(f"Successful responses: {successful_questions}")
    print(f"Error rate: {((total_questions - successful_questions) / total_questions * 100):.1f}%")
    
    # Source distribution
    print("\n--- SOURCE DISTRIBUTION ---")
    if "source_used" in df.columns:
        source_counts = df["source_used"].value_counts()
        for source, count in source_counts.items():
            percentage = (count / total_questions) * 100
            print(f"{source}: {count} ({percentage:.1f}%)")
    
    # Context retrieval statistics
    print("\n--- CONTEXT RETRIEVAL ---")
    context_retrieved_count = df["context_retrieved"].sum() if "context_retrieved" in df.columns else 0
    print(f"Questions with retrieved context: {context_retrieved_count} ({context_retrieved_count/total_questions*100:.1f}%)")
    
    # Similarity scores
    if "similarity_score" in df.columns:
        valid_scores = df[df["similarity_score"] > 0]["similarity_score"]
        if len(valid_scores) > 0:
            print("\n--- SIMILARITY SCORES ---")
            print(f"Mean: {valid_scores.mean():.3f}")
            print(f"Median: {valid_scores.median():.3f}")
            print(f"Min: {valid_scores.min():.3f}")
            print(f"Max: {valid_scores.max():.3f}")
            print(f"Std Dev: {valid_scores.std():.3f}")
    
    # Source accuracy
    if "source_match" in df.columns:
        source_matches = df["source_match"].sum()
        expected_count = df["expected_source"].notna().sum()
        if expected_count > 0:
            print("\n--- SOURCE ACCURACY ---")
            print(f"Source matches: {source_matches} / {expected_count} ({source_matches/expected_count*100:.1f}%)")
    
    # Answer grounding
    if "is_grounded" in df.columns:
        grounded_count = df["is_grounded"].sum()
        print("\n--- ANSWER GROUNDING ---")
        print(f"Grounded answers: {grounded_count} ({grounded_count/total_questions*100:.1f}%)")
        print(f"Fallback answers: {total_questions - grounded_count} ({(total_questions - grounded_count)/total_questions*100:.1f}%)")
    
    # Latency statistics
    if "latency_ms" in df.columns:
        valid_latencies = df[df["latency_ms"].notna()]["latency_ms"]
        if len(valid_latencies) > 0:
            print("\n--- LATENCY STATISTICS (ms) ---")
            print(f"Mean: {valid_latencies.mean():.1f}")
            print(f"Median: {valid_latencies.median():.1f}")
            print(f"Min: {valid_latencies.min():.1f}")
            print(f"Max: {valid_latencies.max():.1f}")
            print(f"Std Dev: {valid_latencies.std():.1f}")
    
    # Latency by source type
    if "latency_ms" in df.columns and "source_used" in df.columns:
        print("\n--- LATENCY BY SOURCE TYPE (ms) ---")
        for source in df["source_used"].unique():
            if source != "error":
                source_latencies = df[df["source_used"] == source]["latency_ms"]
                valid_source_latencies = source_latencies[source_latencies.notna()]
                if len(valid_source_latencies) > 0:
                    print(f"{source}: Mean={valid_source_latencies.mean():.1f}, "
                          f"Median={valid_source_latencies.median():.1f}")
    
    # Topic breakdown
    if "topic" in df.columns:
        print("\n--- RESULTS BY TOPIC ---")
        for topic in df["topic"].unique():
            topic_df = df[df["topic"] == topic]
            topic_count = len(topic_df)
            if "source_used" in topic_df.columns:
                topic_sources = topic_df["source_used"].value_counts()
                print(f"\n{topic} ({topic_count} questions):")
                for source, count in topic_sources.items():
                    print(f"  {source}: {count}")
    
    print("\n" + "="*80)
    print("Detailed results table:")
    print("="*80)
    
    # Display detailed table
    display_columns = [
        "question_id", "question", "source_used", "similarity_score",
        "context_retrieved", "is_grounded", "latency_ms"
    ]
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].to_string(index=False))
    
    return df


def main():
    """Main evaluation function."""
    print("Loading evaluation set...")
    evaluation_questions = load_evaluation_set()
    print(f"Loaded {len(evaluation_questions)} evaluation questions.")
    
    print("\nStarting evaluation...")
    print("Note: Ensure the Flask server is running on http://127.0.0.1:5001")
    
    results = []
    for i, question_data in enumerate(evaluation_questions, 1):
        print(f"\nEvaluating question {i}/{len(evaluation_questions)}: {question_data['question'][:50]}...")
        result = evaluate_question(question_data)
        results.append(result)
        time.sleep(0.5)  # Rate limiting
    
    # Generate report
    df = generate_evaluation_report(results)
    
    # Save results to CSV
    output_path = Path(__file__).parent / "evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Save summary JSON
    summary = {
        "total_questions": len(df),
        "successful_responses": len(df[df["source_used"] != "error"]) if "source_used" in df.columns else 0,
        "context_retrieval_rate": df["context_retrieved"].mean() if "context_retrieved" in df.columns else 0,
        "mean_similarity": df["similarity_score"].mean() if "similarity_score" in df.columns else 0,
        "mean_latency_ms": df["latency_ms"].mean() if "latency_ms" in df.columns else 0,
        "grounding_rate": df["is_grounded"].mean() if "is_grounded" in df.columns else 0
    }
    
    summary_path = Path(__file__).parent / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()


