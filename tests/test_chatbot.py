import statistics
import time

import requests

API_URL = "http://127.0.0.1:5001/ask"

TEST_SET = [
    {
        "question": "What is a Python list?",
        "expect_source": "dataset",
    },
    {
        "question": "Explain linear regression in data science.",
        "expect_source": "dataset",
    },
    {
        "question": "Give me a high-level summary of convolutional neural networks.",
        "expect_source": "llm",
    },
    {
        "question": "How do I read a CSV file using pandas?",
        "expect_source": "dataset",
    },
    {
        "question": "Share tips for preparing for a data science interview.",
        "expect_source": None,
    },
]


def evaluate_chatbot():
    latencies = []
    dataset_expectations = hits = 0

    for case in TEST_SET:
        start = time.perf_counter()
        response = requests.post(API_URL, data={"message": case["question"]}).json()
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

        source = response.get("source")
        preview = response.get("response", "")[:120]
        print(f"Q: {case['question']}\n  Source: {source} | Snippet: {preview}\n")

        if case["expect_source"] == "dataset":
            dataset_expectations += 1
            if source == "dataset":
                hits += 1

    dataset_accuracy = (hits / dataset_expectations * 100) if dataset_expectations else 0
    latency_stats = {
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "avg_ms": round(statistics.mean(latencies), 1),
    }

    print("Dataset accuracy: {:.1f}% ({} of {} questions)".format(dataset_accuracy, hits, dataset_expectations))
    print(f"Latency (ms): min {latency_stats['min_ms']}, avg {latency_stats['avg_ms']}, max {latency_stats['max_ms']}")


if __name__ == "__main__":
    evaluate_chatbot()
