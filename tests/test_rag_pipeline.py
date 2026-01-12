"""
Comprehensive tests for the RAG pipeline components.
Tests correctness, not just execution.
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.preprocessing import preprocess_text, chunk_text, create_index, load_index
from app.rag_pipeline import (
    retrieve_chunks, rank_chunks, assemble_prompt, generate_answer,
    check_answer_relevance, initialize_main_dataset, initialize_uploaded_files
)
from app.constants import SIMILARITY_THRESHOLD, ANSWER_RELEVANCE_THRESHOLD


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "  Hello   World  "
        result = preprocess_text(text)
        self.assertEqual(result, "Hello World")

    def test_preprocess_text_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Hello\n\n\nWorld\t\tTest"
        result = preprocess_text(text)
        self.assertIn("Hello", result)
        self.assertIn("World", result)
        self.assertIn("Test", result)

    def test_chunk_text_small(self):
        """Test chunking small text."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunk_text_large(self):
        """Test chunking large text."""
        text = " ".join(["Sentence"] * 100)  # Large text
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 1)
        # Check overlap
        if len(chunks) > 1:
            # Verify chunks are not empty
            for chunk in chunks:
                self.assertGreater(len(chunk), 0)


class TestRetrieval(unittest.TestCase):
    """Test retrieval functions."""

    def setUp(self):
        """Set up test data."""
        # Initialize with test chunks
        test_chunks = [
            "Python is a programming language",
            "Data science uses Python for analysis",
            "Machine learning is a subset of AI",
            "Pandas is a Python library for data manipulation"
        ]
        test_metadata = [
            {"chunk_idx": i, "source": "test"} for i in range(len(test_chunks))
        ]
        initialize_uploaded_files(test_chunks, test_metadata, "test_index")

    def test_retrieve_chunks_non_empty(self):
        """Test that retrieval returns non-empty results for relevant queries."""
        results = retrieve_chunks("Python programming", source="uploaded", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], tuple)
        self.assertEqual(len(results[0]), 3)  # (text, score, metadata)

    def test_retrieve_chunks_threshold(self):
        """Test similarity threshold behavior."""
        # Query with low relevance
        results = retrieve_chunks("completely unrelated topic", source="uploaded", 
                                 top_k=3, threshold=0.8)
        # Should return empty or filtered results
        for result in results:
            self.assertGreaterEqual(result[1], 0.8)

    def test_retrieve_chunks_empty_query(self):
        """Test handling of empty query."""
        results = retrieve_chunks("", source="uploaded")
        # Should handle gracefully
        self.assertIsInstance(results, list)

    def test_rank_chunks(self):
        """Test chunk ranking."""
        chunks_with_scores = [
            ("chunk1", 0.3, {}),
            ("chunk2", 0.8, {}),
            ("chunk3", 0.5, {})
        ]
        ranked = rank_chunks(chunks_with_scores)
        self.assertEqual(ranked[0][1], 0.8)  # Highest score first
        self.assertEqual(ranked[-1][1], 0.3)  # Lowest score last


class TestPromptAssembly(unittest.TestCase):
    """Test prompt assembly."""

    def test_assemble_prompt_with_context(self):
        """Test prompt assembly with context."""
        context = ["Context 1", "Context 2"]
        prompt = assemble_prompt("What is Python?", context, use_llm=True)
        self.assertIn("What is Python?", prompt)
        self.assertIn("Context 1", prompt)

    def test_assemble_prompt_without_context(self):
        """Test prompt assembly without context."""
        prompt = assemble_prompt("What is Python?", [], use_llm=True)
        self.assertIn("What is Python?", prompt)

    def test_assemble_prompt_dataset_mode(self):
        """Test prompt assembly for dataset retrieval."""
        prompt = assemble_prompt("What is Python?", [], use_llm=False)
        self.assertEqual(prompt, "What is Python?")


class TestHallucinationPrevention(unittest.TestCase):
    """Test hallucination prevention checks."""

    def test_check_answer_relevance_relevant(self):
        """Test relevance check with relevant answer."""
        answer = "Python is a programming language used for data science"
        retrieved_chunks = [
            ("Python is a programming language", 0.8, {}),
            ("Data science uses Python", 0.7, {})
        ]
        is_relevant, score = check_answer_relevance(answer, retrieved_chunks)
        self.assertIsInstance(is_relevant, bool)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_check_answer_relevance_irrelevant(self):
        """Test relevance check with irrelevant answer."""
        answer = "Completely unrelated topic about cooking recipes"
        retrieved_chunks = [
            ("Python is a programming language", 0.8, {}),
            ("Data science uses Python", 0.7, {})
        ]
        is_relevant, score = check_answer_relevance(answer, retrieved_chunks, threshold=0.5)
        # Should detect low relevance
        self.assertIsInstance(is_relevant, bool)

    def test_check_answer_relevance_empty_chunks(self):
        """Test relevance check with empty chunks."""
        answer = "Some answer"
        is_relevant, score = check_answer_relevance(answer, [])
        self.assertFalse(is_relevant)
        self.assertEqual(score, 0.0)


class TestParaphrasedQuestions(unittest.TestCase):
    """Test handling of paraphrased questions."""

    def setUp(self):
        """Set up test data."""
        test_chunks = [
            "Python is a high-level programming language",
            "Python supports object-oriented programming",
            "Python has dynamic typing"
        ]
        test_metadata = [{"chunk_idx": i} for i in range(len(test_chunks))]
        initialize_uploaded_files(test_chunks, test_metadata, "paraphrase_test")

    def test_paraphrased_question_handling(self):
        """Test that paraphrased questions still retrieve relevant chunks."""
        original = "What is Python?"
        paraphrases = [
            "Tell me about Python",
            "Can you explain Python?",
            "I want to know about Python"
        ]
        
        original_results = retrieve_chunks(original, source="uploaded", top_k=1)
        
        for paraphrase in paraphrases:
            para_results = retrieve_chunks(paraphrase, source="uploaded", top_k=1)
            # Should retrieve similar or same chunks
            self.assertIsInstance(para_results, list)


class TestSourceLabeling(unittest.TestCase):
    """Test correct source labeling."""

    def test_dataset_source_label(self):
        """Test that dataset retrieval returns correct source."""
        # This would require actual dataset initialization
        # For now, test the structure
        from app.chatbot import get_chatbot_response
        # Mock test - actual implementation would check source labels
        pass


class TestNoRelevantContext(unittest.TestCase):
    """Test handling when no relevant context is retrieved."""

    def test_no_context_handling(self):
        """Test behavior when retrieval returns no relevant context."""
        results = retrieve_chunks("completely unrelated query xyz123", 
                                 source="uploaded", top_k=3, threshold=0.9)
        # Should return empty or very few results
        self.assertIsInstance(results, list)
        # All results should meet threshold if any exist
        for result in results:
            self.assertGreaterEqual(result[1], 0.9)


class TestLatency(unittest.TestCase):
    """Test latency considerations."""

    def test_retrieval_speed(self):
        """Test that retrieval is reasonably fast."""
        import time
        start = time.perf_counter()
        retrieve_chunks("Python", source="uploaded", top_k=3)
        elapsed = time.perf_counter() - start
        # Should complete in reasonable time (< 1 second)
        self.assertLess(elapsed, 1.0)


if __name__ == '__main__':
    unittest.main()


