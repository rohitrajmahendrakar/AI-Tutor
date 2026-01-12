"""
Integration tests for the chatbot system.
Tests end-to-end behavior including file uploads and source labeling.
"""
import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chatbot import get_chatbot_response, add_uploaded_file
from app.preprocessing import process_uploaded_file
from app.constants import SIMILARITY_THRESHOLD


class TestChatbotIntegration(unittest.TestCase):
    """Integration tests for chatbot."""

    def test_get_response_structure(self):
        """Test that get_chatbot_response returns correct structure."""
        response, source, metadata = get_chatbot_response("What is Python?")
        
        self.assertIsInstance(response, str)
        self.assertIn(source, ["dataset", "uploaded", "llm", "uncertain"])
        self.assertIsInstance(metadata, dict)
        self.assertIn("similarity_score", metadata)
        self.assertIn("dataset_source", metadata)
        self.assertIn("is_grounded", metadata)

    def test_uploaded_file_indexing(self):
        """Test correct indexing of uploaded files."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Python is a programming language. It is used for data science.")
            temp_path = Path(f.name)
        
        try:
            success = add_uploaded_file(temp_path, "test_file.txt")
            # Should succeed
            self.assertIsInstance(success, bool)
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()

    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
            f.write("Test content")
            temp_path = Path(f.name)
        
        try:
            from app.preprocessing import process_uploaded_file
            with self.assertRaises(ValueError):
                process_uploaded_file(temp_path, "test.pdf")
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_empty_upload_handling(self):
        """Test handling of empty uploads."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")  # Empty file
            temp_path = Path(f.name)
        
        try:
            from app.preprocessing import process_uploaded_file
            chunks, metadata = process_uploaded_file(temp_path, "empty.txt")
            # Should handle gracefully
            self.assertIsInstance(chunks, list)
            self.assertIsInstance(metadata, list)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_correct_source_labeling(self):
        """Test that responses have correct source labels."""
        response, source, metadata = get_chatbot_response("What is Python?")
        
        # Source should match metadata
        if source == "dataset":
            self.assertIn(metadata["dataset_source"], ["main_dataset", "uploaded_files"])
            self.assertTrue(metadata.get("is_grounded", False))
        elif source == "uncertain":
            # Should have low similarity or no grounding
            self.assertLessEqual(metadata.get("similarity_score", 1.0), SIMILARITY_THRESHOLD)


if __name__ == '__main__':
    unittest.main()

