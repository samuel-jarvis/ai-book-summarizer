import tempfile
import unittest
from pathlib import Path

from app.services.indexing.loaders import load_document


class LoadDocumentTests(unittest.TestCase):
    def test_strips_nul_bytes_postgres_cannot_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "book.txt"
            path.write_bytes(b"\x00Chapter one\x00\nText\x00")

            documents = load_document(path, document_id="d", project_id="p")

        self.assertEqual(documents[0].page_content, "Chapter one\nText")
        self.assertEqual(documents[0].metadata["document_id"], "d")


if __name__ == "__main__":
    unittest.main()
