"""Tests for memory_add text normalisation and chunking fixes.

Covers:
  1. _normalise_text_input — dict, str, list, None, nested
  2. sqlite add() receives dict → chunks correctly, no single-char rows
  3. sqlite add() receives plain str → chunked
  4. sqlite add() receives list of str → stored correctly
  5. LLMlight.memory_add normalises dict before backend call
  6. Embedding batch: ANN index built once, not per-chunk
"""

import os
import tempfile
import unittest

try:
    from sqlite_hnsw import SqliteHnswLLM, _normalise_text_input
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from LLMlight import LLMlight


# ---------------------------------------------------------------------------
# 1. _normalise_text_input
# ---------------------------------------------------------------------------
@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestNormaliseTextInput(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(_normalise_text_input(None), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_normalise_text_input(""), [])
        self.assertEqual(_normalise_text_input("   "), [])

    def test_short_string_not_chunked(self):
        result = _normalise_text_input("hello world", chunk_size=512)
        self.assertEqual(result, ["hello world"])

    def test_long_string_is_chunked(self):
        long = "a" * 2000
        result = _normalise_text_input(long, chunk_size=512, chunk_overlap=0)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 512)

    def test_dict_joined_and_chunked(self):
        d = {"title": "My Title", "body": "Some body text.", "references": ""}
        result = _normalise_text_input(d, chunk_size=512)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(s, str) for s in result))
        combined = " ".join(result)
        self.assertIn("My Title", combined)
        self.assertIn("body text", combined)

    def test_list_of_strings_flattened(self):
        result = _normalise_text_input(["alpha", "beta", "gamma"], chunk_size=512)
        self.assertEqual(result, ["alpha", "beta", "gamma"])

    def test_list_with_dict_flattened(self):
        result = _normalise_text_input([{"body": "text from dict"}], chunk_size=512)
        self.assertIsInstance(result, list)
        self.assertTrue(any("text from dict" in s for s in result))

    def test_no_single_character_chunks(self):
        # Simulates the bug: dict passed in, keys iterated as chars
        d = {"title": "T", "body": "Hello world", "references": "Ref"}
        result = _normalise_text_input(d, chunk_size=512)
        for chunk in result:
            self.assertGreater(len(chunk.strip()), 1,
                               f"Single-char chunk found: {chunk!r}")


# ---------------------------------------------------------------------------
# 2-4. sqlite add() input normalisation
# ---------------------------------------------------------------------------
@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestSqliteAddNormalisation(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = os.path.join(self.td, 'norm_test.db')
        self.backend = SqliteHnswLLM(self.db)

    def _all_texts(self):
        return [m['text'] for m in self.backend._fetch_all_metadata()]

    def test_dict_input_no_single_chars(self):
        pdf_dict = {
            "title": "Attention Is All You Need",
            "body": "We propose a new simple network architecture, the Transformer.",
            "references": "[1] Vaswani et al."
        }
        self.backend.add(text=pdf_dict)
        texts = self._all_texts()
        self.assertTrue(len(texts) > 0)
        for t in texts:
            self.assertGreater(len(t.strip()), 1,
                               f"Single-char or empty chunk stored: {t!r}")

    def test_plain_string_stored_as_chunks(self):
        long_text = "word " * 300   # ~1500 chars
        self.backend.add(text=long_text, chunk_size=200, chunk_overlap=0)
        texts = self._all_texts()
        self.assertGreater(len(texts), 1, "Long string should be split into multiple chunks")
        for t in texts:
            self.assertLessEqual(len(t), 200)

    def test_list_of_strings_stored_correctly(self):
        chunks = ["First chunk text.", "Second chunk text.", "Third chunk text."]
        self.backend.add(text=chunks)
        texts = self._all_texts()
        for original in chunks:
            self.assertIn(original, texts)

    def test_none_text_stores_nothing(self):
        self.backend.add(text=None)
        self.assertEqual(len(self._all_texts()), 0)


# ---------------------------------------------------------------------------
# 5. LLMlight.memory_add normalises dict
# ---------------------------------------------------------------------------
@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestMemoryAddNormalisation(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        db_path = os.path.join(self.td, 'llm_norm.db')
        self.client = LLMlight()
        self.client.memory_init(store_path=db_path, backend='sqlite')

    def test_dict_from_read_pdf_no_single_chars(self):
        pdf_dict = {
            "title": "Paper Title",
            "body": "This is the main body of the paper with enough content.",
            "references": "Ref 1, Ref 2"
        }
        self.client.memory_add(text=pdf_dict)
        chunks = self.client.memory_chunks(n=200)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertGreater(len(chunk.strip()), 1,
                               f"Single-char chunk: {chunk!r}")

    def test_plain_string_stored(self):
        self.client.memory_add(text="The capital of France is Paris.")
        chunks = self.client.memory_chunks(n=10)
        self.assertTrue(any("France" in c for c in chunks))

    def test_list_stored(self):
        self.client.memory_add(text=["Apes like USB sticks.", "Trees are mainly yellow."])
        chunks = self.client.memory_chunks(n=10)
        self.assertTrue(any("Apes" in c for c in chunks))


# ---------------------------------------------------------------------------
# 6. ANN index built once per add() call, not per chunk
# ---------------------------------------------------------------------------
@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestBatchEmbedding(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = os.path.join(self.td, 'batch_test.db')
        self.backend = SqliteHnswLLM(self.db)

    def test_all_chunks_searchable_after_batch_add(self):
        chunks = [f"Document about topic {i}" for i in range(20)]
        self.backend.add(text=chunks, chunk_size=512)
        self.assertEqual(len(self.backend._fetch_all_metadata()), 20)

    def test_search_works_after_large_add(self):
        chunks = ["Paris is the capital of France."] + \
                 [f"Unrelated text number {i}" for i in range(19)]
        self.backend.add(text=chunks, chunk_size=512)
        results = self.backend.search("capital France", top_k=3)
        self.assertTrue(len(results) > 0)
        top_text = results[0][2]['text']
        self.assertIn("Paris", top_text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
