"""Tests for Step 3 — unified retrieval pipeline.

Covers:
  1.  _embed()                     — tfidf, bow shapes; unsupported raises
  2.  _retrieve()                  — ranking, top_k cap, empty input, scores descending
  3.  _filter_by_significance()    — alpha=None passthrough, short list, mask applied
  4.  search()                     — backwards-compat return types
  5.  fit_transform()              — backwards-compat wrapper
  6.  relevant_context_retrieval() — naive_rag list/string, no retrieval, empty context
  7.  relevant_memory_retrieval()  — no store → None, missing file → None
  8.  compute_context_strategy()   — None passthrough, global-reasoning, chunk-wise
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from LLMlight import LLMlight


def _client(**kwargs):
    return LLMlight(**kwargs)


# ---------------------------------------------------------------------------
# 1. _embed
# ---------------------------------------------------------------------------
class TestEmbed(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.chunks = [
            "The cat sat on the mat.",
            "Dogs are loyal companions.",
            "Python is a programming language.",
        ]

    def test_tfidf_shapes(self):
        qv, cv = self.c._embed("cat mat", self.chunks, "tfidf")
        self.assertEqual(qv.shape[0], 1)
        self.assertEqual(cv.shape[0], len(self.chunks))

    def test_bow_shapes(self):
        qv, cv = self.c._embed("cat mat", self.chunks, "bow")
        self.assertEqual(qv.shape[0], 1)
        self.assertEqual(cv.shape[0], len(self.chunks))

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            self.c._embed("query", self.chunks, "no_such_method")

    def test_memvid_raises_as_embed_method(self):
        with self.assertRaises(ValueError):
            self.c._embed("query", self.chunks, "memvid")


# ---------------------------------------------------------------------------
# 2. _retrieve
# ---------------------------------------------------------------------------
class TestRetrieve(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.chunks = [
            "The Eiffel Tower is in Paris.",
            "Mount Everest is the tallest mountain.",
            "The Amazon river is in South America.",
            "Paris is the capital of France.",
            "Python snakes live in tropical regions.",
        ]

    def test_returns_list_of_tuples(self):
        result = self.c._retrieve("Paris France", self.chunks, "tfidf", top_k=3)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in result))

    def test_top_k_respected(self):
        self.assertEqual(len(self.c._retrieve("Paris", self.chunks, "tfidf", top_k=2)), 2)

    def test_top_k_capped(self):
        self.assertEqual(len(self.c._retrieve("Paris", self.chunks, "tfidf", top_k=999)), len(self.chunks))

    def test_scores_descending(self):
        scores = [s for s, _ in self.c._retrieve("Paris France", self.chunks, "tfidf", top_k=5)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_relevant_chunk_in_top(self):
        result = self.c._retrieve("Paris France capital", self.chunks, "tfidf", top_k=3)
        top_texts = [t for _, t in result]
        paris = [c for c in self.chunks if "Paris" in c]
        self.assertTrue(any(p in top_texts for p in paris))

    def test_empty_input(self):
        self.assertEqual(self.c._retrieve("query", [], "tfidf", top_k=5), [])


# ---------------------------------------------------------------------------
# 3. _filter_by_significance
# ---------------------------------------------------------------------------
class TestFilterBySignificance(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.scored = [(0.9, "very relevant"), (0.5, "somewhat"), (0.1, "irrelevant")]

    def test_alpha_none_passthrough(self):
        self.c.alpha = None
        self.assertEqual(self.c._filter_by_significance("q", self.scored), self.scored)

    def test_single_item_passthrough(self):
        self.c.alpha = 0.05
        single = [(0.9, "only")]
        self.assertEqual(self.c._filter_by_significance("q", single), single)

    def test_compute_probability_none_passthrough(self):
        self.c.alpha = 0.05
        with patch.object(self.c, 'compute_probability', return_value=None):
            self.assertEqual(self.c._filter_by_significance("q", self.scored), self.scored)

    def test_mask_filters_correctly(self):
        self.c.alpha = 0.05
        fake_out = {'y_bool': np.array([True, False, True])}
        with patch.object(self.c, 'compute_probability', return_value=fake_out):
            result = self.c._filter_by_significance("q", self.scored)
        self.assertEqual(len(result), 2)
        texts = [t for _, t in result]
        self.assertIn("very relevant", texts)
        self.assertIn("irrelevant", texts)
        self.assertNotIn("somewhat", texts)


# ---------------------------------------------------------------------------
# 4. search() backwards-compat wrapper
# ---------------------------------------------------------------------------
class TestSearchWrapper(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.chunks = [
            "The quick brown fox jumps.",
            "A lazy dog sleeps all day.",
            "Foxes are cunning animals.",
        ]

    def test_return_score(self):
        result = self.c.search("fox", self.chunks, return_type='score', top_chunks=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(r, tuple) for r in result))

    def test_return_list(self):
        result = self.c.search("fox", self.chunks, return_type='list', top_chunks=2)
        self.assertTrue(all(isinstance(r, str) for r in result))

    def test_return_string_flat(self):
        result = self.c.search("fox", self.chunks, return_type='string_flat', top_chunks=2)
        self.assertIsInstance(result, str)
        self.assertNotIn('\n', result)

    def test_return_other_is_string(self):
        result = self.c.search("fox", self.chunks, return_type='joined', top_chunks=2)
        self.assertIsInstance(result, str)

    def test_explicit_embedding(self):
        result = self.c.search("fox", self.chunks, return_type='list',
                               top_chunks=2, embedding='tfidf')
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# 5. fit_transform() backwards-compat wrapper
# ---------------------------------------------------------------------------
class TestFitTransformWrapper(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.chunks = ["apple pie", "banana smoothie", "cherry tart"]

    def test_string_embedding(self):
        qv, cv = self.c.fit_transform("apple", self.chunks, embedding='tfidf')
        self.assertEqual(cv.shape[0], 3)

    def test_dict_embedding(self):
        qv, cv = self.c.fit_transform(
            "apple", self.chunks,
            embedding={'context': 'tfidf', 'memory': 'memvid'},
        )
        self.assertEqual(cv.shape[0], 3)

    def test_none_embedding_uses_self(self):
        self.c.embedding = {'context': 'tfidf', 'memory': 'memvid'}
        qv, cv = self.c.fit_transform("apple", self.chunks, embedding=None)
        self.assertEqual(cv.shape[0], 3)


# ---------------------------------------------------------------------------
# 6. relevant_context_retrieval
# ---------------------------------------------------------------------------
class TestRelevantContextRetrieval(unittest.TestCase):

    def setUp(self):
        self.c = _client()
        self.c.chunks = {'method': 'chars', 'size': 50, 'overlap': 0}

    def test_none_context(self):
        self.assertIsNone(self.c.relevant_context_retrieval("q", None))

    def test_empty_context(self):
        self.assertFalse(self.c.relevant_context_retrieval("q", ""))

    def test_no_retrieval_method_returns_full_context(self):
        self.c.retrieval_method = None
        ctx = "Full context text unchanged."
        self.assertEqual(self.c.relevant_context_retrieval("q", ctx), ctx)

    def test_naive_rag_returns_list(self):
        self.c.retrieval_method = 'naive_rag'
        self.c.embedding = {'context': 'tfidf', 'memory': 'memvid'}
        self.c.top_chunks = 2
        ctx = " ".join(["word"] * 300)
        result = self.c.relevant_context_retrieval("word", ctx)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2)

    def test_naive_rag_return_string(self):
        self.c.retrieval_method = 'naive_rag'
        self.c.embedding = {'context': 'tfidf', 'memory': 'memvid'}
        self.c.top_chunks = 2
        ctx = " ".join(["word"] * 300)
        result = self.c.relevant_context_retrieval("word", ctx, return_type='string')
        self.assertIsInstance(result, str)
        self.assertIn("### Chunk", result)


# ---------------------------------------------------------------------------
# 7. relevant_memory_retrieval — no store
# ---------------------------------------------------------------------------
class TestRelevantMemoryRetrievalNoStore(unittest.TestCase):

    def test_none_store_path(self):
        c = _client()
        self.assertIsNone(c.relevant_memory_retrieval("anything"))

    def test_missing_file(self):
        c = _client()
        c.store_path = '/nonexistent/path/store.db'
        self.assertIsNone(c.relevant_memory_retrieval("anything"))


# ---------------------------------------------------------------------------
# 8. compute_context_strategy
# ---------------------------------------------------------------------------
class TestComputeContextStrategy(unittest.TestCase):

    def test_none_strategy_passthrough(self):
        c = _client()
        c.context_strategy = None
        self.assertEqual(c.compute_context_strategy("q", "ctx", "i", "s"), "ctx")

    def test_none_context_returns_none(self):
        c = _client()
        c.context_strategy = None
        self.assertIsNone(c.compute_context_strategy("q", None, "i", "s"))

    def test_global_reasoning_delegated(self):
        c = _client()
        c.context_strategy = 'global-reasoning'
        with patch.object(c, 'global_reasoning', return_value="summary") as mock:
            result = c.compute_context_strategy("q", "ctx", "i", "s")
        mock.assert_called_once()
        self.assertEqual(result, "summary")

    def test_chunk_wise_delegated(self):
        c = _client()
        c.context_strategy = 'chunk-wise'
        with patch.object(c, 'chunk_wise', return_value="chunked") as mock:
            result = c.compute_context_strategy("q", "ctx", "i", "s")
        mock.assert_called_once()
        self.assertEqual(result, "chunked")


if __name__ == '__main__':
    unittest.main(verbosity=2)