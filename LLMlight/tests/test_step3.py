"""Unit tests for Step 3 — unified retrieval pipeline.

Tests cover:
  1. _embed()                  — all supported embedding methods
  2. _retrieve()               — ranking correctness, top_k cap
  3. _filter_by_significance() — passthrough when alpha=None, short list
  4. search()                  — backwards-compat wrapper return types
  5. fit_transform()           — backwards-compat wrapper
  6. relevant_context_retrieval() — naive_rag, no-retrieval, empty context
  7. relevant_memory_retrieval()  — no store loaded → None
  8. compute_context_strategy()   — None strategy passthrough

Run with:
    python test_step3.py
"""

import os, sys, types

# ---------------------------------------------------------------------------
# Minimal stubs so we can import LLMlight without a live server
# ---------------------------------------------------------------------------
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Stub heavy optional packages before import
for mod in ('llama_cpp', 'memvid', 'distfit'):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import LLMlight as ll_mod
from LLMlight import LLMlight


def make_client(**kwargs):
    """Return a LLMlight instance with model=None (no HTTP calls)."""
    return LLMlight(**kwargs)


# ---------------------------------------------------------------------------
# 1. _embed
# ---------------------------------------------------------------------------
class TestEmbed(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.chunks = [
            "The cat sat on the mat.",
            "Dogs are loyal companions.",
            "Python is a programming language.",
        ]

    def test_tfidf_shapes(self):
        qv, cv = self.client._embed("cat mat", self.chunks, "tfidf")
        self.assertEqual(qv.shape[0], 1)
        self.assertEqual(cv.shape[0], len(self.chunks))

    def test_bow_shapes(self):
        qv, cv = self.client._embed("cat mat", self.chunks, "bow")
        self.assertEqual(qv.shape[0], 1)
        self.assertEqual(cv.shape[0], len(self.chunks))

    def test_unsupported_raises(self):
        with self.assertRaises(ValueError):
            self.client._embed("query", self.chunks, "nonexistent_method")

    def test_memvid_embedding_raises(self):
        """'memvid' is not a valid _embed method (it's a backend, not an embedder)."""
        with self.assertRaises(ValueError):
            self.client._embed("query", self.chunks, "memvid")


# ---------------------------------------------------------------------------
# 2. _retrieve
# ---------------------------------------------------------------------------
class TestRetrieve(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.chunks = [
            "The Eiffel Tower is in Paris.",
            "Mount Everest is the tallest mountain.",
            "The Amazon river is in South America.",
            "Paris is the capital of France.",
            "Python snakes live in tropical regions.",
        ]

    def test_returns_list_of_tuples(self):
        result = self.client._retrieve("Paris France", self.chunks, "tfidf", top_k=3)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(r, tuple) and len(r) == 2 for r in result))

    def test_top_k_respected(self):
        result = self.client._retrieve("Paris", self.chunks, "tfidf", top_k=2)
        self.assertEqual(len(result), 2)

    def test_top_k_capped_at_n_chunks(self):
        result = self.client._retrieve("Paris", self.chunks, "tfidf", top_k=100)
        self.assertEqual(len(result), len(self.chunks))

    def test_scores_descending(self):
        result = self.client._retrieve("Paris France capital", self.chunks, "tfidf", top_k=5)
        scores = [s for s, _ in result]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_relevant_chunk_ranked_first(self):
        result = self.client._retrieve("Paris France capital", self.chunks, "tfidf", top_k=3)
        top_texts = [t for _, t in result]
        # At least one of the Paris-related chunks should be in the top 3
        paris_chunks = [c for c in self.chunks if "Paris" in c]
        self.assertTrue(any(pc in top_texts for pc in paris_chunks))

    def test_empty_chunks_returns_empty(self):
        result = self.client._retrieve("query", [], "tfidf", top_k=5)
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# 3. _filter_by_significance
# ---------------------------------------------------------------------------
class TestFilterBySignificance(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.scored = [(0.9, "very relevant"), (0.5, "somewhat"), (0.1, "irrelevant")]

    def test_passthrough_when_alpha_none(self):
        self.client.alpha = None
        result = self.client._filter_by_significance("q", self.scored)
        self.assertEqual(result, self.scored)

    def test_passthrough_when_too_few_scores(self):
        self.client.alpha = 0.05
        result = self.client._filter_by_significance("q", [(0.9, "only one")])
        self.assertEqual(result, [(0.9, "only one")])

    def test_passthrough_when_compute_probability_returns_none(self):
        self.client.alpha = 0.05
        with patch.object(self.client, 'compute_probability', return_value=None):
            result = self.client._filter_by_significance("q", self.scored)
        self.assertEqual(result, self.scored)

    def test_filters_using_y_bool_mask(self):
        self.client.alpha = 0.05
        fake_out = {'y_bool': np.array([True, False, True])}
        with patch.object(self.client, 'compute_probability', return_value=fake_out):
            result = self.client._filter_by_significance("q", self.scored)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], "very relevant")
        self.assertEqual(result[1][1], "irrelevant")


# ---------------------------------------------------------------------------
# 4. search() backwards-compat wrapper
# ---------------------------------------------------------------------------
class TestSearchWrapper(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.chunks = [
            "The quick brown fox jumps.",
            "A lazy dog sleeps all day.",
            "Foxes are cunning animals.",
        ]

    def test_return_type_score(self):
        result = self.client.search("fox", self.chunks, return_type='score', top_chunks=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(r, tuple) for r in result))

    def test_return_type_list(self):
        result = self.client.search("fox", self.chunks, return_type='list', top_chunks=2)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(r, str) for r in result))

    def test_return_type_string_flat(self):
        result = self.client.search("fox", self.chunks, return_type='string_flat', top_chunks=2)
        self.assertIsInstance(result, str)
        self.assertNotIn('\n', result)

    def test_return_type_string_default(self):
        result = self.client.search("fox", self.chunks, return_type='other', top_chunks=2)
        self.assertIsInstance(result, str)

    def test_explicit_embedding_kwarg(self):
        # should not raise
        result = self.client.search("fox", self.chunks, return_type='list',
                                    top_chunks=2, embedding='tfidf')
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# 5. fit_transform() backwards-compat wrapper
# ---------------------------------------------------------------------------
class TestFitTransformWrapper(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.chunks = ["apple pie recipe", "banana smoothie", "cherry tart"]

    def test_returns_tuple(self):
        qv, cv = self.client.fit_transform("apple", self.chunks, embedding='tfidf')
        self.assertEqual(qv.shape[0], 1)
        self.assertEqual(cv.shape[0], 3)

    def test_dict_embedding_uses_context_key(self):
        qv, cv = self.client.fit_transform("apple", self.chunks,
                                           embedding={'context': 'tfidf', 'memory': 'memvid'})
        self.assertEqual(cv.shape[0], 3)


# ---------------------------------------------------------------------------
# 6. relevant_context_retrieval
# ---------------------------------------------------------------------------
class TestRelevantContextRetrieval(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.client.chunks = {'method': 'chars', 'size': 50, 'overlap': 0}

    def test_none_context_returns_none(self):
        result = self.client.relevant_context_retrieval("query", None)
        self.assertIsNone(result)

    def test_empty_string_returns_empty(self):
        result = self.client.relevant_context_retrieval("query", "")
        # empty string is falsy → returned as-is
        self.assertFalse(result)

    def test_no_retrieval_method_returns_full_context(self):
        self.client.retrieval_method = None
        ctx = "Full context text that should be returned unchanged."
        result = self.client.relevant_context_retrieval("query", ctx)
        self.assertEqual(result, ctx)

    def test_naive_rag_returns_list_by_default(self):
        self.client.retrieval_method = 'naive_rag'
        self.client.embedding = {'context': 'tfidf', 'memory': 'memvid'}
        self.client.top_chunks = 2
        ctx = " ".join(["word"] * 200)   # long enough to produce multiple chunks
        result = self.client.relevant_context_retrieval("word", ctx)
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2)

    def test_naive_rag_return_type_string(self):
        self.client.retrieval_method = 'naive_rag'
        self.client.embedding = {'context': 'tfidf', 'memory': 'memvid'}
        self.client.top_chunks = 2
        ctx = " ".join(["word"] * 200)
        result = self.client.relevant_context_retrieval("word", ctx, return_type='string')
        self.assertIsInstance(result, str)
        self.assertIn("### Chunk", result)


# ---------------------------------------------------------------------------
# 7. relevant_memory_retrieval — no store
# ---------------------------------------------------------------------------
class TestRelevantMemoryRetrievalNoStore(unittest.TestCase):

    def test_returns_none_when_no_store_path(self):
        client = make_client()
        # store_path is None → should return None immediately
        self.assertIsNone(client.relevant_memory_retrieval("anything"))

    def test_returns_none_when_store_file_missing(self):
        client = make_client()
        client.store_path = '/nonexistent/path/store.db'
        self.assertIsNone(client.relevant_memory_retrieval("anything"))


# ---------------------------------------------------------------------------
# 8. compute_context_strategy passthrough
# ---------------------------------------------------------------------------
class TestComputeContextStrategy(unittest.TestCase):

    def test_none_strategy_returns_context_unchanged(self):
        client = make_client()
        client.context_strategy = None
        ctx = "some context"
        result = client.compute_context_strategy("q", ctx, "instr", "sys")
        self.assertEqual(result, ctx)

    def test_none_context_returns_none(self):
        client = make_client()
        client.context_strategy = None
        result = client.compute_context_strategy("q", None, "instr", "sys")
        self.assertIsNone(result)

    def test_global_reasoning_delegates(self):
        client = make_client()
        client.context_strategy = 'global-reasoning'
        with patch.object(client, 'global_reasoning', return_value="summary") as mock_gr:
            result = client.compute_context_strategy("q", "ctx", "instr", "sys")
        mock_gr.assert_called_once()
        self.assertEqual(result, "summary")

    def test_chunk_wise_delegates(self):
        client = make_client()
        client.context_strategy = 'chunk-wise'
        with patch.object(client, 'chunk_wise', return_value="chunked") as mock_cw:
            result = client.compute_context_strategy("q", "ctx", "instr", "sys")
        mock_cw.assert_called_once()
        self.assertEqual(result, "chunked")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main(verbosity=2)