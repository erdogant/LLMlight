"""Unit tests for Step 2 — parameter consolidation & validation.

Tests cover:
  4.  LLMlight.__init__       — normalised values stored, alpha=None default,
                                mutable-default safety
  5.  compute_probability()   — distfit crash guards (empty scores, few uniques)

Run with:
    python test_step2.py
"""

import os, sys, types, unittest
from unittest.mock import MagicMock, patch
import numpy as np


for mod in ('llama_cpp', 'memvid', 'distfit'):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import LLMlight as ll_mod
from LLMlight import (
    LLMlight,
    get_embeddings,
)



# ---------------------------------------------------------------------------
# 4. LLMlight.__init__ stores validated values
# ---------------------------------------------------------------------------
class TestLLMlightInit(unittest.TestCase):

    def test_alpha_default_is_none(self):
        c = LLMlight()
        self.assertIsNone(c.alpha)

    def test_alpha_stored_correctly(self):
        c = LLMlight(alpha=0.05)
        self.assertAlmostEqual(c.alpha, 0.05)

    def test_embedding_always_dict(self):
        c = LLMlight(embedding='tfidf')
        self.assertIsInstance(c.embedding, dict)
        self.assertIn('memory', c.embedding)
        self.assertIn('context', c.embedding)

    def test_embedding_none_gives_defaults(self):
        c = LLMlight(embedding=None)
        self.assertIsInstance(c.embedding, dict)

    def test_chunks_normalised(self):
        c = LLMlight(chunks={'size': 512})
        self.assertEqual(c.chunks['size'], 512)
        self.assertIn('method', c.chunks)
        self.assertIn('overlap', c.chunks)

    def test_chunks_legacy_type_alias(self):
        c = LLMlight(chunks={'type': 'words', 'size': 200, 'overlap': 10})
        self.assertEqual(c.chunks['method'], 'words')
        self.assertNotIn('type', c.chunks)

    def test_mutable_default_isolation(self):
        """Two instances must not share the same chunks / embedding dict."""
        c1 = LLMlight()
        c2 = LLMlight()
        c1.chunks['size'] = 9999
        self.assertNotEqual(c2.chunks.get('size'), 9999)

    def test_bad_retrieval_method_raises(self):
        with self.assertRaises(ValueError):
            LLMlight(retrieval_method='bad_method')

    def test_bad_embedding_raises(self):
        with self.assertRaises(ValueError):
            LLMlight(embedding='no_such_embedding')

    def test_bad_chunks_method_raises(self):
        with self.assertRaises(ValueError):
            LLMlight(chunks={'method': 'sentences', 'size': 100, 'overlap': 0})


# ---------------------------------------------------------------------------
# 5. compute_probability — distfit crash guards
# ---------------------------------------------------------------------------
class TestComputeProbabilityGuards(unittest.TestCase):

    def _make_client(self):
        c = LLMlight(alpha=0.05)
        # Attach a minimal mock memory with get_random_chunks
        c.memory = MagicMock()
        c.store_path = '/fake/store.db'
        c.embedding = {'memory': 'tfidf', 'context': 'tfidf'}
        return c

    def test_no_memory_returns_none(self):
        c = LLMlight(alpha=0.05)
        # no self.memory attribute
        result = c.compute_probability("q", [0.9, 0.8], 'tfidf')
        self.assertIsNone(result)

    def test_too_few_scores_returns_none(self):
        c = self._make_client()
        result = c.compute_probability("q", [0.9], 'tfidf')
        self.assertIsNone(result)

    def test_empty_random_chunks_returns_none(self):
        c = self._make_client()
        c.memory.get_random_chunks.return_value = []
        result = c.compute_probability("q", [0.9, 0.8], 'tfidf')
        self.assertIsNone(result)

    def test_too_few_unique_random_scores_returns_none(self):
        c = self._make_client()
        # All identical scores → < 5 unique values
        c.memory.get_random_chunks.return_value = ["chunk a", "chunk b"] * 10

        # _embed returns identical vectors → cosine sim all ~1.0
        fake_qv = np.array([[1.0, 0.0]])
        fake_cv = np.tile([1.0, 0.0], (20, 1))
        with patch.object(c, '_embed', return_value=(fake_qv, fake_cv)):
            result = c.compute_probability("q", [0.9, 0.8], 'tfidf')
        self.assertIsNone(result)

    def test_histdata_none_guard_returns_none(self):
        """The histdata=None guard prevents the distfit IndexError."""
        c = self._make_client()
        c.memory.get_random_chunks.return_value = ["chunk a", "chunk b"] * 15

        fake_qv = np.array([[1.0, 0.0]])
        # 30 rows with enough unique values to pass earlier guards
        fake_cv = np.random.rand(30, 2)

        with patch.object(c, '_embed', return_value=(fake_qv, fake_cv)):
            # Make distfit instance return histdata=None after fit_transform
            mock_instance = MagicMock()
            mock_instance.histdata = None
            mock_distfit_cls = MagicMock(return_value=mock_instance)

            # Patch the import inside compute_probability
            import sys
            real_distfit = sys.modules.get('distfit')
            sys.modules['distfit'] = MagicMock(distfit=mock_distfit_cls)
            try:
                result = c.compute_probability("q", [0.9, 0.8], 'tfidf')
            finally:
                if real_distfit is not None:
                    sys.modules['distfit'] = real_distfit
                else:
                    del sys.modules['distfit']

        # Guard should have caught histdata=None and returned None
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main(verbosity=2)