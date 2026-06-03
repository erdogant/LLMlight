"""Unit tests for Step 2 — parameter consolidation & validation.

Tests cover:
  1.  _resolve_embedding()    — all input forms, bad values, 'memvid' context guard
  2.  _resolve_chunks()       — defaults, legacy key aliases, validation errors
  3.  _validate_params()      — valid combinations, each bad-value path
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
    _resolve_embedding,
    _resolve_chunks,
    _validate_params,
    get_embeddings,
)


# ---------------------------------------------------------------------------
# 1. _resolve_embedding
# ---------------------------------------------------------------------------
class TestResolveEmbedding(unittest.TestCase):

    def test_none_returns_defaults(self):
        r = _resolve_embedding(None)
        self.assertIn('memory', r)
        self.assertIn('context', r)
        self.assertNotEqual(r['context'], 'memvid')

    def test_automatic_returns_defaults(self):
        self.assertEqual(_resolve_embedding('automatic'), _resolve_embedding(None))

    def test_string_tfidf_sets_both(self):
        r = _resolve_embedding('tfidf')
        self.assertEqual(r['memory'], 'tfidf')
        self.assertEqual(r['context'], 'tfidf')

    def test_string_bert_sets_both(self):
        r = _resolve_embedding('bert')
        self.assertEqual(r['memory'], 'bert')
        self.assertEqual(r['context'], 'bert')

    def test_string_memvid_memory_ok_context_fallback(self):
        # 'memvid' as string → memory=memvid, context falls back (not memvid)
        r = _resolve_embedding('memvid')
        self.assertEqual(r['memory'], 'memvid')
        self.assertNotEqual(r['context'], 'memvid')

    def test_dict_explicit_keys(self):
        r = _resolve_embedding({'memory': 'tfidf', 'context': 'bert'})
        self.assertEqual(r['memory'], 'tfidf')
        self.assertEqual(r['context'], 'bert')

    def test_dict_context_memvid_fallback(self):
        # Setting context=memvid in a dict should warn and fall back
        r = _resolve_embedding({'memory': 'memvid', 'context': 'memvid'})
        self.assertNotEqual(r['context'], 'memvid')

    def test_dict_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            _resolve_embedding({'memory': 'tfidf', 'typo_key': 'bert'})

    def test_bad_string_raises(self):
        with self.assertRaises(ValueError):
            _resolve_embedding('nonexistent_embedding')

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            _resolve_embedding(42)

    def test_bad_memory_value_raises(self):
        with self.assertRaises(ValueError):
            _resolve_embedding({'memory': 'bad', 'context': 'tfidf'})

    def test_bad_context_value_raises(self):
        with self.assertRaises(ValueError):
            _resolve_embedding({'memory': 'tfidf', 'context': 'bad'})

    def test_get_embeddings_returns_list(self):
        embs = get_embeddings()
        self.assertIsInstance(embs, list)
        self.assertIn('tfidf', embs)
        self.assertIn('memvid', embs)


# ---------------------------------------------------------------------------
# 2. _resolve_chunks
# ---------------------------------------------------------------------------
class TestResolveChunks(unittest.TestCase):

    def test_none_returns_defaults(self):
        r = _resolve_chunks(None)
        self.assertIn('method', r)
        self.assertIn('size', r)
        self.assertIn('overlap', r)

    def test_partial_dict_merges_with_defaults(self):
        r = _resolve_chunks({'size': 500})
        self.assertEqual(r['size'], 500)
        self.assertIn('method', r)   # filled from defaults
        self.assertIn('overlap', r)

    def test_legacy_type_alias(self):
        r = _resolve_chunks({'type': 'words', 'size': 200, 'overlap': 10})
        self.assertEqual(r['method'], 'words')
        self.assertNotIn('type', r)

    def test_legacy_chunk_size_alias(self):
        r = _resolve_chunks({'chunk_size': 300, 'overlap': 0})
        self.assertEqual(r['size'], 300)
        self.assertNotIn('chunk_size', r)

    def test_bad_method_raises(self):
        with self.assertRaises(ValueError):
            _resolve_chunks({'method': 'sentences', 'size': 100, 'overlap': 0})

    def test_bad_size_raises(self):
        with self.assertRaises(ValueError):
            _resolve_chunks({'method': 'chars', 'size': 0, 'overlap': 0})

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError):
            _resolve_chunks({'method': 'chars', 'size': 100, 'overlap': -1})

    def test_overlap_ge_size_raises(self):
        with self.assertRaises(ValueError):
            _resolve_chunks({'method': 'chars', 'size': 100, 'overlap': 100})

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            _resolve_chunks("chars:100")


# ---------------------------------------------------------------------------
# 3. _validate_params
# ---------------------------------------------------------------------------
def _base_params(**overrides):
    base = dict(
        model=None, retrieval_method='naive_rag', embedding=None,
        context_strategy=None, alpha=None, top_chunks=5,
        temperature=0.7, top_p=1.0, chunks=None, n_ctx=4096,
    )
    base.update(overrides)
    return base


class TestValidateParams(unittest.TestCase):

    def test_valid_defaults_ok(self):
        r = _validate_params(**_base_params())
        self.assertIn('embedding', r)
        self.assertIn('chunks', r)

    def test_bad_retrieval_method(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(retrieval_method='magic_rag'))

    def test_none_retrieval_method_ok(self):
        r = _validate_params(**_base_params(retrieval_method=None))
        self.assertIsNone(r['retrieval_method'])

    def test_bad_context_strategy(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(context_strategy='turbo-reasoning'))

    def test_alpha_zero_raises(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(alpha=0.0))

    def test_alpha_one_raises(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(alpha=1.0))

    def test_alpha_none_ok(self):
        r = _validate_params(**_base_params(alpha=None))
        self.assertIsNone(r['alpha'])

    def test_alpha_valid_ok(self):
        r = _validate_params(**_base_params(alpha=0.05))
        self.assertAlmostEqual(r['alpha'], 0.05)

    def test_bad_top_chunks(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(top_chunks=0))

    def test_bad_temperature(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(temperature=3.0))

    def test_bad_top_p(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(top_p=0.0))

    def test_bad_n_ctx(self):
        with self.assertRaises(ValueError):
            _validate_params(**_base_params(n_ctx=64))

    def test_temperature_stored_as_float(self):
        r = _validate_params(**_base_params(temperature=1))
        self.assertIsInstance(r['temperature'], float)


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