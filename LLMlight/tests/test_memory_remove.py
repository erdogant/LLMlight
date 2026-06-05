"""Tests for memory remove functionality.

Covers:
  1. SqliteHNSWBackend.remove(ids=...)    — by id
  2. SqliteHNSWBackend.remove(query=...)  — by query
  3. SqliteHNSWBackend.remove()           — bad call raises
  4. SqliteBackend.remove()              — delegates to impl
  5. LLMlight.memory_remove()            — public API
  6. remove() returns correct ids
  7. removed chunk no longer in get_all_chunks()
  8. remove all docs leaves empty store
"""

import os
import tempfile
import unittest

from LLMlight import memory as mem_module
from LLMlight import LLMlight

SQLITE_AVAILABLE = False

try:
    from sqlite_hnsw import SqliteHnswLLM
    SQLITE_AVAILABLE = True
except ImportError:
    try:
        from LLMlight.sqlite_hnsw import SqliteHnswLLM
        SQLITE_AVAILABLE = True
    except ImportError:
        pass


@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestSqliteRemoveById(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = os.path.join(self.td, 'test.db')
        self.backend = SqliteHnswLLM(self.db)
        self.backend.add(text=['Apple pie recipe', 'Banana smoothie', 'Cherry tart'])

    def test_remove_by_id_returns_id(self):
        all_meta = self.backend._fetch_all_metadata()
        target_id = all_meta[0]['id']
        removed = self.backend.remove(ids=target_id)
        self.assertEqual(removed, [target_id])

    def test_remove_by_id_reduces_count(self):
        all_meta = self.backend._fetch_all_metadata()
        target_id = all_meta[0]['id']
        before = len(self.backend._fetch_all_metadata())
        self.backend.remove(ids=target_id)
        after = len(self.backend._fetch_all_metadata())
        self.assertEqual(after, before - 1)

    def test_removed_text_not_in_db(self):
        all_meta = self.backend._fetch_all_metadata()
        target = all_meta[0]
        self.backend.remove(ids=target['id'])
        remaining = [m['text'] for m in self.backend._fetch_all_metadata()]
        self.assertNotIn(target['text'], remaining)

    def test_remove_list_of_ids(self):
        all_meta = self.backend._fetch_all_metadata()
        ids = [all_meta[0]['id'], all_meta[1]['id']]
        removed = self.backend.remove(ids=ids)
        self.assertEqual(sorted(removed), sorted(ids))
        self.assertEqual(len(self.backend._fetch_all_metadata()), 1)

    def test_remove_all_docs(self):
        all_meta = self.backend._fetch_all_metadata()
        ids = [m['id'] for m in all_meta]
        self.backend.remove(ids=ids)
        self.assertEqual(len(self.backend._fetch_all_metadata()), 0)


@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestSqliteRemoveByQuery(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = os.path.join(self.td, 'test.db')
        self.backend = SqliteHnswLLM(self.db)
        self.backend.add(text=['BMC test data', 'Banana smoothie', 'Cherry tart'])

    def test_remove_by_query_returns_id(self):
        removed = self.backend.remove(query='BMC', top_k=1)
        self.assertEqual(len(removed), 1)

    def test_remove_by_query_text_gone(self):
        self.backend.remove(query='BMC', top_k=1)
        remaining = [m['text'] for m in self.backend._fetch_all_metadata()]
        self.assertFalse(any('BMC' in t for t in remaining))

    def test_remove_top_k_removes_multiple(self):
        before = len(self.backend._fetch_all_metadata())
        self.backend.remove(query='a', top_k=2)
        after = len(self.backend._fetch_all_metadata())
        self.assertLessEqual(after, before - 1)


@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestSqliteRemoveBadCall(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db = os.path.join(self.td, 'test.db')
        self.backend = SqliteHnswLLM(self.db)
        self.backend.add(text=['Some text'])

    def test_no_args_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.backend.remove()


@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestSqliteBackendWrapperRemove(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        db_path = os.path.join(self.td, 'wrap.db')
        self.backend = mem_module.SqliteBackend(db_path)
        self.backend.add(text=['Alpha chunk', 'Beta chunk', 'Gamma chunk'])

    def test_remove_by_id_via_wrapper(self):
        results = self.backend.search('Alpha', top_k=1)
        target_id = results[0][0]
        removed = self.backend.remove(ids=target_id)
        self.assertIn(target_id, removed)

    def test_remove_by_query_via_wrapper(self):
        removed = self.backend.remove(query='Gamma', top_k=1)
        self.assertEqual(len(removed), 1)


@unittest.skipUnless(SQLITE_AVAILABLE, "sqlite_hnsw not available")
class TestLLMlightMemoryRemove(unittest.TestCase):

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.db_path = os.path.join(self.td, 'llm_test.db')
        self.client = LLMlight()
        self.client.memory_init(store_path=self.db_path, backend='sqlite')
        self.client.memory_add(text=['BMC test', 'Unrelated content', 'More data'])

    def test_memory_remove_no_init_raises(self):
        c = LLMlight()
        with self.assertRaises(RuntimeError):
            c.memory_remove(ids=1)

    def test_memory_remove_by_id(self):
        results = self.client.memory.search('BMC', top_k=1)
        target_id = results[0][0]
        removed = self.client.memory_remove(ids=target_id)
        self.assertIn(target_id, removed)

    def test_memory_remove_by_query(self):
        removed = self.client.memory_remove(query='BMC', top_k=1)
        self.assertEqual(len(removed), 1)

    def test_removed_chunk_not_in_all_chunks(self):
        results = self.client.memory.search('BMC', top_k=1)
        target_id = results[0][0]
        self.client.memory_remove(ids=target_id)
        chunks = self.client.memory_chunks()
        self.assertFalse(any('BMC' in c for c in chunks))

    def test_memory_remove_returns_list(self):
        removed = self.client.memory_remove(query='BMC', top_k=1)
        self.assertIsInstance(removed, list)

    def test_memory_remove_no_args_raises(self):
        with self.assertRaises(ValueError):
            self.client.memory_remove()


if __name__ == '__main__':
    unittest.main(verbosity=2)
