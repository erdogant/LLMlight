"""Tests for Step 1 — backend unification & memory init cleanup.

Covers:
  1. Package-level imports (create_memory_backend, renamed classes)
  2. _resolve_store_path extension normalisation
  3. SqliteBackend creation via factory (skipped when deps missing)
  4. MemvidBackend extension validation
  5. LLMlight._resolve_file_path and get_full_path alias
  6. self.store_path set correctly during __init__
  7. memory_init idempotency and store_path propagation
"""

import os
import tempfile
import unittest

# All imports go through the installed package — no sys.path hacks.
import LLMlight as ll_pkg
from LLMlight import LLMlight
from LLMlight import memory as mem_module


# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------
class TestImports(unittest.TestCase):

    def test_create_memory_backend_present(self):
        self.assertTrue(hasattr(mem_module, 'create_memory_backend'))

    def test_memvid_backend_class_present(self):
        self.assertTrue(hasattr(mem_module, 'MemvidBackend'))

    def test_sqlite_backend_class_present(self):
        self.assertTrue(hasattr(mem_module, 'SqliteBackend'))

    def test_old_name_memvid_llm_removed(self):
        self.assertFalse(hasattr(mem_module, 'memvid_llm'))

    def test_old_class_memvid_llm_removed(self):
        self.assertFalse(hasattr(mem_module, 'MemvidLLM'))


# ---------------------------------------------------------------------------
# 2. _resolve_store_path
# ---------------------------------------------------------------------------
class TestResolveStorePath(unittest.TestCase):

    def setUp(self):
        self.resolve = mem_module._resolve_store_path

    def test_sqlite_coerces_mp4_to_db(self):
        p = self.resolve('mystore.mp4', 'sqlite')
        self.assertTrue(p.endswith('.db'), p)

    def test_sqlite_bare_name_gets_db(self):
        p = self.resolve('mystore', 'sqlite')
        self.assertTrue(p.endswith('.db'), p)

    def test_memvid_coerces_db_to_mp4(self):
        p = self.resolve('mystore.db', 'memvid')
        self.assertTrue(p.endswith('.mp4'), p)

    def test_memvid_keeps_mp4(self):
        p = self.resolve('mystore.mp4', 'memvid')
        self.assertTrue(p.endswith('.mp4'), p)

    def test_none_sqlite_defaults_to_db(self):
        p = self.resolve(None, 'sqlite')
        self.assertTrue(p.endswith('.db'), p)

    def test_none_memvid_defaults_to_mp4(self):
        p = self.resolve(None, 'memvid')
        self.assertTrue(p.endswith('.mp4'), p)

    def test_result_is_absolute(self):
        p = self.resolve('relative/path.db', 'sqlite')
        self.assertTrue(os.path.isabs(p), p)


# ---------------------------------------------------------------------------
# 3. SqliteBackend via factory
# ---------------------------------------------------------------------------
class TestSqliteBackendFactory(unittest.TestCase):

    def test_store_path_set_correctly(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, 'test_store.db')
            try:
                backend = mem_module.create_memory_backend(db_path, backend='sqlite')
            except ImportError as exc:
                self.skipTest(f"sqlite deps not installed: {exc}")
            self.assertEqual(backend.store_path, db_path)

    def test_interface_methods_present(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, 'iface_test.db')
            try:
                backend = mem_module.create_memory_backend(db_path, backend='sqlite')
            except ImportError as exc:
                self.skipTest(f"sqlite deps not installed: {exc}")
            for method in ('add', 'load', 'save', 'search',
                           'get_all_chunks', 'get_random_chunks', 'show_stats'):
                self.assertTrue(hasattr(backend, method), f"Missing: {method}")

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            mem_module.create_memory_backend('x.db', backend='nonexistent')


# ---------------------------------------------------------------------------
# 4. MemvidBackend extension validation
# ---------------------------------------------------------------------------
class TestMemvidBackendValidation(unittest.TestCase):

    def test_rejects_non_video_extension(self):
        try:
            mem_module.MemvidBackend(os.path.join(tempfile.gettempdir(), 'bad_extension.db'))
            self.fail("Should have raised ValueError")
        except ValueError as exc:
            self.assertTrue('video' in str(exc).lower() or '.mp4' in str(exc), str(exc))
        except ImportError:
            self.skipTest("memvid not installed")


# ---------------------------------------------------------------------------
# 5. LLMlight._resolve_file_path
# ---------------------------------------------------------------------------
class TestResolveFilePath(unittest.TestCase):

    def setUp(self):
        self.client = LLMlight()  # model=None → early return, no HTTP call

    def test_none_returns_none(self):
        self.assertIsNone(self.client._resolve_file_path(None))

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.client._resolve_file_path(''))

    def test_absolute_path_passes_through(self):
        # Use a platform-appropriate absolute path
        p = os.path.join(tempfile.gettempdir(), 'myfile.db')
        self.assertEqual(self.client._resolve_file_path(p), p)

    def test_relative_becomes_absolute(self):
        rel = self.client._resolve_file_path('myfile.db')
        self.assertTrue(os.path.isabs(rel), rel)

    def test_relative_resolves_under_tempdir(self):
        rel = self.client._resolve_file_path('myfile.db')
        expected = os.path.join(self.client.tempdir, 'myfile.db')
        self.assertEqual(os.path.normcase(rel), os.path.normcase(expected))

    def test_get_full_path_alias(self):
        p = os.path.join(tempfile.gettempdir(), 'x')
        self.assertEqual(
            self.client.get_full_path(p),
            self.client._resolve_file_path(p),
        )


# ---------------------------------------------------------------------------
# 6. store_path during __init__
# ---------------------------------------------------------------------------
class TestStorePath(unittest.TestCase):

    def test_store_path_none_when_no_file_path(self):
        c = LLMlight()
        self.assertTrue(hasattr(c, 'store_path'))
        self.assertIsNone(c.store_path)

    def test_store_path_set_from_file_path(self):
        c = LLMlight(file_path='knowledge.db')
        self.assertIsNotNone(c.store_path)
        self.assertTrue(c.store_path.endswith('knowledge.db'), c.store_path)
        self.assertTrue(os.path.isabs(c.store_path), c.store_path)


# ---------------------------------------------------------------------------
# 7. memory_init idempotency
# ---------------------------------------------------------------------------
class TestMemoryInit(unittest.TestCase):

    def test_memory_init_sets_store_path(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, 'init_test.db')
            c = LLMlight()
            try:
                c.memory_init(store_path=db_path, backend='sqlite')
            except ImportError as exc:
                self.skipTest(f"sqlite deps not installed: {exc}")
            self.assertTrue(hasattr(c, 'memory'))
            self.assertEqual(c.memory.store_path, db_path)
            self.assertEqual(c.store_path, db_path)

    def test_memory_init_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = os.path.join(td, 'idem_test.db')
            c = LLMlight()
            try:
                c.memory_init(store_path=db_path, backend='sqlite')
            except ImportError as exc:
                self.skipTest(f"sqlite deps not installed: {exc}")
            original_memory = c.memory
            c.memory_init(store_path=db_path, backend='sqlite')
            self.assertIs(c.memory, original_memory)


if __name__ == '__main__':
    unittest.main(verbosity=2)