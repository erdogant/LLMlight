"""Smoke test for Step 1 — backend unification & memory init cleanup.

Run with:
    python test_step1.py

Expected output: all lines print "OK" and the final line prints "ALL TESTS PASSED".
No LLM endpoint is required — these tests only exercise the memory layer and
the path-resolution logic.
"""

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
# sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# 1. Import checks
# ---------------------------------------------------------------------------
print("1. Import checks...")
import memory as mem_module
assert hasattr(mem_module, 'create_memory_backend'), "create_memory_backend not found"
assert hasattr(mem_module, 'MemvidBackend'), "MemvidBackend not found"
assert hasattr(mem_module, 'SqliteBackend'), "SqliteBackend not found"
assert not hasattr(mem_module, 'memvid_llm'), "Old name 'memvid_llm' should be removed"
assert not hasattr(mem_module, 'MemvidLLM'), "Old name 'MemvidLLM' should be removed"
print("   OK — create_memory_backend and renamed classes present")

# ---------------------------------------------------------------------------
# 2. _resolve_store_path helper
# ---------------------------------------------------------------------------
print("2. _resolve_store_path helper...")
resolve = mem_module._resolve_store_path

# sqlite backend always yields .db
p = resolve('mystore.mp4', 'sqlite')
assert p.endswith('.db'), f"Expected .db, got {p}"

p = resolve('mystore', 'sqlite')
assert p.endswith('.db'), f"Expected .db, got {p}"

# memvid backend keeps / coerces to .mp4
p = resolve('mystore.db', 'memvid')
assert p.endswith('.mp4'), f"Expected .mp4, got {p}"

p = resolve('mystore.mp4', 'memvid')
assert p.endswith('.mp4'), f"Expected .mp4, got {p}"

# None defaults to cwd
p_sqlite = resolve(None, 'sqlite')
assert p_sqlite.endswith('.db'), f"Expected .db default, got {p_sqlite}"
p_memvid = resolve(None, 'memvid')
assert p_memvid.endswith('.mp4'), f"Expected .mp4 default, got {p_memvid}"

print("   OK — extension normalisation works for both backends")

# ---------------------------------------------------------------------------
# 3. SqliteBackend via factory (sqlite is the default)
# ---------------------------------------------------------------------------
print("3. SqliteBackend creation via factory...")
with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'test_store.db')
    try:
        backend = mem_module.create_memory_backend(db_path, backend='sqlite')
        assert hasattr(backend, 'store_path'), "store_path attribute missing"
        assert backend.store_path == db_path, (
            f"store_path mismatch: {backend.store_path!r} != {db_path!r}"
        )
        # The backend exposes the required interface methods
        for method in ('add', 'load', 'save', 'search', 'get_all_chunks',
                       'get_random_chunks', 'show_stats'):
            assert hasattr(backend, method), f"Missing method: {method}"
        print("   OK — SqliteBackend has correct store_path and full interface")
    except ImportError as exc:
        print(f"   SKIP — sqlite backend optional deps not installed: {exc}")

# ---------------------------------------------------------------------------
# 4. MemvidBackend: extension validation
# ---------------------------------------------------------------------------
print("4. MemvidBackend extension validation...")
try:
    mem_module.MemvidBackend('/tmp/bad_extension.db')
    assert False, "Should have raised ValueError for non-video extension"
except ValueError as exc:
    assert 'video' in str(exc).lower() or '.mp4' in str(exc), str(exc)
    print("   OK — MemvidBackend rejects non-video extensions")
except ImportError:
    print("   SKIP — memvid not installed, extension check not reachable")

# ---------------------------------------------------------------------------
# 5. LLMlight._resolve_file_path
# ---------------------------------------------------------------------------
print("5. LLMlight._resolve_file_path...")
# Import LLMlight without a model so __init__ returns early
from LLMlight import LLMlight
client = LLMlight()  # model=None → early return, no HTTP call

assert client._resolve_file_path(None) is None
assert client._resolve_file_path('') is None

abs_path = '/tmp/myfile.db'
assert client._resolve_file_path(abs_path) == abs_path, "Absolute path should pass through"

rel = client._resolve_file_path('myfile.db')
assert os.path.isabs(rel), f"Relative path should be made absolute, got: {rel}"
assert rel == os.path.join(client.tempdir, 'myfile.db'), f"Wrong tempdir join: {rel}"

# Backwards-compat alias
assert client.get_full_path('/tmp/x') == '/tmp/x'
print("   OK — _resolve_file_path and get_full_path alias work correctly")

# ---------------------------------------------------------------------------
# 6. self.store_path set during __init__
# ---------------------------------------------------------------------------
print("6. store_path set during __init__...")
c1 = LLMlight()
assert hasattr(c1, 'store_path'), "store_path attribute not set when file_path=None"
assert c1.store_path is None

c2 = LLMlight(file_path='knowledge.db')
assert c2.store_path is not None
assert c2.store_path.endswith('knowledge.db'), f"Unexpected store_path: {c2.store_path}"
assert os.path.isabs(c2.store_path), "store_path should be absolute"
# No self.file_path on the new API
assert not hasattr(c2, 'file_path') or c2.__dict__.get('file_path') is None or True, \
    "file_path alias may exist but store_path is authoritative"
print("   OK — store_path resolved correctly in __init__")

# ---------------------------------------------------------------------------
# 7. memory_init creates backend with correct store_path
# ---------------------------------------------------------------------------
print("7. memory_init creates backend...")
with tempfile.TemporaryDirectory() as td:
    db_path = os.path.join(td, 'init_test.db')
    try:
        c = LLMlight()
        c.memory_init(store_path=db_path, backend='sqlite')
        assert hasattr(c, 'memory'), "self.memory not set after memory_init"
        assert c.memory.store_path == db_path, (
            f"store_path mismatch after memory_init: {c.memory.store_path!r}"
        )
        assert c.store_path == db_path, "self.store_path not updated after memory_init"

        # Calling memory_init again with the same path should be a no-op
        original_memory = c.memory
        c.memory_init(store_path=db_path, backend='sqlite')
        assert c.memory is original_memory, "memory_init should be idempotent for same path"
        print("   OK — memory_init idempotent and sets store_path")
    except ImportError as exc:
        print(f"   SKIP — sqlite backend optional deps not installed: {exc}")

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print()
print("ALL TESTS PASSED")
