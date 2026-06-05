"""Memory backends for LLMlight.

Public API (backend-agnostic)
------------------------------
create_memory_backend(store_path, config, backend)
    Factory that returns a backend instance.  Callers never need to import a
    concrete backend class.

Every backend exposes the same interface:
    .store_path  (str)  – resolved absolute path to the persisted store
    .add(text, input_files, dirpath, ...)
    .load()
    .save(...)
    .search(query, top_k)  -> list[str]
    .remove(ids, query)    -> list[int]
    .get_all_chunks()      -> list[str]
    .get_random_chunks(n)  -> list[str]
    .show_stats()
"""

import logging
from typing import List, Union
import os
from pathlib import Path
import json
import time
import LLMlight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_store_path(store_path: str, backend: str) -> str:
    """Return an absolute path appropriate for the requested backend.

    Rules
    -----
    - None / empty  → use cwd with a sensible default filename.
    - Relative path → kept as-is (caller already resolved to tempdir if needed).
    - Extension is normalised so the backend always gets what it expects:
        memvid  : .mp4
        sqlite  : .db
    """
    if not store_path:
        ext = '.mp4' if backend == 'memvid' else '.db'
        store_path = os.path.join(os.getcwd(), f'llmlight_store{ext}')

    store_path = os.path.abspath(store_path)
    base, ext = os.path.splitext(store_path)

    if backend == 'memvid':
        # Ensure a video-compatible extension
        if ext.lower() not in ('.mp4', '.avi', '.mkv'):
            store_path = base + '.mp4'
    else:
        # sqlite backend always uses .db
        store_path = base + '.db'

    return store_path


# ---------------------------------------------------------------------------
# Memvid backend
# ---------------------------------------------------------------------------

class MemvidBackend:
    """Memory backend that encodes chunks into a QR-code video file.

    This wraps the *memvid* library.  It is selected when
    ``backend='memvid'`` is passed to :func:`create_memory_backend`.
    """

    # The public attribute every backend must expose.
    store_path: str = None

    def __init__(self, store_path: str, config: dict = None):
        self.store_path = None
        self.index_path = None

        if store_path is None:
            return

        self._set_store_path(store_path)

        if os.path.isfile(self.store_path):
            logger.info(f'Initializing existing memory store: {self.store_path}')
        else:
            logger.info(f'Initializing new memory store: {self.store_path}')

        try:
            from memvid import MemvidEncoder
        except Exception as e:
            raise ImportError(
                "The 'memvid' package is required for the memvid backend. "
                "Install it with: pip install memvid"
            ) from e

        self.encoder = MemvidEncoder(config=config)
        self.config = self.encoder.config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_store_path(self, store_path: str):
        """Derive and store the video path and its companion index path."""
        store_path = os.path.abspath(store_path)
        directory = os.path.dirname(store_path)
        name, extension = os.path.splitext(os.path.basename(store_path))

        if extension.lower() not in ('.mp4', '.avi', '.mkv'):
            raise ValueError(
                f"Memvid backend expects a video file (.mp4 / .avi / .mkv), "
                f"got: '{extension}'"
            )

        self.store_path = store_path
        # Keep index alongside the video with the same stem
        self.index_path = os.path.join(directory, name) + '.json'

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------

    def load(self):
        """Load the video and its index so the retriever is ready."""
        if not self.store_path or not os.path.isfile(self.store_path):
            logger.warning(f'Store file not found, skipping load: {self.store_path}')
            return

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f'Index file missing: {self.index_path}')

        # Basic integrity check
        try:
            with open(self.index_path, 'r') as fh:
                index_data = json.load(fh)
            chunk_count = len(index_data.get('metadata', []))
        except Exception as exc:
            raise ValueError(f'Index file corrupted: {exc}') from exc

        size_mb = Path(self.store_path).stat().st_size / (1024 * 1024)
        logger.info(f'Loading memory store ({size_mb:.1f} MB, {chunk_count} chunks):')
        logger.info(f'  store : {self.store_path}')
        logger.info(f'  index : {self.index_path}')

        try:
            from memvid import MemvidRetriever
        except Exception as exc:
            raise ImportError(
                "The 'memvid' package is required to load the memory store. "
                "Install it with: pip install memvid"
            ) from exc

        self.retriever = MemvidRetriever(
            video_file=self.store_path,
            index_file=self.index_path,
            config=self.config,
        )

    def add(self,
            text: Union[str, List[str]] = None,
            input_files: Union[str, List[str]] = None,
            dirpath: str = None,
            filetypes: List[str] = None,
            chunk_size: int = 512,
            chunk_overlap: int = 100,
            overwrite: bool = True,
            tempdir: str = None):
        """Add text chunks or files to the pending encoder buffer."""
        if filetypes is None:
            filetypes = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx',
                         '.rtf', '.html', '.htm']

        if not hasattr(self, 'encoder'):
            raise RuntimeError(
                'Memory store is not initialised. Call memory_init() first.'
            )
        if self.store_path and os.path.isfile(self.store_path) and not overwrite:
            logger.warning(f'Store already exists and overwrite=False: {self.store_path}')
            return

        if isinstance(text, str):
            text = [text]
        if isinstance(input_files, str):
            input_files = [input_files]

        # Collect files from a directory
        if dirpath and os.path.isdir(dirpath):
            if input_files is None:
                input_files = []
            for root, _, files in os.walk(dirpath):
                for fname in files:
                    if any(fname.lower().endswith(ext) for ext in filetypes):
                        input_files.append(os.path.join(root, fname))

        # Add raw text chunks
        if text:
            logger.info(f'Adding {len(text)} text chunks to memory buffer.')
            self.encoder.add_chunks(text)

        # Download URLs, then process files
        if input_files:
            resolved = []
            for path in input_files:
                if path.startswith('http'):
                    try:
                        logger.info(f'Downloading: {path}')
                        fname = LLMlight.wget.filename_from_url(path)
                        dest = os.path.join(tempdir or os.getcwd(), fname)
                        LLMlight.wget.download(path, dest)
                        resolved.append(dest)
                    except Exception as exc:
                        logger.warning(f'Download failed for {path}: {exc}')
                elif os.path.isfile(path):
                    resolved.append(path)
                else:
                    logger.warning(f'File not found, skipping: {path}')

            logger.info(f'Adding {len(resolved)} file(s) to memory buffer.')
            for fpath in resolved:
                self._ingest_file(fpath, filetypes, chunk_size, chunk_overlap)

    def _ingest_file(self, file_path, filetypes, chunk_size, chunk_overlap):
        """Add a single file to the encoder buffer."""
        filename = os.path.basename(file_path)
        _, ext = os.path.splitext(filename.lower())

        if ext == '.pdf' and ext in filetypes:
            self.encoder.add_pdf(file_path, chunk_size=chunk_size, overlap=chunk_overlap)
        elif ext == '.epub' and ext in filetypes:
            self.encoder.add_epub(file_path, chunk_size=chunk_size, overlap=chunk_overlap)
        elif ext in ('.html', '.htm') and ext in filetypes:
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                logger.warning(f'BeautifulSoup not available, skipping HTML: {file_path}')
                return
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                soup = BeautifulSoup(fh.read(), 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                raw = soup.get_text()
                lines = (ln.strip() for ln in raw.splitlines())
                phrases = (ph.strip() for ln in lines for ph in ln.split('  '))
                clean = ' '.join(ph for ph in phrases if ph)
                if clean.strip():
                    self.encoder.add_text(clean, chunk_size=chunk_size, overlap=chunk_overlap)
        elif ext in filetypes:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                self.encoder.add_text(fh.read(), chunk_size=chunk_size, overlap=chunk_overlap)
        else:
            logger.debug(f'Skipping unsupported file type: {filename}')
            return

        logger.info(f'Added to memory buffer: {filename}')

    def save(self,
             store_path: str = None,
             codec: str = 'mp4v',
             auto_build_docker: bool = False,
             allow_fallback: bool = True,
             overwrite: bool = True,
             show_progress: bool = True):
        """Encode buffered chunks and write the store to disk."""
        if not hasattr(self, 'store_path') or not self.store_path:
            raise RuntimeError('Memory store path is not set. Call memory_init() first.')

        if not hasattr(self, 'encoder') or len(self.encoder.chunks) == 0:
            logger.warning('No chunks in buffer — nothing to save. Use memory_add() first.')
            return

        if store_path:
            self._set_store_path(store_path)

        if os.path.isfile(self.store_path) and not overwrite:
            logger.warning(f'Store exists and overwrite=False: {self.store_path}')
            return

        # Remove old files when overwriting
        if overwrite:
            for fpath in (self.store_path, self.index_path):
                if fpath and os.path.isfile(fpath):
                    os.remove(fpath)

        # Merge previously-saved chunks with new buffer (for incremental saves)
        if hasattr(self, 'retriever'):
            existing = [m.get('text') for m in self.retriever.index_manager.metadata]
            merged = list(set(existing + self.encoder.chunks))
            self.encoder.clear()
            self.encoder.chunks = merged

        logger.info(f'Saving memory store: {self.store_path}')
        logger.info(f'Total chunks to encode: {len(self.encoder.chunks)}')

        t0 = time.time()
        build_stats = self._build_with_fallback(
            codec=codec,
            show_progress=show_progress,
            auto_build_docker=auto_build_docker,
            allow_fallback=allow_fallback,
        )
        build_stats['encoding_time'] = time.time() - t0

        self.encoder.chunks = []
        self.encoder.clear()
        self.build_stats = build_stats
        logger.info(f'Memory store saved: {self.store_path}')

    def _build_with_fallback(self, codec, show_progress, auto_build_docker, allow_fallback):
        """Attempt build_video, falling back to Flat index on FAISS IVF errors."""
        kwargs = dict(
            output_file=self.store_path,
            index_file=self.index_path,
            codec=codec,
            show_progress=show_progress,
            auto_build_docker=auto_build_docker,
            allow_fallback=allow_fallback,
        )
        try:
            return self.encoder.build_video(**kwargs)
        except Exception as exc:
            err = str(exc)
            if any(tok in err for tok in ('is_trained', 'IndexIVFFlat', 'training')):
                logger.warning(f'FAISS IVF training failed, retrying with Flat index: {exc}')
                self.encoder.config['index']['type'] = 'Flat'
                self.encoder._setup_index()
                return self.encoder.build_video(**kwargs)
            raise

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Return the top-k most relevant chunks for *query*."""
        if not hasattr(self, 'retriever'):
            logger.warning('No retriever loaded. Call load() or save() first.')
            return []
        results = self.retriever.index_manager.search(query, top_k=top_k)
        return [r[2]['text'] for r in results]

    def search_with_scores(self, query: str, top_k: int = 5):
        """Return (score, text) tuples for the top-k results."""
        if not hasattr(self, 'retriever'):
            logger.warning('No retriever loaded. Call load() or save() first.')
            return []
        results = self.retriever.index_manager.search(query, top_k=top_k)
        return [(r[1], r[2]['text']) for r in results]

    def get_all_chunks(self) -> List[str]:
        """Return all stored chunks from disk."""
        if not hasattr(self, 'retriever'):
            return []
        return [m.get('text') for m in self.retriever.index_manager.metadata]

    def get_random_chunks(self, n: int = 1000) -> List[str]:
        """Return *n* chunks with shuffled words (used to build a null distribution)."""
        import random
        chunks = self.get_all_chunks()[:n]
        if not chunks:
            return []

        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.replace('\n', ' ').replace('\t', ' ').split())

        buckets: List[List[str]] = [[] for _ in chunks]
        for word in all_words:
            buckets[random.randint(0, len(buckets) - 1)].append(word)

        return [' '.join(b) for b in buckets]

    def remove(self,
               ids: List[int] = None,
               query: str = None,
               top_k: int = 1) -> List[int]:
        """Remove chunks from the store by id(s) or by search query.

        For the memvid backend, removal is applied to the loaded index
        metadata in-memory.  Call :meth:`save` afterwards to persist the
        change (a new video will be built without the removed chunks).

        Parameters
        ----------
        ids : list of int, optional
            Chunk ids to remove (the first element of each search result).
        query : str, optional
            Search query — the top-*top_k* matching chunks are removed.
            Ignored when *ids* is provided.
        top_k : int
            Number of top search results to remove when using *query*.

        Returns
        -------
        list of int
            The ids that were marked for removal.
        """
        if not hasattr(self, 'retriever'):
            logger.warning("remove(): no retriever loaded — call load() first.")
            return []

        to_delete_ids: List[int] = []

        if ids is not None:
            if isinstance(ids, int):
                ids = [ids]
            to_delete_ids = list(ids)
        elif query is not None:
            results = self.retriever.index_manager.search(query, top_k=top_k)
            # results are (frame_index, score, metadata_dict)
            to_delete_ids = [r[0] for r in results]
        else:
            raise ValueError("Provide either ids= or query= to remove().")

        if not to_delete_ids:
            logger.info("remove(): nothing to delete.")
            return []

        # Filter metadata in-memory
        before = len(self.retriever.index_manager.metadata)
        self.retriever.index_manager.metadata = [
            m for m in self.retriever.index_manager.metadata
            if m.get('frame_index', m.get('id')) not in to_delete_ids
        ]
        after = len(self.retriever.index_manager.metadata)
        logger.info(
            "remove(): marked %d chunk(s) for removal (%d -> %d). "
            "Call save() to persist.", before - after, before, after,
        )

        # Stage removed texts so save() will rebuild without them
        if not hasattr(self, '_pending_remove_ids'):
            self._pending_remove_ids = set()
        self._pending_remove_ids.update(to_delete_ids)

        return to_delete_ids

    def show_stats(self):
        """Log a summary of the last save operation."""
        if not hasattr(self, 'build_stats'):
            logger.warning('No build statistics available — call save() first.')
            return

        stats = self.build_stats
        size_mb = Path(self.store_path).stat().st_size / (1024 * 1024) if (
            self.store_path and os.path.isfile(self.store_path)
        ) else 0.0
        enc_time = stats.get('encoding_time')

        logger.info('Memory store statistics:')
        logger.info(f'  store   : {self.store_path}')
        logger.info(f'  index   : {self.index_path}')
        logger.info(f'  chunks  : {stats.get("total_chunks", "unknown")}')
        logger.info(f'  frames  : {stats.get("total_frames", "unknown")}')
        logger.info(f'  size    : {size_mb:.1f} MB')
        if enc_time is not None:
            logger.info(f'  encoded : {enc_time:.2f}s')


# ---------------------------------------------------------------------------
# SQLite+HNSW backend  (thin wrapper — the real class lives in db_backends/)
# ---------------------------------------------------------------------------

def _import_sqlite_impl():
    """Import the SqliteHnswLLM class, trying every plausible import path.

    The module may be installed as a top-level package, as part of the
    LLMlight package, or inside a db_backends sub-package depending on the
    project layout.  We try them all and raise a clear error only when every
    path fails.
    """
    candidates = [
        # Installed as sibling module inside the LLMlight package
        ("LLMlight.sqlite_hnsw",          "SqliteHnswLLM"),
        # Top-level module (editable install, tests, standalone script)
        ("sqlite_hnsw",                   "SqliteHnswLLM"),
        # Sub-package layout used in some project structures
        ("LLMlight.db_backends.sqlite_hnsw", "SqliteHnswLLM"),
        ("db_backends.sqlite_hnsw",          "SqliteHnswLLM"),
    ]
    last_exc = None
    for module_path, class_name in candidates:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            last_exc = exc
            continue

    raise ImportError(
        "Could not import SqliteHnswLLM from any known location. "
        "Make sure sqlite_hnsw.py is inside the LLMlight package directory "
        "or is importable from the Python path."
    ) from last_exc


class SqliteBackend:
    """Memory backend that stores chunks in a local SQLite database with an
    optional HNSW index for fast approximate-nearest-neighbour search.

    This is the default backend.  It is selected when ``backend='sqlite'``
    (or any string starting with ``'sqlite'``) is passed to
    :func:`create_memory_backend`.
    """

    def __init__(self, store_path: str, config: dict = None):
        _Impl = _import_sqlite_impl()
        self._impl = _Impl(store_path, config=config)
        self.store_path = store_path

    # Delegate everything to the wrapped implementation
    def close(self):
        if hasattr(self._impl, "close"):
            self._impl.close()
    def __getattr__(self, name):
        # Only called when the attribute is NOT found on SqliteBackend itself,
        # so self._impl is always available via __dict__ lookup.
        return getattr(self._impl, name)

    def load(self):
        if hasattr(self._impl, 'load'):
            self._impl.load()

    def search(self, query: str, top_k: int = 5) -> List[str]:
        return self._impl.search(query, top_k=top_k)

    def search_with_scores(self, query: str, top_k: int = 5):
        if hasattr(self._impl, 'search_with_scores'):
            return self._impl.search_with_scores(query, top_k=top_k)
        # Fallback: call plain search and return (1.0, text) pairs
        return [(1.0, t) for t in self.search(query, top_k=top_k)]

    def get_all_chunks(self) -> List[str]:
        if hasattr(self._impl, 'get_all_chunks'):
            return self._impl.get_all_chunks()
        # Fallback via metadata if the impl exposes a retriever
        if hasattr(self._impl, 'retriever') and hasattr(self._impl.retriever, 'index_manager'):
            return [m.get('text') for m in self._impl.retriever.index_manager.metadata]
        return []

    def get_random_chunks(self, n: int = 1000) -> List[str]:
        if hasattr(self._impl, 'get_random_chunks'):
            return self._impl.get_random_chunks(n)
        import random
        chunks = self.get_all_chunks()[:n]
        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.replace('\n', ' ').replace('\t', ' ').split())
        buckets: List[List[str]] = [[] for _ in chunks]
        for word in all_words:
            buckets[random.randint(0, len(buckets) - 1)].append(word)
        return [' '.join(b) for b in buckets]

    def remove(self,
               ids: List[int] = None,
               query: str = None,
               top_k: int = 1) -> List[int]:
        """Remove chunks by id(s) or search query. See backend docs for details."""
        if hasattr(self._impl, 'remove'):
            return self._impl.remove(ids=ids, query=query, top_k=top_k)
        raise NotImplementedError(
            "The current backend implementation does not support remove()."
        )

    def show_stats(self):
        if hasattr(self._impl, 'show_stats'):
            self._impl.show_stats()


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_memory_backend(store_path: str = None,
                          config: dict = None,
                          backend: str = 'sqlite') -> Union[MemvidBackend, SqliteBackend]:
    """Create and return a memory backend instance.

    Parameters
    ----------
    store_path : str, optional
        Path for the persistent store.  Relative paths are kept as-is
        (the caller is responsible for resolving them to an absolute path
        before calling here).  If *None*, a default filename is created in
        the current working directory.
    config : dict, optional
        Backend-specific configuration dict passed straight through.
    backend : str
        ``'sqlite'`` (default) — SQLite + HNSW index, no extra dependencies.
        ``'memvid'``           — QR-code video store, requires *memvid*.
        Any string starting with ``'sqlite'`` selects the sqlite backend.

    Returns
    -------
    MemvidBackend or SqliteBackend
        A backend instance with a uniform interface.
    """
    if backend is None:
        backend = 'sqlite'
    backend = backend.lower().strip()

    resolved = _resolve_store_path(store_path, backend)

    if backend == 'memvid':
        logger.info(f'Creating memvid backend: {resolved}')
        return MemvidBackend(resolved, config=config)

    if backend.startswith('sqlite'):
        logger.info(f'Creating sqlite backend: {resolved}')
        return SqliteBackend(resolved, config=config)

    raise ValueError(
        f"Unknown memory backend: '{backend}'. "
        f"Valid choices: 'sqlite' (default), 'memvid'."
    )