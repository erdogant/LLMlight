"""
SQLite + HNSW lightweight retrieval backend for LLMlight.

Provides a small, embeddable retrieval backend that stores documents and metadata in
SQLite and an in-memory (optionally persisted) HNSW index using `hnswlib` for fast
approximate nearest neighbor search. If `hnswlib` or `sentence-transformers` are
not available, it falls back to a TF-IDF based search using scikit-learn.

This is intentionally small and has a minimal API surface to match the expectations
from the rest of LLMlight (methods: add, save, load, retriever.index_manager.search,
retriever.index_manager.metadata).
"""

import os
import json
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalise_text_input(text, chunk_size: int = 512, chunk_overlap: int = 100) -> List[str]:
    """Convert any text input into a flat list of non-empty string chunks.

    Handles:
      - None                  → []
      - str                   → chunked into <=chunk_size character pieces
      - dict (from read_pdf)  → values joined, then chunked
      - list of str/dict/mix  → each item processed recursively and flattened

    Chunking prevents single huge strings from being stored as one row
    (which makes similarity search useless and embeddings slow).
    """
    if text is None:
        return []

    # Dict: join all string values (handles read_pdf return value)
    if isinstance(text, dict):
        combined = " ".join(str(v) for v in text.values() if v)
        return _normalise_text_input(combined, chunk_size, chunk_overlap)

    # Plain string: chunk it
    if isinstance(text, str):
        text = text.strip()
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]
        # Simple overlapping character chunking
        step = max(1, chunk_size - chunk_overlap)
        return [text[i:i + chunk_size] for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]

    # List: flatten each item
    if isinstance(text, (list, tuple)):
        result = []
        for item in text:
            result.extend(_normalise_text_input(item, chunk_size, chunk_overlap))
        return result

    # Fallback: coerce to string
    return _normalise_text_input(str(text), chunk_size, chunk_overlap)


class IndexManager:
    """Provides search and metadata access used by LLMlight.retriever calls."""

    def __init__(self, backend: "SqliteHNSWBackend"):
        self.backend = backend

    @property
    def metadata(self) -> List[Dict]:
        return self.backend._fetch_all_metadata()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Search the index for a text query. Returns list of tuples: (id, score, metadata_dict).
        Score is a distance (smaller is better) when using ANN, or cosine similarity when using TF-IDF.
        """
        return self.backend.search(query, top_k=top_k)


class Retriever:
    """Thin wrapper carrying an IndexManager to match memvid interface."""

    def __init__(self, backend: "SqliteHNSWBackend"):
        self.index_manager = IndexManager(backend)


class SqliteHNSWBackend:
    """Main backend class.

    Args:
        db_path: path to sqlite database file. If not exists, it will be created.
        config: optional dict. Supported keys:
            - dim: embedding dimension (default 384)
            - embedding_model: name to pass to sentence-transformers (optional)
            - index_path: path to persist the hnswlib index (optional)
    """

    def __init__(self, db_path: str = "llmlight_store.db", config: Optional[dict] = None):
        self.db_path = os.path.abspath(db_path)
        self.config = config or {}
        self.dim = int(self.config.get("dim", 384))
        self.index_path = self.config.get("index_path", os.path.splitext(self.db_path)[0] + ".hnsw")

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_tables()

        # Expose file_path (to match memvid API) and ensure index file path
        self.file_path = self.db_path
        # index_path is already set above

        # In-memory arrays for embeddings (kept in sync with DB when possible)
        self._ids = []
        self._embeddings = None  # numpy array shape (n, dim)

        # Optional ANN index (hnswlib)
        self._ann = None
        self._use_ann = False

        # Retriever wrapper
        self.retriever = Retriever(self)

        # Embedder is loaded lazily on first use (search / add / reindex).
        # This avoids an HTTP round-trip to HuggingFace when just creating or
        # opening an empty database.
        self._embedder = None
        self._embedder_loaded = False  # tracks whether we already attempted load

        # Try to load ANN index if present
        self._maybe_load_ann()

    # Database helpers
    def _ensure_tables(self):
        c = self._conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self._conn.commit()

    def _insert_document(self, text: str, metadata: Optional[dict] = None) -> int:
        c = self._conn.cursor()
        c.execute("INSERT INTO documents (text, metadata) VALUES (?, ?)", (text, json.dumps(metadata or {})))
        self._conn.commit()
        return c.lastrowid

    def _fetch_all_metadata(self) -> List[Dict]:
        c = self._conn.cursor()
        c.execute("SELECT id, text, metadata FROM documents ORDER BY id")
        rows = c.fetchall()
        result = []
        for id_, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            meta.update({"text": text, "id": id_})
            result.append(meta)
        return result

    def _fetch_by_ids(self, ids: List[int]) -> List[Tuple[int, str, dict]]:
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        c = self._conn.cursor()
        c.execute(f"SELECT id, text, metadata FROM documents WHERE id IN ({placeholders})", ids)
        rows = c.fetchall()
        results = []
        for id_, text, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            meta.update({"text": text, "id": id_})
            results.append((id_, text, meta))
        return results

    # Public API
    def add(self,
            text: List[str] = None,
            input_files: List[str] = None,
            dirpath: str = None,
            filetypes: List[str] = None,
            metadata: List[dict] = None,
            chunk_size: int = 512,
            chunk_overlap: int = 100,
            overwrite: bool = True,
            tempdir: str = None,
            **kwargs):
        """Add textual chunks or files into the sqlite store.

        This method accepts a superset of the memvid API parameters for compatibility.
        - `text`: list of text chunks
        - `input_files`: list of file paths or URLs to ingest
        - `dirpath`: directory to scan for files (filtered by `filetypes`)

        Returns list of inserted row ids.
        """
        # Normalize inputs
        if filetypes is None:
            filetypes = ['.pdf', '.txt', '.epub', '.md', '.doc', '.docx', '.rtf', '.html', '.htm']

        # Normalise text to a flat list of non-empty strings.
        # Handles: None, a bare str, a dict (from read_pdf), or a list thereof.
        text = _normalise_text_input(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # If a directory was provided, scan for files
        if dirpath is not None and os.path.isdir(dirpath):
            files_from_dir = []
            for root, _, files in os.walk(dirpath):
                for fname in files:
                    if any(fname.lower().endswith(ext) for ext in filetypes):
                        files_from_dir.append(os.path.join(root, fname))
            if input_files is None:
                input_files = files_from_dir
            else:
                input_files = list(input_files) + files_from_dir

        # If input_files provided, ingest them
        if input_files is not None:
            if isinstance(input_files, str):
                input_files = [input_files]
            files_clean = []
            for input_file in input_files:
                if isinstance(input_file, str) and input_file.lower().startswith('http'):
                    # try download
                    try:
                        import requests
                        import tempfile as _tempfile
                        logger.info(f'Downloading file from url: {input_file}')
                        r = requests.get(input_file, stream=True, timeout=30)
                        r.raise_for_status()
                        filename = os.path.basename(input_file.split('?')[0]) or 'download.tmp'
                        tmp_path = os.path.join(_tempfile.gettempdir(), filename)
                        with open(tmp_path, 'wb') as fh:
                            for chunk in r.iter_content(8192):
                                fh.write(chunk)
                        files_clean.append(tmp_path)
                    except Exception as e:
                        logger.warning(f'Could not download file from {input_file}: {e}')
                elif os.path.isfile(input_file):
                    files_clean.append(input_file)
            input_files = files_clean

            for fp in input_files:
                _, ext = os.path.splitext(fp.lower())
                if ext not in filetypes:
                    logger.debug(f'Skipping file (unsupported extension): {fp}')
                    continue
                # Handle PDF via pymupdf when available
                if ext == '.pdf':
                    try:
                        import fitz  # pymupdf
                        doc = fitz.open(fp)
                        txt_pages = []
                        for pg in range(doc.page_count):
                            txt_pages.append(doc[pg].get_text('text'))
                        text.append('\n'.join(txt_pages))
                    except Exception:
                        logger.warning(f'Could not extract text from PDF (pymupdf missing or failed): {fp}')
                        continue
                else:
                    try:
                        with open(fp, 'r', encoding='utf-8', errors='ignore') as fh:
                            content = fh.read()
                            text.append(content)
                    except Exception as e:
                        logger.warning(f'Failed to read file {fp}: {e}')

        # Ensure metadata list length matches text length
        if metadata is None:
            metadata = [None] * len(text)
        elif len(metadata) < len(text):
            metadata = list(metadata) + [None] * (len(text) - len(metadata))

        ids = []
        for t, m in zip(text, metadata):
            id_ = self._insert_document(t, m)
            ids.append(id_)

        # Compute embeddings in one batch, then rebuild the ANN index once.
        if self._get_embedder() is not None and ids:
            try:
                vectors = self._get_embedder().encode(text, convert_to_numpy=True, show_progress_bar=len(text) > 50)
            except TypeError:
                vectors = np.asarray(self._get_embedder().encode(text))
            self._add_embeddings(ids, vectors)

        return ids

    def _add_embeddings(self, ids: List[int], vectors: np.ndarray):
        """Accumulate embeddings and rebuild the ANN index once for the whole batch."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = vectors.copy()
            self._ids = list(ids)
        else:
            self._embeddings = np.vstack([self._embeddings, vectors])
            self._ids.extend(ids)

        # Build / rebuild the ANN index once using all accumulated embeddings.
        try:
            import hnswlib
            dim = self._embeddings.shape[1]
            p = hnswlib.Index(space='cosine', dim=dim)
            p.init_index(max_elements=len(self._ids), ef_construction=200, M=16)
            p.add_items(self._embeddings, np.array(self._ids, dtype=np.int32))
            p.set_ef(max(50, len(self._ids) // 10 or 1))
            self._ann = p
            self._use_ann = True
            logger.info(f"Built in-memory HNSW index with {len(self._ids)} elements")
        except Exception:
            self._ann = None
            self._use_ann = False
            logger.info("hnswlib not available; using brute-force numpy search (pip install hnswlib)")

    def reindex(self, batch_size: int = 128, save_index: bool = True):
        """Rebuild the ANN index from all documents in the SQLite DB.

        Parameters
        - batch_size: number of texts to encode per batch when computing embeddings.
        - save_index: whether to persist the rebuilt hnswlib index to disk (self.index_path).

        Requirements: `sentence-transformers` for embeddings. `hnswlib` is required to build the ANN.
        If some packages are missing the method will raise an informative ImportError.
        """
        # Ensure we have documents
        c = self._conn.cursor()
        c.execute("SELECT id, text FROM documents ORDER BY id")
        rows = c.fetchall()
        if not rows:
            logger.info("No documents found in DB; nothing to reindex.")
            return

        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        # Need sentence-transformers to compute embeddings
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise ImportError("sentence-transformers is required to compute embeddings for reindex(). Install via 'pip install sentence-transformers'") from e

        # If backend provided a model earlier, reuse; otherwise instantiate default
        if self._embedder is None:
            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model for reindex: {model_name}")
            self._embedder = SentenceTransformer(model_name)

        # Compute embeddings in batches
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vecs = self._embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(vecs)
        embeddings = np.vstack(embeddings).astype(np.float32)

        # Store in-memory arrays
        self._embeddings = embeddings
        self._ids = ids.copy()

        # Build hnswlib index
        try:
            import hnswlib
        except Exception as e:
            raise ImportError("hnswlib is required to build ANN index. Install via 'pip install hnswlib'") from e

        dim = self._embeddings.shape[1]
        p = hnswlib.Index(space='cosine', dim=dim)
        # init index with estimated size
        p.init_index(max_elements=len(self._ids), ef_construction=200, M=16)
        p.add_items(self._embeddings, np.array(self._ids, dtype=np.int32))
        p.set_ef(max(50, int(len(self._ids) / 10)))

        self._ann = p
        self._use_ann = True
        logger.info(f"Rebuilt in-memory HNSW index with {len(self._ids)} elements")

        if save_index:
            try:
                p.save_index(self.index_path)
                logger.info(f"Saved rebuilt ANN index to {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to save rebuilt ANN index: {e}")

        return True
    def remove(self,
               ids: List[int] = None,
               query: str = None,
               top_k: int = 1) -> List[int]:
        """Remove chunks from the store by id(s) or by search query.

        Parameters
        ----------
        ids : list of int, optional
            Row ids to delete (as returned in the first element of each
            search result tuple).
        query : str, optional
            Search query — the top-*top_k* matching chunks are deleted.
            Ignored when *ids* is provided.
        top_k : int
            Number of top search results to remove when using *query*.

        Returns
        -------
        list of int
            The ids that were actually deleted.
        """
        to_delete: List[int] = []

        if ids is not None:
            if isinstance(ids, int):
                ids = [ids]
            to_delete = list(ids)
        elif query is not None:
            results = self.search(query, top_k=top_k)
            to_delete = [r[0] for r in results]
        else:
            raise ValueError("Provide either ids= or query= to remove().")

        if not to_delete:
            logger.info("remove(): nothing to delete.")
            return []

        # Delete from SQLite
        placeholders = ",".join(["?"] * len(to_delete))
        c = self._conn.cursor()
        c.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", to_delete)
        deleted = c.rowcount
        self._conn.commit()
        logger.info(f"remove(): deleted {deleted} document(s) with id(s) {to_delete}.")

        # Rebuild in-memory ANN arrays to stay consistent.
        # hnswlib does not support deletion, so we reload everything from DB.
        if self._use_ann and self._ann is not None and self._get_embedder() is not None:
            c.execute("SELECT id, text FROM documents ORDER BY id")
            rows = c.fetchall()
            if rows:
                ids_remaining = [r[0] for r in rows]
                texts_remaining = [r[1] for r in rows]
                vecs = self._get_embedder().encode(texts_remaining, convert_to_numpy=True).astype(np.float32)
                self._embeddings = vecs
                self._ids = ids_remaining
                try:
                    import hnswlib
                    dim = vecs.shape[1]
                    p = hnswlib.Index(space='cosine', dim=dim)
                    p.init_index(max_elements=len(ids_remaining), ef_construction=200, M=16)
                    p.add_items(vecs, np.array(ids_remaining, dtype=np.int32))
                    p.set_ef(max(50, len(ids_remaining) // 10 or 1))
                    self._ann = p
                    logger.info(f"Rebuilt ANN index with {len(ids_remaining)} remaining document(s).")
                except Exception as exc:
                    logger.warning(f"Could not rebuild ANN index after removal: {exc}")
            else:
                # All docs deleted
                self._embeddings = None
                self._ids = []
                self._ann = None
                self._use_ann = False

        return to_delete

    def save(self, file_path: Optional[str] = None, codec: str = None, auto_build_docker: bool = False, allow_fallback: bool = True, overwrite: bool = True, show_progress: bool = True):
        """Persist both SQLite DB (already on disk) and ANN index if present."""
        # SQLite is already persisted; just save ANN index to self.index_path
        if self._ann is not None and hasattr(self._ann, 'save_index'):
            try:
                self._ann.save_index(self.index_path)
                logger.info(f"Saved ANN index to {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to save ANN index: {e}")

    def load(self):
        """Load index & retriever state from disk. If ANN index exists, load it; otherwise rely on TF-IDF fallback."""
        # Ensure DB file exists
        if not os.path.isfile(self.db_path):
            raise FileNotFoundError(f"Sqlite DB not found: {self.db_path}")

        # Load embeddings from DB if available (not implemented: for now assume embeddings are only in memory)
        # Try to load ANN index
        self._maybe_load_ann()

    def _get_embedder(self):
        """Return the SentenceTransformer embedder, loading it on first call.

        This is the single place where sentence-transformers is imported and
        the model is downloaded/loaded.  Calling it from __init__ is intentionally
        avoided so that constructing or opening an empty database does not trigger
        an HTTP round-trip to HuggingFace.
        """
        if not self._embedder_loaded:
            self._embedder_loaded = True  # set before attempt so we don't retry on failure
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
                self._embedder = SentenceTransformer(model_name)
                logger.info(f"Using sentence-transformers model for embeddings: {model_name}")
            except Exception:
                logger.info("sentence-transformers not available; falling back to TF-IDF text search for retrieval.")
                self._embedder = None
        return self._embedder

    def _maybe_load_ann(self):
        if os.path.isfile(self.index_path):
            try:
                import hnswlib
                # Attempt to infer dim from config
                dim = self.dim
                p = hnswlib.Index(space='cosine', dim=dim)
                p.load_index(self.index_path)
                self._ann = p
                # set a reasonable ef (search parameter) after loading
                try:
                    if hasattr(self._ann, 'set_ef'):
                        self._ann.set_ef(max(50, 200))
                except Exception:
                    pass
                self._use_ann = True
                logger.info(f"Loaded ANN index from {self.index_path}")
            except Exception as e:
                logger.info(f"Could not load ANN index from {self.index_path}: {e}")
                self._ann = None
                self._use_ann = False
        else:
            self._ann = None
            self._use_ann = False

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search by query string. If embedder+ANN available: use ANN; else use TF-IDF cosine similarity."""
        # If we have ANN and an embedder, use that
        if self._use_ann and self._ann is not None and self._get_embedder() is not None:
            q_vec = self._get_embedder().encode([query], convert_to_numpy=True).astype(np.float32)
            # Tune ef for search: ef should be >= top_k and larger values improve recall.
            try:
                ef_search = max(50, int(top_k * 10))
                if hasattr(self._ann, 'set_ef'):
                    self._ann.set_ef(ef_search)
            except Exception:
                pass
            try:
                k = min(top_k, len(self._ids))
                labels, distances = self._ann.knn_query(q_vec, k=k)
                # labels, distances = self._ann.knn_query(q_vec, k=top_k)
                labels = labels[0].tolist()
                distances = distances[0].tolist()
            except Exception as e:
                # If ANN search fails, log and fall back to TF-IDF
                logger.warning(f"ANN search failed ({e}). Falling back to TF-IDF search.")
                return self._tfidf_search(query, top_k=top_k)

            rows = self._fetch_by_ids(labels)
            # Prepare results as (id, score, metadata)
            results = []
            id_to_meta = {r[0]: r[2] for r in rows}
            for lbl, dist in zip(labels, distances):
                meta = id_to_meta.get(lbl, {"id": lbl})
                results.append((lbl, float(dist), meta))
            return results

        # Fall back to TF-IDF on raw texts
        return self._tfidf_search(query, top_k=top_k)

    def _tfidf_search(self, query: str, top_k: int = 5):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except Exception:
            raise ImportError("scikit-learn is required for TF-IDF fallback. Install via 'pip install scikit-learn'")

        # Load all docs
        c = self._conn.cursor()
        c.execute("SELECT id, text FROM documents")
        rows = c.fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]

        vectorizer = TfidfVectorizer().fit_transform([query] + texts)
        q_vec = vectorizer[0:1]
        doc_vecs = vectorizer[1:]
        sims = cosine_similarity(q_vec, doc_vecs)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            id_ = ids[int(idx)]
            score = float(sims[int(idx)])
            # fetch metadata
            meta = self._fetch_by_ids([id_])[0][2]
            results.append((id_, score, meta))
        return results

    def close(self):
        """Explicitly close the SQLite connection and release the file lock.

        On Windows the OS holds a file lock for the duration of the connection.
        Call this when you are done with the backend so temp-directory cleanup
        (and test teardown) can delete the file without a PermissionError.
        """
        try:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None
                logger.debug("SqliteHNSWBackend: connection closed.")
        except Exception as exc:
            logger.warning(f"Error closing SQLite connection: {exc}")


# Expose a factory-like symbol named SqliteHnswLLM for use by memory factory
class SqliteHnswLLM(SqliteHNSWBackend):
    pass