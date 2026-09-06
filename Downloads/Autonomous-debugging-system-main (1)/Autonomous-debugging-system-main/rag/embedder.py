"""
Code Embedder for RAG retrieval.
Uses sentence-transformers locally. Falls back to TF-IDF if model unavailable.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CACHE_DIR = Path("outputs/embedding_cache")


class Embedder:
    """
    Produces dense embeddings for code chunks.
    Primary: sentence-transformers (all-MiniLM-L6-v2)
    Fallback: TF-IDF sparse vectors (always available)
    """

    def __init__(self):
        self._model = None
        self._tfidf = None
        self._use_tfidf = False
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> bool:
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[Embedder] Loading {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("[Embedder] Model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"[Embedder] Cannot load sentence-transformers: {e}. Using TF-IDF fallback.")
            self._use_tfidf = True
            return False

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings.
        Returns numpy array of shape (N, dim).
        """
        if not texts:
            return np.zeros((0, EMBEDDING_DIM))

        # Try neural embedding
        if not self._use_tfidf and self._load_model():
            try:
                vectors = self._model.encode(
                    texts,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
                return np.array(vectors, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Embedder] Neural embed failed: {e} — switching to TF-IDF")
                self._use_tfidf = True

        # TF-IDF fallback
        return self._tfidf_embed(texts)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """
        TF-IDF based embedding fallback.
        Produces fixed-dim vectors via hashing trick.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        import scipy.sparse

        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                max_features=EMBEDDING_DIM,
                sublinear_tf=True,
                analyzer="word",
                token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]{1,}",
            )

        try:
            if not hasattr(self._tfidf, "vocabulary_"):
                # Not fitted yet
                matrix = self._tfidf.fit_transform(texts)
            else:
                matrix = self._tfidf.transform(texts)

            dense = matrix.toarray().astype(np.float32)
            # Pad/truncate to EMBEDDING_DIM
            if dense.shape[1] < EMBEDDING_DIM:
                pad = np.zeros((dense.shape[0], EMBEDDING_DIM - dense.shape[1]), dtype=np.float32)
                dense = np.hstack([dense, pad])
            elif dense.shape[1] > EMBEDDING_DIM:
                dense = dense[:, :EMBEDDING_DIM]

            # L2 normalize
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            return dense / norms

        except Exception as e:
            logger.error(f"[Embedder] TF-IDF failed: {e} — returning zeros")
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    @staticmethod
    def cache_key(texts: List[str]) -> str:
        content = "||".join(texts[:100])
        return hashlib.md5(content.encode()).hexdigest()
