"""
FAISS-based vector index for code chunk retrieval.
Falls back to keyword search if FAISS is unavailable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from rag.embedder import Embedder, EMBEDDING_DIM
from utils.logger import get_logger

logger = get_logger(__name__)

INDEX_SAVE_PATH = Path("outputs/faiss_index")
MAX_CHUNK_TOKENS = 150   # roughly 600 chars per chunk
CHUNK_OVERLAP = 20       # lines overlap between chunks


@dataclass
class CodeChunk:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)


class FAISSIndex:
    """
    FAISS-backed semantic search over code chunks.
    Combines:
      - Dense semantic similarity (FAISS)
      - Keyword overlap scoring (BM25-lite)
    Falls back to pure keyword if FAISS unavailable.
    """

    def __init__(self):
        self._chunks: List[CodeChunk] = []
        self._index = None         # faiss.IndexFlatIP or None
        self._embedder = Embedder()
        self._faiss_available = False
        self._try_import_faiss()

    def _try_import_faiss(self):
        try:
            import faiss  # noqa
            self._faiss_available = True
            logger.info("[FAISSIndex] FAISS available")
        except ImportError:
            logger.warning("[FAISSIndex] FAISS not available — using keyword-only search")

    def index_repository(self, repo_path: str) -> int:
        """
        Chunk and index all source files in a repository.
        Returns number of chunks indexed.
        """
        from tools.file_reader import find_files, read_file

        self._chunks = []
        files = find_files(repo_path)

        if not files:
            logger.warning(f"[FAISSIndex] No source files found in {repo_path}")
            return 0

        logger.info(f"[FAISSIndex] Indexing {len(files)} files from {repo_path}")

        all_contents = []
        for file_path in files:
            content, err = read_file(file_path)
            if err or not content:
                continue

            chunks = self._chunk_file(file_path, content)
            self._chunks.extend(chunks)
            all_contents.extend(c.content for c in chunks)

        if not self._chunks:
            return 0

        logger.info(f"[FAISSIndex] Building index over {len(self._chunks)} chunks")

        if self._faiss_available:
            self._build_faiss_index(all_contents)
        else:
            logger.info("[FAISSIndex] Skipping FAISS build — keyword search only")

        return len(self._chunks)

    def _chunk_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Split file into overlapping line-based chunks."""
        lines = content.splitlines()
        chunks = []
        step = MAX_CHUNK_TOKENS - CHUNK_OVERLAP
        step = max(step, 20)

        for i in range(0, len(lines), step):
            chunk_lines = lines[i: i + MAX_CHUNK_TOKENS]
            if not any(l.strip() for l in chunk_lines):
                continue

            chunk_content = "\n".join(chunk_lines)
            chunk_id = f"{Path(file_path).name}:{i+1}-{i+len(chunk_lines)}"

            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_line=i + 1,
                end_line=i + len(chunk_lines),
                content=chunk_content,
                language=_detect_language(file_path),
            ))

        return chunks

    def _build_faiss_index(self, contents: List[str]):
        try:
            import faiss

            vectors = self._embedder.embed(contents)
            if vectors.shape[0] == 0:
                return

            dim = vectors.shape[1]
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine if normalized
            index.add(vectors.astype(np.float32))
            self._index = index
            logger.info(f"[FAISSIndex] FAISS index built: {index.ntotal} vectors, dim={dim}")

        except Exception as e:
            logger.error(f"[FAISSIndex] FAISS build failed: {e}")
            self._faiss_available = False

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant code chunks for a query.
        Combines semantic + keyword scores.

        Returns list of dicts with chunk info + score.
        """
        if not self._chunks:
            logger.warning("[FAISSIndex] Index empty — no results")
            return []

        k = min(k, len(self._chunks))
        results = []

        # ── Semantic search ───────────────────────────────────────────────
        if self._faiss_available and self._index is not None:
            try:
                query_vec = self._embedder.embed_single(query).reshape(1, -1).astype(np.float32)
                scores, indices = self._index.search(query_vec, min(k * 3, len(self._chunks)))

                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self._chunks):
                        continue
                    chunk = self._chunks[idx]
                    kw_score = self._keyword_score(query, chunk.content)
                    combined = float(score) * 0.7 + kw_score * 0.3
                    results.append((combined, chunk))

            except Exception as e:
                logger.warning(f"[FAISSIndex] FAISS search failed: {e} — falling back to keyword")
                results = []

        # ── Keyword-only fallback ─────────────────────────────────────────
        if not results:
            for chunk in self._chunks:
                kw_score = self._keyword_score(query, chunk.content)
                if kw_score > 0:
                    results.append((kw_score, chunk))

        # Sort and deduplicate by file
        results.sort(key=lambda x: x[0], reverse=True)

        seen_files = set()
        final = []
        for score, chunk in results:
            if score < min_score:
                continue
            # Allow at most 2 chunks per file to avoid redundancy
            file_count = sum(1 for _, c in final if c.file_path == chunk.file_path)
            if file_count >= 2:
                continue
            if len(final) >= k:
                break

            seen_files.add(chunk.file_path)
            final.append((score, chunk))

        return [
            {
                "chunk_id": chunk.chunk_id,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content[:2000],   # Cap for state
                "score": round(score, 4),
                "language": chunk.language,
            }
            for score, chunk in final
        ]

    @staticmethod
    def _keyword_score(query: str, content: str) -> float:
        """Simple keyword overlap score (Jaccard-like)."""
        q_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", query.lower()))
        c_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", content.lower()))
        if not q_tokens:
            return 0.0
        intersection = q_tokens & c_tokens
        return len(intersection) / len(q_tokens)


def _detect_language(file_path: str) -> str:
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".java": "java", ".go": "go", ".rb": "ruby",
        ".rs": "rust", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    }
    return ext_map.get(Path(file_path).suffix.lower(), "unknown")


def retrieve_relevant_code(
    query: str,
    repo_path: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Public API: build or reuse index for repo_path, run retrieval.
    """
    # Module-level index cache (per-process)
    if not hasattr(retrieve_relevant_code, "_cache"):
        retrieve_relevant_code._cache = {}

    cache_key = repo_path
    if cache_key not in retrieve_relevant_code._cache:
        index = FAISSIndex()
        count = index.index_repository(repo_path)
        logger.info(f"[RAG] Indexed {count} chunks from {repo_path}")
        retrieve_relevant_code._cache[cache_key] = index

    index = retrieve_relevant_code._cache[cache_key]
    return index.retrieve(query, k=k)
