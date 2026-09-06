"""
Repo Navigator Agent
Maps stack trace frames to source files.
Uses AST extraction + RAG retrieval to gather code context.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.state import DebugState
from tools.ast_parser import extract_function, get_imports
from tools.file_reader import find_files, read_file
from rag.faiss_index import retrieve_relevant_code
from utils.logger import get_logger

logger = get_logger(__name__)


class RepoNavigatorAgent:

    def run(self, state: DebugState) -> DebugState:
        logger.info("[RepoNavigator] Starting repo navigation")

        repo_path = state.get("repo_path", "")
        log_analysis = state.get("log_analysis", {})
        triage = state.get("triage", {})
        bug_report = state.get("bug_report", {})

        # ── Guard ──────────────────────────────────────────────────────────
        if not repo_path or not os.path.isdir(repo_path):
            logger.warning(f"[RepoNavigator] Invalid repo path: {repo_path}")
            state["retrieved_code"] = []
            return state

        retrieved = []

        # ── 1. Map stack trace frames → source files ──────────────────────
        traces = log_analysis.get("stack_traces", [])
        for trace in traces[:3]:  # top 3 traces
            pf = trace.get("primary_frame")
            if pf and pf.get("file") and pf.get("lineno"):
                # Resolve file path relative to repo
                resolved = self._resolve_path(pf["file"], repo_path)
                if resolved:
                    func_info = extract_function(resolved, pf["lineno"])
                    if func_info:
                        retrieved.append({
                            "source": "ast_trace",
                            "file": resolved,
                            "function": func_info.get("function_name"),
                            "start_line": func_info.get("start_line"),
                            "end_line": func_info.get("end_line"),
                            "content": func_info.get("source_code", "")[:2000],
                            "score": 1.0,
                        })

        # ── 2. RAG retrieval using triage + bug report as query ────────────
        failure_surface = triage.get("failure_surface", "")
        description = bug_report.get("description", "")
        component = bug_report.get("component", "")

        rag_queries = [q for q in [failure_surface, component, description[:200]] if q]

        for query in rag_queries[:2]:
            try:
                chunks = retrieve_relevant_code(query, repo_path, k=3)
                for chunk in chunks:
                    # Avoid duplicating files already found via trace
                    existing_files = {r["file"] for r in retrieved}
                    if chunk["file_path"] not in existing_files:
                        retrieved.append({
                            "source": "rag",
                            "file": chunk["file_path"],
                            "function": None,
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "content": chunk["content"][:2000],
                            "score": chunk["score"],
                        })
            except Exception as e:
                logger.warning(f"[RepoNavigator] RAG retrieval error for '{query}': {e}")

        # ── 3. Direct file resolution from bug report ──────────────────────
        if component:
            direct_path = self._find_component_file(component, repo_path)
            if direct_path:
                content, err = read_file(direct_path)
                if content and direct_path not in {r["file"] for r in retrieved}:
                    retrieved.append({
                        "source": "direct",
                        "file": direct_path,
                        "function": None,
                        "start_line": 1,
                        "end_line": len(content.splitlines()),
                        "content": content[:3000],
                        "score": 0.95,
                    })

        # Sort by score, cap at 8
        retrieved.sort(key=lambda x: x["score"], reverse=True)
        retrieved = retrieved[:8]

        logger.info(f"[RepoNavigator] Retrieved {len(retrieved)} code chunks")
        state["retrieved_code"] = retrieved
        return state

    def _resolve_path(self, frame_file: str, repo_root: str) -> Optional[str]:
        """
        Resolve a stack frame file path to an actual path on disk.
        Stack traces often contain container paths like /opt/app/...
        """
        # Try as-is
        if os.path.exists(frame_file):
            return frame_file

        # Strip common container prefixes
        for prefix in ("/opt/app/", "/app/", "/home/app/", "/usr/src/app/"):
            stripped = frame_file.replace(prefix, "")
            candidate = os.path.join(repo_root, stripped)
            if os.path.exists(candidate):
                return candidate

        # Try just the filename
        fname = os.path.basename(frame_file)
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
            if fname in files:
                return os.path.join(root, fname)

        return None

    def _find_component_file(self, component: str, repo_root: str) -> Optional[str]:
        """Find a source file matching the component hint in bug report."""
        # e.g., "payments/processor.py" → direct lookup
        candidate = os.path.join(repo_root, component)
        if os.path.exists(candidate):
            return candidate

        # Try fuzzy match
        component_lower = component.lower().replace("/", os.sep)
        for root, dirs, files in os.walk(repo_root):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git")]
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, repo_root).lower()
                if component_lower in rel or component_lower.split("/")[-1] in f.lower():
                    return full

        return None
