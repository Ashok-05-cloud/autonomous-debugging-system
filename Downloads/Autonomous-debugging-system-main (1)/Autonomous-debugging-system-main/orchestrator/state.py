"""
Shared state definition for the multi-agent debugging pipeline.
All agents read and write to this state structure.
"""
from typing import Any, Dict, List, Optional, TypedDict


class DebugState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    bug_report: Dict[str, Any]       # Parsed bug report (JSON/Markdown)
    logs: str                         # Raw log content
    repo_path: str                    # Path to repo on disk (optional)

    # ── Triage outputs ──────────────────────────────────────────────────────
    triage: Dict[str, Any]           # Structured triage analysis

    # ── Log analysis outputs ─────────────────────────────────────────────────
    log_analysis: Dict[str, Any]     # Stack traces, error signatures, patterns

    # ── RAG / code retrieval ─────────────────────────────────────────────────
    retrieved_code: List[Dict]       # Top-k relevant code chunks

    # ── Reproduction ─────────────────────────────────────────────────────────
    repro_code: str                  # Generated repro script content
    repro_file: str                  # Path to saved repro file
    execution_result: Dict[str, Any] # Output from running repro

    # ── Root cause & fix ─────────────────────────────────────────────────────
    root_cause: Dict[str, Any]       # Hypothesis with confidence
    patch_plan: Dict[str, Any]       # Proposed fix and validation

    # ── Review ───────────────────────────────────────────────────────────────
    review: Dict[str, Any]           # Reviewer agent critique

    # ── Pipeline metadata ────────────────────────────────────────────────────
    confidence: float                 # Overall pipeline confidence [0,1]
    errors: List[str]                 # Non-fatal errors encountered
    agent_trace: List[Dict]          # Per-agent execution trace
    retry_count: int                  # Repro retry attempts
    llm_available: bool              # Whether Ollama/LLM is reachable


def initial_state(
    bug_report: Dict,
    logs: str,
    repo_path: str = "",
) -> DebugState:
    """Create a fresh state for a new debugging session."""
    return DebugState(
        bug_report=bug_report,
        logs=logs,
        repo_path=repo_path,
        triage={},
        log_analysis={},
        retrieved_code=[],
        repro_code="",
        repro_file="",
        execution_result={},
        root_cause={},
        patch_plan={},
        review={},
        confidence=0.0,
        errors=[],
        agent_trace=[],
        retry_count=0,
        llm_available=False,
    )
