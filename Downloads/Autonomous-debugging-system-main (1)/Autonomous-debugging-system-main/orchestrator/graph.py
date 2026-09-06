"""
Orchestrator: LangGraph-based multi-agent pipeline coordinator.

Flow:
  preprocess → triage → log_analyst → repo_navigator → repro → fix_planner → reviewer → finalize

Edge cases:
- Every node is wrapped in error isolation — node failures don't crash the pipeline
- State is validated before and after each agent
- Retry logic for reproduction stage
"""
from __future__ import annotations

import logging
import time
import traceback
from typing import Any, Dict

from langgraph.graph import StateGraph, END

from orchestrator.state import DebugState
from agents.triage_agent import TriageAgent
from agents.log_agent import LogAnalystAgent
from agents.repro_agent import ReproductionAgent
from agents.fix_agent import FixPlannerAgent
from agents.reviewer_agent import ReviewerAgent
from agents.repo_navigator import RepoNavigatorAgent
from validation.sanity_checks import SanityChecker
from validation.confidence import ConfidenceScorer
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Node wrappers ────────────────────────────────────────────────────────────

def _safe_node(name: str, agent_callable):
    """
    Wraps an agent call so that:
    1. Exceptions are caught and logged (never crash pipeline)
    2. Execution time is recorded in agent_trace
    3. State validity is checked post-call
    """
    def node(state: DebugState) -> DebugState:
        start = time.time()
        trace_entry = {"agent": name, "start": start, "status": "running"}

        try:
            logger.info(f"[ORCHESTRATOR] Starting agent: {name}")
            result = agent_callable(state)

            duration = time.time() - start
            trace_entry.update({"status": "success", "duration_s": round(duration, 3)})
            logger.info(f"[ORCHESTRATOR] Agent {name} completed in {duration:.2f}s")

            # Merge agent trace
            result["agent_trace"] = list(state.get("agent_trace", [])) + [trace_entry]
            return result

        except Exception as exc:
            duration = time.time() - start
            error_msg = f"Agent {name} raised {type(exc).__name__}: {exc}"
            tb = traceback.format_exc()

            logger.error(f"[ORCHESTRATOR] {error_msg}\n{tb}")

            trace_entry.update({
                "status": "error",
                "duration_s": round(duration, 3),
                "error": error_msg,
                "traceback": tb,
            })

            errors = list(state.get("errors", []))
            errors.append(error_msg)

            # Return state with error appended — pipeline continues
            return {
                **state,
                "errors": errors,
                "agent_trace": list(state.get("agent_trace", [])) + [trace_entry],
            }

    node.__name__ = name
    return node


# ─── Conditional routing ─────────────────────────────────────────────────────

def _should_retry_repro(state: DebugState) -> str:
    """Route to retry or continue based on repro success."""
    exec_result = state.get("execution_result", {})
    retry_count = state.get("retry_count", 0)

    # If repro already failed and we haven't retried yet → retry once
    if exec_result.get("status") == "error" and retry_count < 1:
        logger.info("[ORCHESTRATOR] Repro failed, routing to retry")
        return "retry_repro"

    return "fix_planner"


def _should_run_repo_nav(state: DebugState) -> str:
    """Only run repo_navigator if a repo path is provided."""
    if state.get("repo_path", "").strip():
        return "repo_navigator"
    logger.info("[ORCHESTRATOR] No repo path — skipping repo_navigator")
    return "repro"


# ─── Agent instances ─────────────────────────────────────────────────────────

_triage_agent = TriageAgent()
_log_agent = LogAnalystAgent()
_repro_agent = ReproductionAgent()
_fix_agent = FixPlannerAgent()
_reviewer_agent = ReviewerAgent()
_repo_nav = RepoNavigatorAgent()
_sanity = SanityChecker()
_confidence = ConfidenceScorer()


def _node_preprocess(state: DebugState) -> DebugState:
    """Validate and normalize inputs before starting agent pipeline."""
    logger.info("[ORCHESTRATOR] Preprocessing inputs")

    errors = list(state.get("errors", []))

    # Validate bug report
    br = state.get("bug_report", {})
    if not br:
        errors.append("WARNING: Empty bug report — continuing with minimal context")
        state["bug_report"] = {"title": "Unknown bug", "description": "", "environment": {}}

    # Validate logs
    logs = state.get("logs", "")
    if not logs or len(logs.strip()) < 10:
        errors.append("WARNING: No logs provided — will rely on bug report only")
        state["logs"] = ""

    # Check LLM availability
    from utils.llm_client import LLMClient
    client = LLMClient()
    state["llm_available"] = client.is_available()
    if not state["llm_available"]:
        errors.append("WARNING: Ollama/LLM not available — using heuristic fallbacks")
        logger.warning("[ORCHESTRATOR] LLM not available, using fallback analysis")

    state["errors"] = errors
    return state


def _node_finalize(state: DebugState) -> DebugState:
    """Run validation and compute final confidence score."""
    logger.info("[ORCHESTRATOR] Finalizing — running sanity checks")

    # Sanity checks — wrapped so a crash here never kills the pipeline
    try:
        checks = _sanity.run_all(state)
        state["validation_checks"] = checks
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Sanity checks failed: {e}")
        state["validation_checks"] = {"error": str(e)}

    # Confidence scoring — wrapped so a crash here never kills the pipeline
    try:
        state["confidence"] = _confidence.compute(state)
    except Exception as e:
        logger.error(f"[ORCHESTRATOR] Confidence scoring failed: {e}")
        # Fall back to a safe estimate based on repro status
        exec_status = (state.get("execution_result") or {}).get("status", "unknown")
        state["confidence"] = 0.35 if exec_status == "fail" else 0.10

    conf = state.get("confidence") or 0.0
    try:
        logger.info(f"[ORCHESTRATOR] Final confidence: {float(conf):.2f}")
    except Exception:
        logger.info(f"[ORCHESTRATOR] Final confidence: {conf}")
    return state


def _node_retry_repro(state: DebugState) -> DebugState:
    """Increment retry counter and re-run repro agent with fallback mode."""
    state["retry_count"] = state.get("retry_count", 0) + 1
    logger.info(f"[ORCHESTRATOR] Repro retry attempt #{state['retry_count']}")
    state["_repro_fallback"] = True  # signal to repro agent
    return _repro_agent.run(state)


# ─── Build graph ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(DebugState)

    # Register nodes
    graph.add_node("preprocess", _safe_node("preprocess", _node_preprocess))
    graph.add_node("triage", _safe_node("triage", _triage_agent.run))
    graph.add_node("log_analyst", _safe_node("log_analyst", _log_agent.run))
    graph.add_node("repo_navigator", _safe_node("repo_navigator", _repo_nav.run))
    graph.add_node("repro", _safe_node("repro", _repro_agent.run))
    graph.add_node("retry_repro", _safe_node("retry_repro", _node_retry_repro))
    graph.add_node("fix_planner", _safe_node("fix_planner", _fix_agent.run))
    graph.add_node("reviewer", _safe_node("reviewer", _reviewer_agent.run))
    graph.add_node("finalize", _safe_node("finalize", _node_finalize))

    # Linear edges
    graph.set_entry_point("preprocess")
    graph.add_edge("preprocess", "triage")
    graph.add_edge("triage", "log_analyst")

    # Conditional: repo navigator
    graph.add_conditional_edges(
        "log_analyst",
        _should_run_repo_nav,
        {"repo_navigator": "repo_navigator", "repro": "repro"},
    )
    graph.add_edge("repo_navigator", "repro")

    # Conditional: repro retry
    graph.add_conditional_edges(
        "repro",
        _should_retry_repro,
        {"retry_repro": "retry_repro", "fix_planner": "fix_planner"},
    )
    graph.add_edge("retry_repro", "fix_planner")

    graph.add_edge("fix_planner", "reviewer")
    graph.add_edge("reviewer", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def run_pipeline(
    bug_report: Dict[str, Any],
    logs: str,
    repo_path: str = "",
) -> DebugState:
    """
    Entry point for the full debugging pipeline.
    Always returns a DebugState even on total failure.
    """
    from orchestrator.state import initial_state

    logger.info("=" * 60)
    logger.info("[ORCHESTRATOR] Starting AI Debugging Pipeline")
    logger.info("=" * 60)

    state = initial_state(bug_report, logs, repo_path)
    graph = build_graph()

    try:
        final_state = graph.invoke(state)
        logger.info("[ORCHESTRATOR] Pipeline completed successfully")
        return final_state
    except Exception as exc:
        logger.error(f"[ORCHESTRATOR] Pipeline failed catastrophically: {exc}")
        logger.error(traceback.format_exc())
        # Return whatever state we have with the error recorded
        return {
            **state,
            "errors": state.get("errors", []) + [f"Pipeline crash: {exc}"],
            "confidence": 0.0,
        }
