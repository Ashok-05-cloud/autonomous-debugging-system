"""
Output Formatter
Converts final DebugState into the required structured JSON report.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = Path("outputs/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_output(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the canonical output JSON from pipeline state.
    Matches required output schema exactly.
    """
    bug_report  = state.get("bug_report", {})
    triage      = state.get("triage", {})
    log_analysis= state.get("log_analysis", {})
    exec_result = state.get("execution_result", {})
    root_cause  = state.get("root_cause", {})
    patch_plan  = state.get("patch_plan", {})
    review      = state.get("review", {})
    # Guard: confidence may be None if finalize agent errored
    _raw_conf = state.get("confidence")
    try:
        confidence = float(_raw_conf) if _raw_conf is not None else 0.0
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    # ── Bug summary ───────────────────────────────────────────────────────
    bug_summary = {
        "title": bug_report.get("title", triage.get("title", "Unknown bug")),
        "severity": triage.get("severity", bug_report.get("severity", "unknown")),
        "symptoms": triage.get("symptoms", []),
        "expected_behavior": triage.get("expected_behavior", ""),
        "actual_behavior": triage.get("actual_behavior", ""),
        "environment": triage.get("environment", bug_report.get("environment", {})),
        "failure_surface": triage.get("failure_surface", "unknown"),
        "scope": bug_report.get("component", "unknown"),
        "impact": bug_report.get("impact", {}),
    }

    # ── Evidence ──────────────────────────────────────────────────────────
    evidence = []

    # Stack traces
    for trace in (log_analysis.get("stack_traces") or [])[:5]:
        pf = trace.get("primary_frame") or {}
        evidence.append({
            "type": "stack_trace",
            "exception": trace.get("exception_type"),
            "message": trace.get("message", "")[:300],
            "file": pf.get("file"),
            "line": pf.get("lineno"),
            "function": pf.get("function"),
            "frequency": trace.get("frequency", 1),
        })

    # Error log lines
    for err_line in (log_analysis.get("error_lines") or [])[:5]:
        evidence.append({
            "type": "log_error",
            "timestamp": err_line.get("timestamp"),
            "message": err_line.get("message", "")[:200],
            "logger": err_line.get("logger"),
        })

    # Anomalies
    for anomaly in (log_analysis.get("anomalies") or []):
        evidence.append({"type": "anomaly", "description": anomaly})

    # ── Repro ─────────────────────────────────────────────────────────────
    repro_file = state.get("repro_file", "")
    repro = {
        "file": repro_file,
        "command": f"python {repro_file}" if repro_file else "N/A",
        "status": exec_result.get("status", "unknown"),
        "exit_code": exec_result.get("exit_code", -1),
        "output": (exec_result.get("output") or "")[:1000],
        "duration_s": exec_result.get("duration_s", 0),
        "bug_confirmed": exec_result.get("status") in ("fail", "crash"),
    }

    # ── Root cause (primary + multi-file list) ───────────────────────────
    def _safe_conf(v, fallback):
        try: return max(0.0, min(1.0, float(v))) if v is not None else fallback
        except: return fallback

    root_cause_out = {
        "hypothesis": root_cause.get("hypothesis") or "Not determined",
        "mechanism": root_cause.get("mechanism") or "",
        "confidence": _safe_conf(root_cause.get("confidence"), confidence),
        "affected_component": root_cause.get("affected_component") or "",
        "bug_type": root_cause.get("bug_type") or "unknown",
        "evidence_alignment": root_cause.get("evidence_alignment") or "",
    }

    # Multi-file root causes list
    root_causes_list = state.get("root_causes") or []

    # ── Patch plan ────────────────────────────────────────────────────────
    patch_plan_out = {
        "summary": patch_plan.get("summary", "No patch plan generated"),
        "files_to_change": patch_plan.get("files_to_change", []),
        "code_change": patch_plan.get("code_change", ""),
        "risks": patch_plan.get("risks", []),
        "breaking_changes": patch_plan.get("breaking_changes", False),
    }

    # ── Validation plan ───────────────────────────────────────────────────
    validation_plan = {
        "testing_approach": patch_plan.get("testing_approach", ""),
        "validation_steps": patch_plan.get("validation_steps", []),
        "regression_tests": patch_plan.get("regression_tests", []),
        "edge_cases": review.get("edge_cases_missed", []),
        "sanity_checks": state.get("validation_checks", {}),
    }

    # ── Open questions ─────────────────────────────────────────────────────
    open_questions = review.get("open_questions", [])

    # ── Pipeline metadata ─────────────────────────────────────────────────
    metadata = {
        "pipeline_confidence": confidence,
        "llm_used": state.get("llm_available", False),
        "agent_trace": state.get("agent_trace", []),
        "pipeline_errors": state.get("errors", []),
        "review_passed": review.get("passed", False),
        "review_recommendation": review.get("recommendation", ""),
        "critiques": review.get("critiques", []),
        "contradictions": review.get("contradictions", []),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    output = {
        "bug_summary": bug_summary,
        "evidence": evidence,
        "repro": repro,
        "root_cause": root_cause_out,
        "patch_plan": patch_plan_out,
        "validation_plan": validation_plan,
        "open_questions": open_questions,
        "root_causes": root_causes_list,
        "metadata": metadata,
    }

    return output


def save_output(output: Dict, filename: str = None) -> str:
    """Save output JSON to disk, return file path."""
    if not filename:
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"debug_report_{ts}.json"

    path = OUTPUT_DIR / filename
    path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    logger.info(f"[OutputFormatter] Report saved to {path}")
    return str(path)
