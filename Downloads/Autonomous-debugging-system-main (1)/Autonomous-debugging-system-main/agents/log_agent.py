"""
Log Analyst Agent
Searches logs for: stack traces, error signatures, frequencies, correlations.
Filters noise. Handles large logs, missing logs, repeated patterns.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from orchestrator.state import DebugState
from tools.log_parser import extract_stack_trace, LogParser
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class LogAnalystAgent:

    def __init__(self):
        self._llm = LLMClient()
        self._parser = LogParser()

    def run(self, state: DebugState) -> DebugState:
        logger.info("[LogAnalyst] Starting log analysis")

        logs = state.get("logs", "") or ""
        bug_report = state.get("bug_report", {})
        triage = state.get("triage", {})

        # ── Guard: no logs ────────────────────────────────────────────────
        if not logs.strip():
            logger.warning("[LogAnalyst] No logs to analyze")
            state["log_analysis"] = {
                "stack_traces": [],
                "primary_error": None,
                "error_patterns": [],
                "timeline": [],
                "anomalies": ["No logs provided — analysis based on bug report only"],
                "stats": {"total_lines": 0, "error_count": 0},
                "_source": "no_logs",
            }
            return state

        # ── Structural parsing (always runs, no LLM needed) ───────────────
        parsed = self._parser.parse(logs)
        parsed_dict = parsed.to_dict()

        # ── Extract primary error ─────────────────────────────────────────
        primary_error = self._identify_primary_error(parsed_dict, triage)

        # ── Extract timeline of failures ──────────────────────────────────
        timeline = self._extract_timeline(logs)

        # ── LLM enrichment (optional) ─────────────────────────────────────
        llm_insights = None
        if state.get("llm_available") and parsed_dict["stack_traces"]:
            llm_insights = self._llm_analyze(
                parsed_dict,
                bug_report,
                logs[:4000],  # cap for LLM context
            )

        # ── Merge results ─────────────────────────────────────────────────
        log_analysis = {
            **parsed_dict,
            "primary_error": primary_error,
            "timeline": timeline,
            "llm_insights": llm_insights,
            "_source": "llm+parser" if llm_insights else "parser_only",
        }

        # Reduce confidence if no stack traces found
        if not parsed_dict["stack_traces"]:
            errors = list(state.get("errors", []))
            errors.append(
                "Log analysis: no stack traces found — evidence is weak"
            )
            state["errors"] = errors

        state["log_analysis"] = log_analysis
        logger.info(
            f"[LogAnalyst] Analysis complete: "
            f"{len(parsed_dict['stack_traces'])} traces, "
            f"primary_error={primary_error.get('exception_type', 'none') if primary_error else 'none'}"
        )
        return state

    def _identify_primary_error(
        self, parsed: Dict, triage: Dict
    ) -> Dict[str, Any]:
        """
        Select the most likely primary error from all stack traces.
        Priority: frequency > severity > alignment with triage hypotheses.
        """
        traces = parsed.get("stack_traces", [])
        if not traces:
            return None

        # Use highest-frequency trace
        primary = traces[0]

        # If triage gave hypotheses, try to align
        failure_surface = triage.get("failure_surface", "")
        if failure_surface and failure_surface != "unknown":
            for trace in traces:
                frames = trace.get("all_frames", [])
                if any(failure_surface.lower() in (f.get("file") or "").lower() for f in frames):
                    primary = trace
                    break

        return primary

    def _extract_timeline(self, logs: str) -> List[Dict]:
        """Extract a chronological list of significant events."""
        timestamp_re = re.compile(
            r"(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,.]?\d*)\s+"
            r"(?P<level>ERROR|WARNING|CRITICAL|INFO)\s+"
            r"\[(?P<logger>[^\]]+)\]\s+(?P<msg>.+)"
        )
        events = []
        for m in timestamp_re.finditer(logs):
            level = m.group("level")
            if level in ("ERROR", "WARNING", "CRITICAL"):
                events.append({
                    "timestamp": m.group("ts"),
                    "level": level,
                    "logger": m.group("logger"),
                    "message": m.group("msg").strip()[:200],
                })
            if len(events) >= 50:  # cap
                break
        return events

    def _llm_analyze(
        self, parsed: Dict, bug_report: Dict, logs_snippet: str
    ) -> Dict:
        """Use LLM to identify patterns and correlations in parsed log data."""
        prompt = f"""
Analyze the following structured log analysis data and provide insights.
Return ONLY a JSON object with this structure:

{{
  "primary_cause": "one-line description of the root cause visible in logs",
  "correlated_events": ["event 1 correlation", "event 2 correlation"],
  "affected_code_paths": ["file:lineno:function", ...],
  "error_frequency_pattern": "description of when/how often errors occur",
  "secondary_issues": ["any other issues spotted"],
  "confidence": 0.0
}}

PARSED LOG DATA:
{json.dumps(parsed, indent=2)[:3000]}

BUG REPORT COMPONENT: {bug_report.get('component', 'unknown')}

RAW LOG SNIPPET:
{logs_snippet}
"""
        result = self._llm.generate(prompt)
        return result or {}
