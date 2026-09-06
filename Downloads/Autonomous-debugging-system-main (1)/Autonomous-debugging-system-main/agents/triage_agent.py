"""
Triage Agent
Extracts: symptoms, expected vs actual behavior, environment, hypotheses.
Handles: missing fields, ambiguous reports, incomplete context.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from orchestrator.state import DebugState
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


class TriageAgent:

    def __init__(self):
        self._llm = LLMClient()

    def run(self, state: DebugState) -> DebugState:
        logger.info("[TriageAgent] Starting triage analysis")

        bug_report = state.get("bug_report", {})
        logs_preview = (state.get("logs", "") or "")[:2000]

        # ── Try LLM-based triage ──────────────────────────────────────────
        if state.get("llm_available"):
            result = self._llm_triage(bug_report, logs_preview)
            if result:
                logger.info("[TriageAgent] LLM triage succeeded")
                state["triage"] = self._sanitize_triage(result)
                return state

        # ── Heuristic fallback ─────────────────────────────────────────────
        logger.info("[TriageAgent] Using heuristic triage fallback")
        state["triage"] = self._heuristic_triage(bug_report, logs_preview)
        return state

    def _llm_triage(self, bug_report: Dict, logs_preview: str) -> Dict:
        prompt = f"""
Analyze the following bug report and logs preview. 
Return ONLY a JSON object with this exact structure:

{{
  "title": "short bug title",
  "severity": "critical|high|medium|low",
  "symptoms": ["symptom 1", "symptom 2"],
  "expected_behavior": "what should happen",
  "actual_behavior": "what actually happens",
  "environment": {{
    "language": "python|java|node|etc",
    "version": "version string or null",
    "os": "os string or null",
    "framework": "framework or null"
  }},
  "failure_surface": "module/component most likely involved",
  "hypotheses": [
    {{"id": "H1", "description": "hypothesis text", "confidence": 0.8, "reasoning": "why"}},
    {{"id": "H2", "description": "alternative hypothesis", "confidence": 0.4, "reasoning": "why"}}
  ],
  "missing_info": ["what additional info would help"],
  "urgency": "immediate|high|normal|low"
}}

BUG REPORT:
{json.dumps(bug_report, indent=2)}

LOGS PREVIEW:
{logs_preview}
"""
        return self._llm.generate(prompt)

    def _heuristic_triage(self, bug_report: Dict, logs_preview: str) -> Dict:
        """Rule-based triage when LLM unavailable."""
        title = bug_report.get("title", "Unknown bug")
        description = bug_report.get("description", "")
        expected = bug_report.get("expected_behavior", "")
        actual = bug_report.get("actual_behavior", "")
        severity = bug_report.get("severity", "MEDIUM").lower()
        environment = bug_report.get("environment", {})
        hints = bug_report.get("reproduction_hints", [])

        # Extract symptoms from description + actual behavior
        symptoms = []
        if actual:
            symptoms.append(actual[:200])
        if "error" in description.lower():
            symptoms.append("Error condition detected")
        if "fail" in description.lower():
            symptoms.append("Failure condition detected")
        if "timeout" in description.lower():
            symptoms.append("Timeout condition")
        if "memory" in description.lower():
            symptoms.append("Memory issue")

        if not symptoms:
            symptoms = [f"Issue described: {description[:200]}"] if description else ["No symptoms extracted"]

        # Determine failure surface from hints/description
        failure_surface = self._infer_failure_surface(description, hints, bug_report)

        # Generate hypotheses from hints
        hypotheses = self._generate_hypotheses(bug_report, hints, logs_preview)

        # Missing info
        missing_info = []
        if not environment:
            missing_info.append("Environment details (language/runtime version)")
        if not hints:
            missing_info.append("Reproduction steps")
        if not logs_preview.strip():
            missing_info.append("Error logs or stack traces")

        return {
            "title": title,
            "severity": severity,
            "symptoms": symptoms,
            "expected_behavior": expected or "Not specified",
            "actual_behavior": actual or "Not specified",
            "environment": {
                "language": environment.get("python_version", environment.get("language", "unknown")),
                "version": str(environment.get("python_version", environment.get("version", "unknown"))),
                "os": environment.get("os", "unknown"),
                "framework": environment.get("framework", "unknown"),
            },
            "failure_surface": failure_surface,
            "hypotheses": hypotheses,
            "missing_info": missing_info,
            "urgency": self._map_severity_to_urgency(severity),
            "_source": "heuristic",
        }

    @staticmethod
    def _infer_failure_surface(description: str, hints: List[str], bug_report: Dict) -> str:
        """Infer the most likely affected component."""
        component = bug_report.get("component", "")
        if component:
            return component

        text = " ".join([description] + hints).lower()

        surface_keywords = {
            "database": ["db", "database", "query", "connection", "pool", "sql"],
            "authentication": ["auth", "token", "jwt", "login", "session", "permission"],
            "payment": ["payment", "transaction", "amount", "currency", "charge", "refund"],
            "api": ["api", "endpoint", "request", "response", "http", "route"],
            "cache": ["cache", "redis", "memcache", "ttl", "eviction"],
            "networking": ["network", "timeout", "connection refused", "socket"],
        }

        scores = {}
        for surface, keywords in surface_keywords.items():
            scores[surface] = sum(1 for kw in keywords if kw in text)

        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best

        return "unknown"

    @staticmethod
    def _generate_hypotheses(bug_report: Dict, hints: List[str], logs: str) -> List[Dict]:
        hypotheses = []

        hints_text = " ".join(hints).lower()
        description = bug_report.get("description", "").lower()
        combined = hints_text + " " + description + " " + logs.lower()

        # Float precision hypothesis
        if any(kw in combined for kw in ["float", "decimal", "precision", "minor", "amount", "cents"]):
            hypotheses.append({
                "id": "H1",
                "description": "Floating-point precision loss when converting Decimal to minor currency units",
                "confidence": 0.9,
                "reasoning": "Bug report and hints explicitly mention float precision and to_minor_units()",
            })

        # Race condition hypothesis
        if any(kw in combined for kw in ["race", "concurrent", "thread", "lock", "pool"]):
            hypotheses.append({
                "id": "H2",
                "description": "Race condition in connection pool or shared resource access",
                "confidence": 0.6,
                "reasoning": "Log mentions concurrent access patterns",
            })

        # Wrong type hypothesis
        if any(kw in combined for kw in ["type", "cast", "convert", "int(", "str("]):
            hypotheses.append({
                "id": "H3",
                "description": "Type conversion error causing incorrect value computation",
                "confidence": 0.7,
                "reasoning": "Hints suggest int() or float() conversion issue",
            })

        # Generic fallback
        if not hypotheses:
            hypotheses.append({
                "id": "H1",
                "description": f"Logic error in {bug_report.get('component', 'unknown component')}",
                "confidence": 0.3,
                "reasoning": "Insufficient evidence for specific hypothesis",
            })

        return hypotheses

    @staticmethod
    def _map_severity_to_urgency(severity: str) -> str:
        return {
            "critical": "immediate",
            "high": "high",
            "medium": "normal",
            "low": "low",
        }.get(severity.lower(), "normal")

    @staticmethod
    def _sanitize_triage(result: Dict) -> Dict:
        """
        Normalize LLM triage output so downstream agents never see None on numeric fields.
        The LLM may return confidence=null, missing lists, string numbers, etc.
        """
        if not isinstance(result, dict):
            return result

        # Normalize hypotheses list
        hypotheses = result.get("hypotheses")
        if isinstance(hypotheses, list):
            clean = []
            for h in hypotheses:
                if not isinstance(h, dict):
                    continue
                raw_conf = h.get("confidence")
                try:
                    conf = float(raw_conf) if raw_conf is not None else 0.5
                    conf = max(0.0, min(1.0, conf))
                except (TypeError, ValueError):
                    conf = 0.5
                h["confidence"] = conf
                # Ensure required string fields exist
                h.setdefault("id", f"H{len(clean)+1}")
                h.setdefault("description", "Unknown hypothesis")
                h.setdefault("reasoning", "")
                clean.append(h)
            result["hypotheses"] = clean

        # Ensure list fields are lists
        for list_field in ("symptoms", "missing_info"):
            val = result.get(list_field)
            if val is None:
                result[list_field] = []
            elif isinstance(val, str):
                result[list_field] = [val]

        # Ensure string fields are strings
        for str_field in ("title", "severity", "expected_behavior", "actual_behavior",
                          "failure_surface", "urgency"):
            val = result.get(str_field)
            if val is None:
                result[str_field] = ""
            elif not isinstance(val, str):
                result[str_field] = str(val)

        # Ensure environment is a dict
        if not isinstance(result.get("environment"), dict):
            result["environment"] = {}

        return result
