"""
Reviewer / Critic Agent
Validates all pipeline outputs:
- Challenges weak assumptions
- Verifies repro is minimal
- Checks fix plan safety
- Flags missing edge cases
- Lowers confidence on contradictions

All field accesses are defensively guarded against None/wrong-type from LLM output.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from orchestrator.state import DebugState
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


def _sf(value: Any, default: float = 0.0) -> float:
    """Safe float coercion — handles None, str, NaN."""
    if value is None:
        return default
    try:
        r = float(value)
        return default if r != r else r   # NaN → default
    except (TypeError, ValueError):
        return default


def _sl(value: Any) -> list:
    return value if isinstance(value, list) else []


def _sd(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


class ReviewerAgent:

    def __init__(self):
        self._llm = LLMClient()

    def run(self, state: DebugState) -> DebugState:
        logger.info("[Reviewer] Starting review")

        triage       = _sd(state.get("triage"))
        log_analysis = _sd(state.get("log_analysis"))
        repro_code   = state.get("repro_code") or ""
        exec_result  = _sd(state.get("execution_result"))
        root_cause   = _sd(state.get("root_cause"))
        patch_plan   = _sd(state.get("patch_plan"))

        # ── LLM review (optional) ─────────────────────────────────────────
        llm_review = None
        if state.get("llm_available"):
            try:
                llm_review = self._llm_review(state)
            except Exception as e:
                logger.warning(f"[Reviewer] LLM review failed: {e}")

        # ── Structural validation ─────────────────────────────────────────
        critiques = self._structural_validate(
            triage, log_analysis, repro_code, exec_result, root_cause, patch_plan
        )

        # ── Repro quality ─────────────────────────────────────────────────
        repro_issues = self._check_repro_quality(repro_code, exec_result)

        # ── Fix plan safety ───────────────────────────────────────────────
        fix_issues = self._check_fix_safety(patch_plan, root_cause)

        # ── Contradiction detection ───────────────────────────────────────
        contradictions = self._detect_contradictions(
            triage, log_analysis, exec_result, root_cause
        )

        # ── Confidence adjustment ─────────────────────────────────────────
        confidence_adjustment = 0.0
        if contradictions:
            confidence_adjustment -= 0.15 * len(contradictions)
        if repro_issues:
            confidence_adjustment -= 0.05 * len(repro_issues)
        if not (root_cause.get("hypothesis") or ""):
            confidence_adjustment -= 0.2

        all_issues = critiques + repro_issues + fix_issues

        review = {
            "passed": len(all_issues) == 0 and len(contradictions) == 0,
            "critiques": critiques,
            "repro_issues": repro_issues,
            "fix_issues": fix_issues,
            "contradictions": contradictions,
            "confidence_adjustment": round(confidence_adjustment, 3),
            "open_questions": self._generate_open_questions(state),
            "edge_cases_missed": self._identify_missed_edge_cases(root_cause, patch_plan),
            "llm_review": llm_review,
            "recommendation": self._recommendation(all_issues, contradictions, exec_result),
        }

        state["review"] = review

        # Apply confidence adjustment — guard current confidence too
        current = _sf(state.get("confidence"), default=0.5)
        state["confidence"] = round(max(0.0, min(1.0, current + confidence_adjustment)), 3)

        logger.info(
            f"[Reviewer] Review complete: "
            f"passed={review['passed']}, "
            f"issues={len(all_issues)}, "
            f"contradictions={len(contradictions)}, "
            f"confidence_adj={confidence_adjustment:+.2f}"
        )
        return state

    def _llm_review(self, state: DebugState) -> Dict:
        """Ask LLM to independently review the full pipeline output."""
        prompt = f"""
You are a senior engineer reviewing an automated bug analysis.
Be critical. Challenge assumptions. Find what is missing.

Return ONLY a JSON object:
{{
  "strengths": ["what the analysis did well"],
  "weaknesses": ["what is weak or unsubstantiated"],
  "missing_evidence": ["what evidence would strengthen the analysis"],
  "alternative_hypotheses": ["other possible root causes not considered"],
  "fix_safety_concerns": ["potential problems with the proposed fix"],
  "confidence_assessment": "your overall confidence assessment as a string"
}}

TRIAGE: {json.dumps(state.get('triage') or {}, indent=2)[:500]}
ROOT CAUSE: {json.dumps(state.get('root_cause') or {}, indent=2)[:500]}
PATCH PLAN: {json.dumps(state.get('patch_plan') or {}, indent=2)[:500]}
REPRO STATUS: {_sd(state.get('execution_result')).get('status', 'unknown')}
"""
        return self._llm.generate(prompt) or {}

    def _structural_validate(
        self,
        triage: Dict,
        log_analysis: Dict,
        repro_code: str,
        exec_result: Dict,
        root_cause: Dict,
        patch_plan: Dict,
    ) -> List[str]:
        """Check for missing or empty required fields."""
        issues = []

        if not _sl(triage.get("hypotheses")):
            issues.append("WARN: No triage hypotheses generated")

        if not _sl(log_analysis.get("stack_traces")) and not _sl(log_analysis.get("error_lines")):
            issues.append("WARN: No structured error evidence extracted from logs")

        if not repro_code or len(repro_code.strip()) < 20:
            issues.append("ERROR: Repro script is empty or trivial")

        if not (root_cause.get("hypothesis") or ""):
            issues.append("ERROR: No root cause hypothesis provided")

        rc_conf = _sf(root_cause.get("confidence"), default=0.0)
        if rc_conf < 0.3:
            issues.append(f"WARN: Very low root cause confidence: {rc_conf:.2f}")

        if not _sl(patch_plan.get("files_to_change")):
            issues.append("WARN: Patch plan does not specify files to change")

        if not _sl(patch_plan.get("validation_steps")):
            issues.append("WARN: No validation steps in patch plan")

        return issues

    def _check_repro_quality(self, repro_code: str, exec_result: Dict) -> List[str]:
        """Check that repro script is minimal and actually demonstrates the bug."""
        issues = []

        if not repro_code:
            return ["ERROR: No repro code generated"]

        lines = [l for l in repro_code.splitlines() if l.strip() and not l.strip().startswith("#")]
        if len(lines) > 100:
            issues.append(f"WARN: Repro script is not minimal ({len(lines)} non-comment lines)")

        # REQUIREMENT §3: status=="fail" = AssertionError confirmed the bug (correct)
        status = exec_result.get("status") or "unknown"
        if status == "fail":
            pass   # correct — repro raised AssertionError, bug is confirmed
        elif status == "pass":
            issues.append(
                "ERROR: Repro script PASSED (exit 0) — bug NOT reproduced. "
                "Script must raise AssertionError to confirm the bug."
            )
        elif status == "timeout":
            issues.append("ERROR: Repro timed out — possible infinite loop")
        elif status == "error":
            output = exec_result.get("output") or ""
            if "modulenotfounderror" in output.lower() or "importerror" in output.lower():
                issues.append(
                    "WARN: Repro import error — ensure all imports are stdlib-only"
                )
            elif "syntaxerror" in output.lower():
                issues.append("ERROR: Repro has SyntaxError — fix the script")
            else:
                issues.append("WARN: Repro script errored before reaching assertion")
        elif status == "unknown":
            issues.append("WARN: Repro execution status unknown")

        # Explicit requirement: repro must have status=="fail"
        if status != "fail":
            issues.append(
                f"Repro did not fail — bug not confirmed (status={status!r}). "
                "Valid repro must exit non-zero via AssertionError."
            )

        # Assertion presence check
        if "assert" not in repro_code.lower() and "assertionerror" not in repro_code.lower() and "raise" not in repro_code.lower():
            issues.append("WARN: Repro has no assertions or raises — failure may not be explicit")

        return issues

    def _check_fix_safety(self, patch_plan: Dict, root_cause: Dict) -> List[str]:
        """Check fix plan for potential regressions or safety issues."""
        issues = []

        if not patch_plan:
            return ["ERROR: No patch plan provided"]

        code_change = patch_plan.get("code_change") or ""
        if not code_change or len(code_change.strip()) < 10:
            issues.append("WARN: Patch plan has no concrete code change")

        if patch_plan.get("breaking_changes") is True:
            issues.append(
                "WARN: Patch plan involves breaking changes — requires API versioning review"
            )

        if not _sl(patch_plan.get("regression_tests")):
            issues.append("WARN: No regression tests specified in patch plan")

        dangerous_patterns = [
            ("eval(", "Proposed fix uses eval() — security risk"),
            ("exec(", "Proposed fix uses exec() — security risk"),
            ("shell=True", "Proposed fix uses shell=True — command injection risk"),
            ("# TODO", "Proposed fix has unresolved TODOs"),
        ]
        for pattern, msg in dangerous_patterns:
            if pattern.lower() in code_change.lower():
                issues.append(f"ERROR: {msg}")

        return issues

    def _detect_contradictions(
        self,
        triage: Dict,
        log_analysis: Dict,
        exec_result: Dict,
        root_cause: Dict,
    ) -> List[str]:
        """Find contradictions between evidence sources."""
        contradictions = []

        exec_status = exec_result.get("status") or "unknown"
        rc_conf = _sf(root_cause.get("confidence"), default=0.0)

        # status=="fail" is CORRECT (bug confirmed) — no contradiction
        # Only flag contradiction if repro errored or passed when confidence is high
        if exec_status in ("error", "pass") and rc_conf > 0.85:
            contradictions.append(
                f"Repro {exec_status!r} but root cause confidence is {rc_conf:.0%} — "
                "high confidence not justified without confirmed repro (AssertionError)"
            )

        # No stack traces but high confidence
        traces = _sl(log_analysis.get("stack_traces"))
        if not traces and rc_conf > 0.7:
            contradictions.append(
                "No stack traces in logs but root cause confidence is high — "
                "confidence should be reduced without log evidence"
            )

        # Exception type not referenced in hypothesis
        primary = _sd(log_analysis.get("primary_error"))
        exc_type = primary.get("exception_type") or ""
        root_hypothesis = (root_cause.get("hypothesis") or "").lower()
        affected = (root_cause.get("affected_component") or "").lower()
        if exc_type and exc_type not in ("GenericError", "UnknownException"):
            if exc_type.lower() not in root_hypothesis and exc_type.lower() not in affected:
                contradictions.append(
                    f"Primary exception '{exc_type}' not referenced in root cause hypothesis — "
                    "hypothesis may not explain the observed error"
                )

        return contradictions

    def _generate_open_questions(self, state: DebugState) -> List[str]:
        """Generate open questions for the engineering team."""
        questions = []
        br = _sd(state.get("bug_report"))
        root_cause = _sd(state.get("root_cause"))

        if not br.get("environment"):
            questions.append("What is the exact Python version and platform?")

        if not (state.get("repo_path") or ""):
            questions.append("Can the repo be provided for deeper static analysis?")

        if _sd(state.get("execution_result")).get("status") != "fail":
            questions.append("Can you provide steps to reliably reproduce this in a local env?")

        affected = root_cause.get("affected_component") or ""
        if affected and "unknown" not in affected:
            func = affected.split(":")[-1]
            questions.append(f"Are there other callers of {func}()?")

        impact = _sd(br.get("impact"))
        if impact.get("customers_affected"):
            questions.append(
                f"Have the {impact['customers_affected']} affected customers been notified?"
            )

        questions.append("Is there a rollback plan if the fix introduces regressions?")
        related = _sl(br.get("related_issues"))
        if related:
            questions.append(
                f"Have the related issues {', '.join(str(r) for r in related)} been reviewed for similar patterns?"
            )

        return questions[:8]

    def _identify_missed_edge_cases(self, root_cause: Dict, patch_plan: Dict) -> List[str]:
        """Suggest edge cases the fix should cover."""
        edge_cases = []
        hypothesis = (root_cause.get("hypothesis") or "").lower()
        code_change = (patch_plan.get("code_change") or "").lower()

        if "float" in hypothesis or "decimal" in hypothesis or "precision" in hypothesis:
            edge_cases += [
                "Test with amount=0.01 (minimum) — int(Decimal('0.01') * 100) = 1",
                "Test with amount=999999.99 (near maximum)",
                "Test JPY (multiplier=1) — no subunit conversion needed",
                "Test EUR and GBP amounts to ensure multiplier consistency",
                "Test negative amounts (should be rejected before reaching to_minor_units)",
            ]
        if "thread" in hypothesis or "race" in hypothesis:
            edge_cases += [
                "Test with pool_size=1 and 50 concurrent threads",
                "Test acquire() after connection is retired (stale)",
                "Test pool growth to max_connections under load",
            ]
        if "jwt" in hypothesis or "token" in hypothesis:
            edge_cases += [
                "Test token at exactly exp - 0s (boundary)",
                "Test token at exp + 1s (should be rejected)",
                "Test with clock skew between issuer and verifier",
            ]

        return edge_cases[:6]

    @staticmethod
    def _recommendation(
        issues: List[str], contradictions: List[str], exec_result: Dict
    ) -> str:
        errors = [i for i in issues if i.startswith("ERROR")]
        warnings = [i for i in issues if i.startswith("WARN")]

        if errors:
            return (
                f"DO NOT SHIP: {len(errors)} critical issue(s) found. "
                "Address all ERROR items before proceeding."
            )
        if contradictions:
            return (
                f"REVIEW REQUIRED: {len(contradictions)} contradiction(s) detected. "
                "Reduce confidence and gather more evidence."
            )
        if warnings:
            return (
                f"PROCEED WITH CAUTION: {len(warnings)} warning(s). "
                "Address warnings before production deployment."
            )

        status = _sd(exec_result).get("status") or "unknown"
        if status == "fail":
            return "READY FOR REVIEW: Repro confirmed, root cause plausible. Conduct code review before merging."

        return "NEEDS MORE EVIDENCE: Pipeline completed but repro was not confirmed. Manual investigation recommended."
