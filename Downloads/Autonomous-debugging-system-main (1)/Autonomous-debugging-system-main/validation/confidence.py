"""
Confidence Scorer — revised scoring aligned with requirements.

  repro status=="fail"  → +0.35  (AssertionError confirmed the bug)
  repro status!="fail"  → −0.20  (not confirmed)
  multi-file evidence   → +0.05  (more coverage)
  root_causes list      → +0.05  (structured multi-file analysis)
"""
from __future__ import annotations
from typing import Any, Dict
from utils.logger import get_logger

logger = get_logger(__name__)


def _sf(v: Any, default: float = 0.0, lo: float = 0.0, hi: float = 1.0) -> float:
    if v is None: return default
    try:
        r = float(v)
        return default if r != r else max(lo, min(hi, r))
    except (TypeError, ValueError):
        return default

def _sl(v: Any) -> list:
    return v if isinstance(v, list) else []

def _sd(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


class ConfidenceScorer:

    def compute(self, state: Dict) -> float:
        score = 0.0

        # ── Log evidence (+0.25) ──────────────────────────────────────────
        log_analysis = _sd(state.get("log_analysis"))
        traces = _sl(log_analysis.get("stack_traces"))
        if traces:
            score += 0.20
            freq = _sf(_sd(traces[0]).get("frequency", 1), default=1.0, lo=0, hi=9999)
            if freq >= 3:
                score += 0.05

        # ── Repro success (+0.35 / −0.20) — REQUIREMENT §4 ────────────────
        exec_result = _sd(state.get("execution_result"))
        exec_status = exec_result.get("status") or "unknown"
        if exec_status == "fail":
            score += 0.35   # AssertionError confirmed the bug
        elif exec_status == "crash":
            score += 0.20
        elif exec_status == "error":
            score += 0.03   # script broken but some signal
        else:
            # "pass" or "unknown" → bug NOT reproduced
            score -= 0.20

        # ── Code alignment (+0.15) ────────────────────────────────────────
        retrieved = _sl(state.get("retrieved_code"))
        if retrieved:
            score += 0.10
            if any(_sd(c).get("source") == "ast_trace" for c in retrieved):
                score += 0.05

        # ── Hypothesis quality (+0.10) ────────────────────────────────────
        triage = _sd(state.get("triage"))
        hypotheses = _sl(triage.get("hypotheses"))
        if hypotheses:
            score += 0.03
            top_conf = _sf(_sd(hypotheses[0]).get("confidence"), default=0.0)
            score += top_conf * 0.07

        # ── Root cause quality (+0.10+bonus) ─────────────────────────────
        rc = _sd(state.get("root_cause"))
        rc_hypothesis = rc.get("hypothesis") or ""
        if isinstance(rc_hypothesis, str) and rc_hypothesis and "unknown" not in rc_hypothesis.lower():
            score += 0.05
        rc_bug_type = rc.get("bug_type") or ""
        if isinstance(rc_bug_type, str) and rc_bug_type not in ("logic_error", "unknown", "other", ""):
            score += 0.05
        rc_conf = _sf(rc.get("confidence"), default=0.0)
        if rc_conf >= 0.80:
            score += 0.03

        # ── Multi-file evidence bonus (+0.05) — REQUIREMENT §6 ───────────
        root_causes = _sl(state.get("root_causes"))
        if len(root_causes) >= 2:
            score += 0.05   # structured multi-file analysis present

        # ── Contradictions (−0.15 each) ───────────────────────────────────
        review = _sd(state.get("review"))
        contradictions = _sl(review.get("contradictions"))
        score -= 0.15 * len(contradictions)

        # ── Hard pipeline errors (−0.05 each) ────────────────────────────
        all_errors = _sl(state.get("errors"))
        hard = [e for e in all_errors if isinstance(e, str)
                and not e.startswith("WARNING") and not e.startswith("WARN")]
        score -= 0.05 * len(hard)

        # ── LLM penalty (−0.03 if unavailable) ───────────────────────────
        if not state.get("llm_available"):
            score -= 0.03

        final = round(max(0.0, min(1.0, score)), 3)
        logger.info(f"[ConfidenceScorer] score={final:.3f} (repro={exec_status})")
        return final
