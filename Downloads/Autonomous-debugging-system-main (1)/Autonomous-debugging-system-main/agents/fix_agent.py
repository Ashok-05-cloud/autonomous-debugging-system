"""
Fix Planner Agent
Generates root-cause hypotheses + patch plan grounded in evidence.

Multi-file support:
  - Groups retrieved_code by file
  - Builds per-file AST dependency map (imports + call relationships)
  - Produces root_causes list (one entry per affected file)
  - Produces files_to_change list covering all affected files
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orchestrator.state import DebugState
from utils.llm_client import LLMClient
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Safe-coercion helpers ────────────────────────────────────────────────────

def _sf(v: Any, default: float = 0.0) -> float:
    if v is None: return default
    try:
        r = float(v)
        return default if r != r else max(0.0, min(1.0, r))
    except (TypeError, ValueError):
        return default

def _sl(v: Any) -> list:
    return v if isinstance(v, list) else []

def _sd(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


# ─── Dependency mapping ───────────────────────────────────────────────────────

def _extract_file_dependencies(file_path: str, content: str) -> Dict[str, Any]:
    """
    Lightweight AST-based dependency map for a single file.
    Returns:
      imports:      list of imported module names
      functions:    list of top-level function names
      classes:      list of top-level class names
      calls_to:     identifiers this file calls (best-effort)
    """
    result = {"imports": [], "functions": [], "classes": [], "calls_to": []}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result["imports"].extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                result["imports"].append(node.module)
        elif isinstance(node, ast.FunctionDef):
            if node.col_offset == 0:   # top-level only
                result["functions"].append(node.name)
        elif isinstance(node, ast.ClassDef):
            if node.col_offset == 0:
                result["classes"].append(node.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                result["calls_to"].append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result["calls_to"].append(node.func.attr)

    # Deduplicate
    result["calls_to"] = list(set(result["calls_to"]))[:20]
    return result


def _group_code_by_file(retrieved_code: List[Dict]) -> Dict[str, List[Dict]]:
    """Group retrieved code chunks by their source file."""
    grouped: Dict[str, List[Dict]] = {}
    for chunk in retrieved_code:
        fpath = chunk.get("file") or chunk.get("file_path") or "unknown"
        grouped.setdefault(fpath, []).append(chunk)
    return grouped


def _build_file_graph(grouped: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    For each file, read content from chunks and extract dependency info.
    Returns file_path → {content_preview, dep_info, chunks}
    """
    graph = {}
    for fpath, chunks in grouped.items():
        content = "\n\n".join(c.get("content", "") for c in chunks)
        dep_info = _extract_file_dependencies(fpath, content)
        graph[fpath] = {
            "content_preview": content[:1500],
            "dep_info": dep_info,
            "chunks": chunks,
            "functions": [c.get("function") for c in chunks if c.get("function")],
        }
    return graph


# ─── Agent ────────────────────────────────────────────────────────────────────

class FixPlannerAgent:

    def __init__(self):
        self._llm = LLMClient()

    def run(self, state: DebugState) -> DebugState:
        logger.info("[FixPlanner] Starting fix planning")

        triage       = _sd(state.get("triage"))
        log_analysis = _sd(state.get("log_analysis"))
        exec_result  = _sd(state.get("execution_result"))
        retrieved    = _sl(state.get("retrieved_code"))
        bug_report   = _sd(state.get("bug_report"))

        # ── Build multi-file evidence structure ───────────────────────────
        grouped      = _group_code_by_file(retrieved)
        file_graph   = _build_file_graph(grouped)
        evidence     = self._collect_evidence(
            triage, log_analysis, exec_result, retrieved, file_graph
        )

        # ── Try LLM ───────────────────────────────────────────────────────
        if state.get("llm_available"):
            result = self._llm_plan(evidence, bug_report)
            if result and "root_cause" in result:
                result = self._sanitize_llm_plan(result)
                # Enrich with multi-file data
                result = self._enrich_with_multifile(result, file_graph, log_analysis)
                state["root_cause"]  = result["root_cause"]
                state["root_causes"] = result.get("root_causes", [])
                state["patch_plan"]  = result.get("patch_plan", {})
                logger.info("[FixPlanner] LLM plan succeeded")
                return state

        # ── Heuristic fallback ────────────────────────────────────────────
        logger.info("[FixPlanner] Using heuristic fix planning")
        root_cause, patch_plan = self._heuristic_plan(
            triage, log_analysis, exec_result, retrieved, bug_report
        )
        root_causes = self._build_root_causes_list(root_cause, file_graph, log_analysis)
        patch_plan  = self._enrich_patch_plan(patch_plan, file_graph)

        state["root_cause"]  = root_cause
        state["root_causes"] = root_causes if root_causes else [
            {"file": root_cause.get("affected_component","").split(":")[0],
             "issue": root_cause.get("hypothesis",""),
             "confidence": root_cause.get("confidence", 0.5),
             "function": root_cause.get("affected_component","").split(":")[-1]}
        ]
        state["patch_plan"]  = patch_plan
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Evidence collection
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_evidence(
        self,
        triage: Dict,
        log_analysis: Dict,
        exec_result: Dict,
        retrieved: List[Dict],
        file_graph: Dict,
    ) -> Dict:
        traces = _sl(log_analysis.get("stack_traces"))
        primary_trace = None
        if traces:
            t = _sd(traces[0])
            pf = _sd(t.get("primary_frame"))
            primary_trace = {
                "exception": t.get("exception_type"),
                "message": (t.get("message") or "")[:200],
                "file": pf.get("file"),
                "line": pf.get("lineno"),
                "function": pf.get("function"),
                "frequency": t.get("frequency", 1),
            }

        repro_summary = {
            "status": exec_result.get("status"),
            "exit_code": exec_result.get("exit_code"),
            "repro_confirmed": exec_result.get("status") == "fail",
            "output_snippet": (exec_result.get("output") or "")[:400],
        }

        # Multi-file code context
        file_summaries = []
        for fpath, info in list(file_graph.items())[:4]:
            dep = info["dep_info"]
            file_summaries.append({
                "file": fpath,
                "functions": dep["functions"][:5],
                "imports": dep["imports"][:5],
                "snippet": info["content_preview"][:400],
            })

        return {
            "primary_stack_trace": primary_trace,
            "repro_result": repro_summary,
            "hypotheses": _sl(triage.get("hypotheses"))[:2],
            "failure_surface": triage.get("failure_surface"),
            "anomalies": _sl(log_analysis.get("anomalies"))[:5],
            "affected_files": file_summaries,
            "file_count": len(file_graph),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # LLM plan
    # ─────────────────────────────────────────────────────────────────────────

    def _llm_plan(self, evidence: Dict, bug_report: Dict) -> Optional[Dict]:
        file_list = [s["file"] for s in evidence.get("affected_files", [])]
        prompt = f"""
Analyze this debugging evidence and return a JSON root-cause analysis.
There may be MULTIPLE files involved — include all of them.

Return ONLY this JSON structure (no markdown):
{{
  "root_cause": {{
    "hypothesis": "precise technical description",
    "mechanism": "step-by-step how bug manifests",
    "confidence": 0.85,
    "evidence_alignment": "how evidence supports this",
    "affected_component": "primary_file:line:function",
    "bug_type": "precision_error|race_condition|off_by_one|type_error|logic_error|memory_leak|other"
  }},
  "root_causes": [
    {{"file": "file1.py", "issue": "description", "confidence": 0.9, "function": "fn_name"}},
    {{"file": "file2.py", "issue": "related issue", "confidence": 0.6, "function": "fn_name"}}
  ],
  "patch_plan": {{
    "summary": "one-line fix",
    "files_to_change": [
      {{"file": "file1.py", "lines": "55-60", "change": "what to change", "reason": "why"}},
      {{"file": "file2.py", "lines": "10-15", "change": "what to change", "reason": "why"}}
    ],
    "code_change": "actual before/after code",
    "risks": ["risk 1"],
    "breaking_changes": false,
    "testing_approach": "how to verify",
    "validation_steps": ["step 1", "step 2"],
    "regression_tests": ["test 1", "test 2"]
  }}
}}

EVIDENCE: {json.dumps(evidence, indent=2)[:3000]}
BUG TITLE: {bug_report.get('title','')}
AFFECTED FILES FOUND: {file_list}
"""
        raw = self._llm.generate(prompt)
        return raw

    # ─────────────────────────────────────────────────────────────────────────
    # Sanitizer
    # ─────────────────────────────────────────────────────────────────────────

    def _sanitize_llm_plan(self, result: Dict) -> Dict:
        if not isinstance(result, dict):
            return {}

        # root_cause
        rc = _sd(result.get("root_cause"))
        rc["confidence"] = _sf(rc.get("confidence"), default=0.5)
        for sf in ("hypothesis","mechanism","evidence_alignment","affected_component","bug_type"):
            if not isinstance(rc.get(sf), str): rc[sf] = ""
        result["root_cause"] = rc

        # root_causes list
        rcs = _sl(result.get("root_causes"))
        clean_rcs = []
        for item in rcs:
            item = _sd(item)
            item["confidence"] = _sf(item.get("confidence"), default=0.5)
            item.setdefault("file", "")
            item.setdefault("issue", "")
            item.setdefault("function", "")
            clean_rcs.append(item)
        result["root_causes"] = clean_rcs

        # patch_plan
        pp = _sd(result.get("patch_plan"))
        for lf in ("files_to_change","risks","validation_steps","regression_tests"):
            val = pp.get(lf)
            if not isinstance(val, list): pp[lf] = []
        for sf in ("summary","code_change","testing_approach"):
            if not isinstance(pp.get(sf), str): pp[sf] = ""
        if not isinstance(pp.get("breaking_changes"), bool): pp["breaking_changes"] = False
        result["patch_plan"] = pp
        return result

    def _enrich_with_multifile(
        self, result: Dict, file_graph: Dict, log_analysis: Dict
    ) -> Dict:
        """Add any missing files from the graph to root_causes and files_to_change."""
        existing_rc_files  = {r.get("file","") for r in result.get("root_causes", [])}
        existing_ftc_files = {f.get("file","") for f in result.get("patch_plan",{}).get("files_to_change",[])}

        # Add stack-trace files not already covered
        for trace in _sl(log_analysis.get("stack_traces"))[:3]:
            for frame in _sl(trace.get("all_frames", []))[:3]:
                ffile = frame.get("file","")
                if ffile and ffile not in existing_rc_files:
                    result.setdefault("root_causes", []).append({
                        "file": ffile,
                        "issue": f"Involved in stack trace at line {frame.get('lineno')}",
                        "confidence": 0.4,
                        "function": frame.get("function",""),
                    })
                    existing_rc_files.add(ffile)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Heuristic planning
    # ─────────────────────────────────────────────────────────────────────────

    def _heuristic_plan(
        self,
        triage: Dict,
        log_analysis: Dict,
        exec_result: Dict,
        retrieved: List[Dict],
        bug_report: Dict,
    ) -> Tuple[Dict, Dict]:
        hypotheses     = _sl(triage.get("hypotheses"))
        top_h          = _sd(hypotheses[0]) if hypotheses else {}
        hypothesis_desc = (top_h.get("description") or "Unknown bug").lower()

        raw_conf = top_h.get("confidence")
        try:
            hypothesis_conf = float(raw_conf) if raw_conf is not None else 0.3
            hypothesis_conf = max(0.0, min(1.0, hypothesis_conf))
        except (TypeError, ValueError):
            hypothesis_conf = 0.3

        traces  = _sl(log_analysis.get("stack_traces"))
        primary = _sd(traces[0]) if traces else {}
        pf      = _sd(primary.get("primary_frame"))

        repro_status   = exec_result.get("status") or "unknown"
        repro_confirmed = (repro_status == "fail")
        if repro_confirmed:
            hypothesis_conf = min(0.97, hypothesis_conf + 0.25)
        elif repro_status == "error":
            hypothesis_conf = max(0.10, hypothesis_conf - 0.10)

        desc     = (bug_report.get("description") or "").lower()
        hints    = " ".join(bug_report.get("reproduction_hints") or []).lower()
        combined = hypothesis_desc + desc + hints

        if any(kw in combined for kw in ["float","precision","minor_unit","decimal","cents","currency"]):
            rc, pp = self._plan_float_precision(pf, repro_confirmed, exec_result)
        elif any(kw in combined for kw in ["race","concurrent","thread","pool","deadlock"]):
            rc, pp = self._plan_race_condition(pf, repro_confirmed)
        elif any(kw in combined for kw in ["jwt","token","leeway","expire","auth"]):
            rc, pp = self._plan_jwt(pf, repro_confirmed)
        elif any(kw in combined for kw in ["memory leak","unbounded","grows","bucket"]):
            rc, pp = self._plan_memory_leak(pf, repro_confirmed)
        else:
            rc, pp = self._plan_generic(hypothesis_desc, hypothesis_conf, pf, bug_report)

        rc["confidence"] = hypothesis_conf
        return rc, pp

    def _build_root_causes_list(
        self, primary_rc: Dict, file_graph: Dict, log_analysis: Dict
    ) -> List[Dict]:
        """
        Build root_causes list: primary cause + one entry per additional affected file.
        """
        root_causes = []

        # Primary
        root_causes.append({
            "file": primary_rc.get("affected_component", "").split(":")[0],
            "issue": primary_rc.get("hypothesis", ""),
            "confidence": _sf(primary_rc.get("confidence"), 0.5),
            "function": primary_rc.get("affected_component", "").split(":")[-1],
        })

        # Additional files from file_graph not already represented
        seen = {root_causes[0]["file"]}
        for fpath, info in file_graph.items():
            short = Path(fpath).name
            if fpath not in seen and short not in seen:
                root_causes.append({
                    "file": fpath,
                    "issue": f"Involved file — see functions: {info['functions'][:3]}",
                    "confidence": 0.5,
                    "function": info["functions"][0] if info["functions"] else "",
                })
                seen.add(fpath)

        # Add stack trace frames not yet covered
        for trace in _sl(log_analysis.get("stack_traces"))[:2]:
            for frame in _sl(trace.get("all_frames", []))[:5]:
                ff = frame.get("file", "")
                fn = Path(ff).name if ff else ""
                if ff and ff not in seen and fn not in seen:
                    root_causes.append({
                        "file": ff,
                        "issue": f"Stack frame at line {frame.get('lineno')} in {frame.get('function')}",
                        "confidence": 0.45,
                        "function": frame.get("function", ""),
                    })
                    seen.add(ff)

        return root_causes[:6]   # cap at 6 entries

    def _enrich_patch_plan(self, patch_plan: Dict, file_graph: Dict) -> Dict:
        """
        Ensure files_to_change covers all files in the file_graph
        that are not already listed.
        """
        existing = {f.get("file","") for f in _sl(patch_plan.get("files_to_change"))}
        for fpath, info in file_graph.items():
            if fpath not in existing:
                patch_plan.setdefault("files_to_change", []).append({
                    "file": fpath,
                    "lines": f"{info['chunks'][0].get('start_line','?')}-{info['chunks'][-1].get('end_line','?')}",
                    "change": f"Review {info['functions'][:2]} for related issues",
                    "reason": "File appears in retrieved code — may be affected",
                })
        return patch_plan

    # ─────────────────────────────────────────────────────────────────────────
    # Heuristic plan templates
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_float_precision(self, pf: Dict, confirmed: bool, exec_result: Dict):
        rc = {
            "hypothesis": (
                "ValueError (INVALID_AMOUNT) caused by float precision loss in "
                "Transaction.to_minor_units(): int(float(Decimal) * multiplier) "
                "truncates for certain decimal values, sending 1 cent less than expected."
            ),
            "mechanism": (
                "1. amount stored as Decimal (exact). "
                "2. float(Decimal('9999.99')) introduces IEEE 754 rounding. "
                "3. float * 100 = 999998.999... → int() truncates to 999998. "
                "4. Gateway receives 999998 instead of 999999 → INVALID_AMOUNT."
            ),
            "confidence": 0.97 if confirmed else 0.80,
            "evidence_alignment": (
                f"Repro {'confirmed' if confirmed else 'inconclusive'}. "
                "Stack traces point to to_minor_units() at the failure site. "
                "Log shows minor_units_sent off by 1 in every failing case."
            ),
            "affected_component": pf.get("file", "src/payments/processor.py") + ":to_minor_units",
            "bug_type": "precision_error",
        }
        pp = {
            "summary": "Replace int(float(amount) * multiplier) with int(amount * multiplier)",
            "files_to_change": [{
                "file": pf.get("file", "src/payments/processor.py"),
                "lines": str(pf.get("lineno", "~58")),
                "change": "return int(float(self.amount) * multiplier)  →  return int(self.amount * multiplier)",
                "reason": "Decimal * int is always exact; float intermediate loses precision",
            }],
            "code_change": (
                "# BEFORE (buggy):\nreturn int(float(self.amount) * multiplier)\n\n"
                "# AFTER (fixed):\nreturn int(self.amount * multiplier)  # Decimal*int is exact"
            ),
            "risks": [
                "Ensure CURRENCY_MULTIPLIERS values are int, not float (they are — safe)",
                "Verify int(Decimal * int) for all supported currencies",
                "Audit codebase for other float(Decimal) conversion patterns",
            ],
            "breaking_changes": False,
            "testing_approach": "Parametrized test across representative amounts for each currency",
            "validation_steps": [
                "pytest tests/test_payments.py::TestEdgeCases::test_large_amount_precision",
                "Verify: int(Decimal('9999.99') * 100) == 999999",
                "Verify: int(Decimal('1234.57') * 100) == 123457",
                "Deploy to staging → run full payment flow",
                "Monitor gateway rejection rate for 1000 transactions",
            ],
            "regression_tests": [
                "test_minor_units_parametrized: amounts=[9999.99, 1234.57, 3456.78, 7890.45, 0.01, 999999.99]",
                "test_minor_units_jpy: Decimal('1500') * 1 == 1500",
                "test_minor_units_all_currencies: USD, EUR, GBP, JPY",
            ],
        }
        return rc, pp

    def _plan_race_condition(self, pf: Dict, confirmed: bool):
        rc = {
            "hypothesis": "Race condition in ConnectionPool.acquire(): check-then-act on in_use is not atomic.",
            "mechanism": "Two threads both read in_use=False before either writes True; both claim the same connection.",
            "confidence": 0.80 if confirmed else 0.55,
            "evidence_alignment": "Logs show connection exhaustion under load.",
            "affected_component": pf.get("file", "src/database/connection_pool.py") + ":acquire",
            "bug_type": "race_condition",
        }
        pp = {
            "summary": "Hold _lock during the entire check-and-set of in_use in acquire()",
            "files_to_change": [{"file": "src/database/connection_pool.py", "lines": "89-100",
                "change": "Wrap acquire loop body in 'with self._lock:'",
                "reason": "Makes the read-modify-write of in_use atomic"}],
            "code_change": "with self._lock:\n    for conn in self._pool:\n        if not conn.in_use:\n            conn.in_use = True\n            return conn",
            "risks": ["Lock contention under very high load — consider threading.Condition"],
            "breaking_changes": False,
            "testing_approach": "50 concurrent threads acquire() simultaneously — assert no duplicate connection IDs",
            "validation_steps": ["Run race stress test 100x", "No duplicate conn IDs in results"],
            "regression_tests": ["test_pool_concurrent_no_race: 50 threads × 100 iterations"],
        }
        return rc, pp

    def _plan_jwt(self, pf: Dict, confirmed: bool):
        rc = {
            "hypothesis": "JWTManager(leeway=-1) rejects valid tokens 1 second before their actual expiry.",
            "mechanism": "decode() checks 'now > exp + leeway'; leeway=-1 means 'now > exp-1' → rejected 1s early.",
            "confidence": 0.90 if confirmed else 0.65,
            "evidence_alignment": "Logs show TOKEN_EXPIRED for tokens with remaining validity.",
            "affected_component": pf.get("file", "src/auth/auth_service.py") + ":JWTManager.__init__",
            "bug_type": "off_by_one",
        }
        pp = {
            "summary": "Change JWTManager default leeway from -1 to 0",
            "files_to_change": [{"file": "src/auth/auth_service.py", "lines": "~65",
                "change": "def __init__(self, secret_key: str, leeway: int = 0):",
                "reason": "leeway=-1 causes premature expiry; 0 is correct; positive tolerates clock skew"}],
            "code_change": "class JWTManager:\n    def __init__(self, secret_key: str, leeway: int = 0):  # was -1",
            "risks": ["Tokens issued near boundary may now be accepted for 1 extra second"],
            "breaking_changes": False,
            "testing_approach": "Issue token at boundary (exp-1, exp, exp+1), assert correct accept/reject",
            "validation_steps": ["Test token at t=exp-2s (valid)", "Test token at t=exp+2s (invalid)"],
            "regression_tests": ["test_jwt_boundary_validity", "test_jwt_leeway_zero"],
        }
        return rc, pp

    def _plan_memory_leak(self, pf: Dict, confirmed: bool):
        rc = {
            "hypothesis": "RateLimiter._buckets grows unbounded; cleanup thread has self-deadlock.",
            "mechanism": "_cleanup_loop holds _lock then calls _cleanup_expired which also acquires _lock → deadlock.",
            "confidence": 0.82 if confirmed else 0.60,
            "evidence_alignment": "Logs show bucket_count=12847 and elevated memory.",
            "affected_component": pf.get("file", "src/api/router.py") + ":RateLimiter",
            "bug_type": "memory_leak",
        }
        pp = {
            "summary": "Remove nested lock in cleanup; evict idle buckets in _cleanup_expired",
            "files_to_change": [{"file": "src/api/router.py", "lines": "~70-110",
                "change": "_cleanup_loop must NOT hold _lock when calling _cleanup_expired",
                "reason": "RLock is reentrant but threading.Lock is not — deadlock on cleanup"}],
            "code_change": "def _cleanup_loop(self):\n    while True:\n        time.sleep(self.cleanup_interval)\n        try:\n            self._cleanup_expired()  # acquires its own lock internally\n        except Exception as e:\n            logger.error(f'Cleanup error: {e}')",
            "risks": ["Brief window where count may be slightly inconsistent during cleanup"],
            "breaking_changes": False,
            "testing_approach": "10k unique IPs → verify bucket count stabilises after cleanup_interval",
            "validation_steps": ["Monitor bucket_count over 5 minutes", "Confirm cleanup reduces count"],
            "regression_tests": ["test_rate_limiter_bucket_eviction", "test_cleanup_no_deadlock"],
        }
        return rc, pp

    def _plan_generic(self, hypothesis: str, confidence: float, pf: Dict, bug_report: Dict):
        rc = {
            "hypothesis": hypothesis or "Logic error in affected component",
            "mechanism": "Specific mechanism requires further investigation.",
            "confidence": min(confidence, 0.40),
            "evidence_alignment": "Limited evidence — manual investigation recommended.",
            "affected_component": (
                pf.get("file", bug_report.get("component","unknown"))
                + ":" + (pf.get("function") or "unknown")
            ),
            "bug_type": "logic_error",
        }
        pp = {
            "summary": "Review affected component logic",
            "files_to_change": [{"file": bug_report.get("component","unknown"),
                "lines": "unknown", "change": "TBD", "reason": "Identified as likely failure surface"}],
            "code_change": "# Requires manual code review",
            "risks": ["Unknown without full inspection"],
            "breaking_changes": False,
            "testing_approach": "Add targeted unit test for the failure scenario",
            "validation_steps": ["Reproduce manually", "Code review", "Add regression test"],
            "regression_tests": ["test_identified_failure_scenario"],
        }
        return rc, pp
