"""
Sanity Checks
Post-pipeline validation layer.
"""
from __future__ import annotations

from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)


class SanityChecker:

    def run_all(self, state: Dict) -> Dict[str, Any]:
        results = {
            "repro_failed": self.verify_repro_failed(state),
            "log_alignment": self.verify_log_alignment(state),
            "patch_consistency": self.verify_patch_consistency(state),
            "output_completeness": self.verify_output_completeness(state),
        }
        passed = sum(1 for v in results.values() if v.get("passed"))
        results["summary"] = {
            "passed": passed,
            "total": len(results) - 1,
            "all_passed": passed == len(results) - 1,
        }
        return results

    @staticmethod
    def verify_repro_failed(state: Dict) -> Dict:
        exec_result = state.get("execution_result", {})
        status = exec_result.get("status", "unknown")
        passed = status in ("fail", "crash")  # repro should demonstrate failure
        return {
            "passed": passed,
            "check": "repro_failed",
            "detail": f"Repro status={status}. Expected 'fail' or 'crash'.",
            "actual": status,
        }

    @staticmethod
    def verify_log_alignment(state: Dict) -> Dict:
        log_analysis = state.get("log_analysis", {})
        root_cause = state.get("root_cause", {})

        traces = log_analysis.get("stack_traces", [])
        hypothesis = root_cause.get("hypothesis", "").lower()
        affected = root_cause.get("affected_component", "").lower()

        if not traces:
            return {"passed": False, "check": "log_alignment", "detail": "No stack traces to align against", "aligned": False}

        primary = traces[0]
        pf = primary.get("primary_frame") or {}
        frame_file = (pf.get("file") or "").lower()

        # Check if root cause references the same file as the trace
        aligned = any([
            frame_file and frame_file.split("/")[-1].replace(".py", "") in hypothesis,
            frame_file and frame_file.split("/")[-1].replace(".py", "") in affected,
            primary.get("exception_type", "").lower() in hypothesis,
        ])

        return {
            "passed": aligned,
            "check": "log_alignment",
            "detail": f"Primary trace file '{frame_file}' alignment with root cause: {aligned}",
            "aligned": aligned,
        }

    @staticmethod
    def verify_patch_consistency(state: Dict) -> Dict:
        patch_plan = state.get("patch_plan", {})
        root_cause = state.get("root_cause", {})

        if not patch_plan or not root_cause:
            return {"passed": False, "check": "patch_consistency", "detail": "Missing patch_plan or root_cause"}

        affected = root_cause.get("affected_component", "")
        files_to_change = patch_plan.get("files_to_change", [])
        has_code_change = bool(patch_plan.get("code_change", "").strip())
        has_validation = bool(patch_plan.get("validation_steps"))

        consistent = has_code_change and has_validation

        return {
            "passed": consistent,
            "check": "patch_consistency",
            "detail": f"has_code_change={has_code_change}, has_validation={has_validation}, files={len(files_to_change)}",
            "consistent": consistent,
        }

    @staticmethod
    def verify_output_completeness(state: Dict) -> Dict:
        required_fields = [
            "bug_report", "triage", "log_analysis",
            "repro_code", "execution_result",
            "root_cause", "patch_plan", "review",
        ]
        missing = [f for f in required_fields if not state.get(f)]
        return {
            "passed": len(missing) == 0,
            "check": "output_completeness",
            "detail": f"Missing fields: {missing}" if missing else "All required fields present",
            "missing_fields": missing,
        }
