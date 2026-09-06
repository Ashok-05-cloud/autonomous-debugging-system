"""
Execution Tool
Safely runs Python test scripts/pytest with timeout, resource limits, crash capture.

Classification rules (deterministic, never ambiguous):
  AssertionError in output → "fail"   (correct repro — bug demonstrated)
  exit_code == 0           → "pass"   (bug NOT reproduced)
  SyntaxError / ImportError → "error" (script broken, not a bug signal)
  Timeout                  → "timeout"
  Segfault                 → "crash"
  Anything else non-zero   → "fail"   (some exception = bug signal)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = int(os.getenv("EXEC_TIMEOUT", "30"))
MAX_OUTPUT_BYTES = 50 * 1024  # 50KB output cap


class ExecutionResult:
    def __init__(
        self,
        status: str,
        output: str,
        exit_code: int,
        duration_s: float,
        command: str,
        stderr: str = "",
    ):
        self.status = status
        self.output = output
        self.exit_code = exit_code
        self.duration_s = duration_s
        self.command = command
        self.stderr = stderr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output[:5000],
            "stderr": self.stderr[:2000],
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "command": self.command,
        }


def classify_execution(output: str, stderr: str, exit_code: int) -> str:
    """
    Single source of truth for mapping subprocess output → status string.

    Priority order (highest wins):
      1. exit_code == 0                   → "pass"   (script ran, no exception)
      2. AssertionError in output/stderr  → "fail"   (repro confirmed the bug)
      3. SyntaxError in output/stderr     → "error"  (script has syntax problem)
      4. ImportError/ModuleNotFoundError  → "error"  (missing dependency)
      5. Segfault (exit -11 / 139)        → "crash"
      6. Any other non-zero exit          → "fail"   (exception = bug signal)
    """
    combined = (output + "\n" + stderr).lower()

    if exit_code == 0:
        return "pass"

    # Segfault takes precedence over text checks
    if exit_code in (-11, 139):
        return "crash"

    # AssertionError → correct repro, bug demonstrated
    if "assertionerror" in combined:
        return "fail"

    # Broken script (not a bug signal — fix the script)
    if "syntaxerror" in combined:
        return "error"
    if "modulenotfounderror" in combined or "importerror" in combined:
        return "error"

    # Any other non-zero exit with exception output → fail (bug signal)
    if any(kw in combined for kw in ("exception", "error", "traceback", "raise")):
        return "fail"

    # Non-zero exit with no recognisable exception
    return "fail"


def run_test(
    file_path: str,
    timeout: int = DEFAULT_TIMEOUT,
    use_pytest: bool = True,
    extra_args: Optional[List[str]] = None,
    working_dir: Optional[str] = None,
    env_vars: Optional[Dict[str, str]] = None,
) -> ExecutionResult:
    """Run a Python test file or script safely."""
    path = Path(file_path)

    if not path.exists():
        return ExecutionResult(
            status="error",
            output=f"File not found: {file_path}",
            exit_code=-1, duration_s=0.0, command=file_path,
        )

    if path.suffix != ".py":
        return ExecutionResult(
            status="error",
            output=f"Expected .py file, got: {path.suffix}",
            exit_code=-1, duration_s=0.0, command=file_path,
        )

    if use_pytest and _pytest_available():
        cmd = [sys.executable, "-m", "pytest", str(path), "-v", "--tb=short", "--no-header"]
    else:
        cmd = [sys.executable, str(path)]

    if extra_args:
        cmd.extend(extra_args)

    command_str = " ".join(cmd)
    cwd = working_dir or str(path.parent)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    repo_root = _find_repo_root(path)
    if repo_root:
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else repo_root
    if env_vars:
        env.update(env_vars)

    logger.info(f"[Execution] Running: {command_str} (timeout={timeout}s, cwd={cwd})")
    start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            _kill_process(proc)
            logger.warning(f"[Execution] Process timed out after {timeout}s")
            return ExecutionResult(
                status="timeout",
                output=f"[TIMEOUT] Process killed after {timeout}s — possible infinite loop.",
                exit_code=-9, duration_s=duration, command=command_str,
            )

        duration = time.time() - start
        stdout = _decode_output(stdout_bytes)
        stderr = _decode_output(stderr_bytes)

        # ── Use the single classification function ────────────────────────
        status = classify_execution(stdout, stderr, proc.returncode)

        logger.info(
            f"[Execution] Completed in {duration:.2f}s — "
            f"exit_code={proc.returncode} status={status}"
        )
        return ExecutionResult(
            status=status, output=stdout, stderr=stderr,
            exit_code=proc.returncode, duration_s=duration, command=command_str,
        )

    except FileNotFoundError as e:
        return ExecutionResult(
            status="error", output=f"Interpreter not found: {e}",
            exit_code=-1, duration_s=time.time()-start, command=command_str,
        )
    except PermissionError as e:
        return ExecutionResult(
            status="error", output=f"Permission denied: {e}",
            exit_code=-1, duration_s=time.time()-start, command=command_str,
        )
    except Exception as e:
        logger.error(f"[Execution] Unexpected error: {e}")
        return ExecutionResult(
            status="error", output=f"Framework error: {type(e).__name__}: {e}",
            exit_code=-1, duration_s=time.time()-start, command=command_str,
        )


def run_inline_code(
    code: str,
    timeout: int = DEFAULT_TIMEOUT,
    working_dir: Optional[str] = None,
) -> ExecutionResult:
    """Write code to a temp file and execute it."""
    tmp_dir = Path(working_dir or tempfile.gettempdir())
    tmp_file = tmp_dir / f"repro_inline_{int(time.time())}.py"
    try:
        tmp_file.write_text(code, encoding="utf-8")
        return run_test(str(tmp_file), timeout=timeout, use_pytest=False, working_dir=working_dir)
    finally:
        try:
            tmp_file.unlink(missing_ok=True)
        except Exception:
            pass


def _kill_process(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
        proc.wait(timeout=3)
    except Exception:
        pass


def _decode_output(data: bytes) -> str:
    if not data:
        return ""
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_BYTES:
        text = text[:MAX_OUTPUT_BYTES] + f"\n[... truncated at {MAX_OUTPUT_BYTES} chars ...]"
    return text


def _pytest_available() -> bool:
    try:
        import pytest  # noqa
        return True
    except ImportError:
        return False


def _find_repo_root(script_path: Path) -> Optional[str]:
    for parent in script_path.parents:
        if any((parent / ind).exists() for ind in
               ["setup.py", "pyproject.toml", "setup.cfg", "requirements.txt", ".git"]):
            return str(parent)
    return str(script_path.parent)
