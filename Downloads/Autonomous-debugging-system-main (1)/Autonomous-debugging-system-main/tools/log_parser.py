"""
Log Parser Tool
Extracts stack traces, error signatures, frequencies, and anomalies from raw logs.
Handles: missing logs, corrupted entries, mixed formats, multiple stack traces.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Regex patterns ──────────────────────────────────────────────────────────

# Standard Python traceback
_TRACEBACK_START = re.compile(r"Traceback \(most recent call last\):", re.MULTILINE)
_TRACEBACK_FILE_LINE = re.compile(
    r'^\s+File "(?P<file>[^"]+)", line (?P<lineno>\d+), in (?P<func>\S+)', re.MULTILINE
)
_EXCEPTION_LINE = re.compile(
    r'^(?P<exc_type>[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*Error|'
    r'[A-Za-z][A-Za-z0-9_]*Exception|[A-Za-z][A-Za-z0-9_]*Warning|'
    r'KeyError|ValueError|TypeError|RuntimeError|AttributeError|ImportError|'
    r'OSError|IOError|IndexError|ZeroDivisionError|OverflowError|'
    r'StopIteration|NotImplementedError|AssertionError|PermissionError|'
    r'TimeoutError|ConnectionError|FileNotFoundError):\s*(?P<message>.*)$',
    re.MULTILINE,
)

# Generic error/warning patterns
_ERROR_PATTERN = re.compile(
    r'(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[,\.]?\d*)\s+'
    r'(?P<level>ERROR|CRITICAL|FATAL|WARNING|WARN)\s+'
    r'\[(?P<logger>[^\]]+)\]\s+(?P<message>.+)',
    re.IGNORECASE,
)

# Java-style stack trace
_JAVA_EXCEPTION = re.compile(
    r'^(?P<exc_class>[a-zA-Z][a-zA-Z0-9_.]+Exception|[a-zA-Z][a-zA-Z0-9_.]+Error):\s*(?P<message>.*)',
    re.MULTILINE,
)
_JAVA_FRAME = re.compile(
    r'^\s+at (?P<class>[a-zA-Z0-9_.]+)\.(?P<method>[a-zA-Z0-9_<>$]+)'
    r'\((?P<file>[^:)]+):?(?P<line>\d+)?\)',
    re.MULTILINE,
)

# Noise patterns to filter
_NOISE_PATTERNS = [
    re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO.*healthcheck.*', re.IGNORECASE),
    re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO.*nginx.*HTTP.*200\s', re.IGNORECASE),
    re.compile(r'^\d{4}-\d{2}-\d{2}.*INFO.*pool (stats|health)', re.IGNORECASE),
]


@dataclass
class StackFrame:
    file: str
    lineno: int
    function: str
    raw: str = ""


@dataclass
class ParsedStackTrace:
    exception_type: str
    message: str
    frames: List[StackFrame]
    raw: str
    frequency: int = 1
    source_language: str = "python"  # python | java | node | unknown

    @property
    def primary_frame(self) -> Optional[StackFrame]:
        """The most likely culprit frame (innermost application frame)."""
        # Filter out stdlib/vendor frames
        app_frames = [
            f for f in self.frames
            if not any(
                skip in f.file
                for skip in ("/lib/python", "site-packages", "<frozen", "runpy")
            )
        ]
        return app_frames[-1] if app_frames else (self.frames[-1] if self.frames else None)


@dataclass
class LogAnalysisResult:
    stack_traces: List[ParsedStackTrace] = field(default_factory=list)
    error_lines: List[Dict] = field(default_factory=list)
    top_errors: List[Dict] = field(default_factory=list)  # by frequency
    anomalies: List[str] = field(default_factory=list)
    noise_filtered_count: int = 0
    total_lines: int = 0
    error_count: int = 0
    warning_count: int = 0
    parse_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "stack_traces": [
                {
                    "exception_type": st.exception_type,
                    "message": st.message,
                    "primary_frame": {
                        "file": st.primary_frame.file,
                        "lineno": st.primary_frame.lineno,
                        "function": st.primary_frame.function,
                    } if st.primary_frame else None,
                    "all_frames": [
                        {"file": f.file, "lineno": f.lineno, "function": f.function}
                        for f in st.frames
                    ],
                    "frequency": st.frequency,
                    "source_language": st.source_language,
                }
                for st in self.stack_traces
            ],
            "error_lines": self.error_lines[:20],  # cap
            "top_errors": self.top_errors,
            "anomalies": self.anomalies,
            "stats": {
                "total_lines": self.total_lines,
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "noise_filtered": self.noise_filtered_count,
                "unique_exceptions": len(self.stack_traces),
            },
            "parse_warnings": self.parse_warnings,
        }


class LogParser:
    """
    Robust log parser with multi-format support.
    Handles empty logs, corrupted content, encoding issues.
    """

    def parse(self, raw_logs: str) -> LogAnalysisResult:
        result = LogAnalysisResult()

        # ── Guard: empty / None logs ──────────────────────────────────────
        if not raw_logs or not raw_logs.strip():
            result.parse_warnings.append("No log content provided — analysis skipped")
            logger.warning("[LogParser] Empty logs received")
            return result

        # ── Sanitize encoding ─────────────────────────────────────────────
        try:
            if isinstance(raw_logs, bytes):
                raw_logs = raw_logs.decode("utf-8", errors="replace")
        except Exception as e:
            result.parse_warnings.append(f"Encoding issue: {e}")

        lines = raw_logs.splitlines()
        result.total_lines = len(lines)
        logger.info(f"[LogParser] Parsing {result.total_lines} log lines")

        # ── Filter noise ──────────────────────────────────────────────────
        clean_lines = []
        for line in lines:
            if any(p.match(line) for p in _NOISE_PATTERNS):
                result.noise_filtered_count += 1
            else:
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)

        # ── Extract structured error lines ────────────────────────────────
        for m in _ERROR_PATTERN.finditer(raw_logs):
            level = m.group("level").upper()
            if level in ("ERROR", "CRITICAL", "FATAL"):
                result.error_count += 1
                result.error_lines.append({
                    "timestamp": m.group("timestamp"),
                    "level": level,
                    "logger": m.group("logger"),
                    "message": m.group("message").strip(),
                })
            elif level in ("WARNING", "WARN"):
                result.warning_count += 1

        # ── Extract Python stack traces ───────────────────────────────────
        python_traces = self._extract_python_traces(clean_text)
        result.stack_traces.extend(python_traces)

        # ── Extract Java stack traces ─────────────────────────────────────
        java_traces = self._extract_java_traces(clean_text)
        result.stack_traces.extend(java_traces)

        # ── Fallback: generic error extraction if no structured traces ─────
        if not result.stack_traces and not result.error_lines:
            result.parse_warnings.append(
                "No structured stack traces found — attempting generic extraction"
            )
            result.stack_traces = self._generic_error_extraction(clean_text)

        # ── Deduplicate and rank by frequency ─────────────────────────────
        result.stack_traces = self._deduplicate_traces(result.stack_traces)

        # ── Top errors by type ────────────────────────────────────────────
        exc_counter: Counter = Counter()
        for st in result.stack_traces:
            exc_counter[st.exception_type] += st.frequency

        result.top_errors = [
            {"exception_type": exc, "count": count}
            for exc, count in exc_counter.most_common(10)
        ]

        # ── Anomaly detection ─────────────────────────────────────────────
        result.anomalies = self._detect_anomalies(raw_logs, result)

        logger.info(
            f"[LogParser] Found {len(result.stack_traces)} unique traces, "
            f"{result.error_count} errors, {result.warning_count} warnings"
        )
        return result

    def _extract_python_traces(self, text: str) -> List[ParsedStackTrace]:
        """Extract all Python traceback blocks from log text."""
        traces = []

        # Split on traceback starts
        parts = _TRACEBACK_START.split(text)

        for part in parts[1:]:  # Skip text before first traceback
            # Find end of traceback (exception line)
            exc_match = _EXCEPTION_LINE.search(part)
            if not exc_match:
                # Partial/corrupted traceback — extract what we can
                frames = self._extract_frames_from_partial(part)
                if frames:
                    traces.append(ParsedStackTrace(
                        exception_type="UnknownException",
                        message="Corrupted traceback — exception line missing",
                        frames=frames,
                        raw=part[:500],
                    ))
                continue

            exc_type = exc_match.group("exc_type")
            exc_message = exc_match.group("message").strip()

            # Extract frames
            frames = []
            for fm in _TRACEBACK_FILE_LINE.finditer(part[: exc_match.start()]):
                frames.append(StackFrame(
                    file=fm.group("file"),
                    lineno=int(fm.group("lineno")),
                    function=fm.group("func"),
                    raw=fm.group(0),
                ))

            raw_block = "Traceback (most recent call last):\n" + part[: exc_match.end()]

            traces.append(ParsedStackTrace(
                exception_type=exc_type,
                message=exc_message,
                frames=frames,
                raw=raw_block[:1000],
                source_language="python",
            ))

        return traces

    def _extract_java_traces(self, text: str) -> List[ParsedStackTrace]:
        """Extract Java-style stack traces."""
        traces = []
        for exc_m in _JAVA_EXCEPTION.finditer(text):
            frames = []
            # Scan ahead for `at` lines
            search_start = exc_m.end()
            search_end = min(search_start + 2000, len(text))
            snippet = text[search_start:search_end]
            for fm in _JAVA_FRAME.finditer(snippet):
                frames.append(StackFrame(
                    file=fm.group("file"),
                    lineno=int(fm.group("line")) if fm.group("line") else 0,
                    function=f"{fm.group('class')}.{fm.group('method')}",
                ))
                if len(frames) > 20:
                    break

            if frames:
                traces.append(ParsedStackTrace(
                    exception_type=exc_m.group("exc_class").split(".")[-1],
                    message=exc_m.group("message"),
                    frames=frames,
                    raw=exc_m.group(0)[:500],
                    source_language="java",
                ))

        return traces

    def _extract_frames_from_partial(self, text: str) -> List[StackFrame]:
        """Best-effort frame extraction for corrupted tracebacks."""
        frames = []
        for m in _TRACEBACK_FILE_LINE.finditer(text):
            frames.append(StackFrame(
                file=m.group("file"),
                lineno=int(m.group("lineno")),
                function=m.group("func"),
            ))
        return frames

    def _generic_error_extraction(self, text: str) -> List[ParsedStackTrace]:
        """Last-resort: extract any ERROR-level line as a synthetic trace."""
        traces = []
        for m in _ERROR_PATTERN.finditer(text):
            level = m.group("level").upper()
            if level in ("ERROR", "CRITICAL"):
                traces.append(ParsedStackTrace(
                    exception_type="GenericError",
                    message=m.group("message").strip(),
                    frames=[],
                    raw=m.group(0),
                ))
        return traces

    def _deduplicate_traces(self, traces: List[ParsedStackTrace]) -> List[ParsedStackTrace]:
        """
        Merge identical traces, counting frequency.
        Identity: (exception_type, message[:100], primary_frame_file+lineno)
        """
        seen: Dict[str, ParsedStackTrace] = {}
        for trace in traces:
            pf = trace.primary_frame
            key = (
                trace.exception_type,
                trace.message[:80],
                f"{pf.file}:{pf.lineno}" if pf else "",
            )
            key_str = "|".join(str(k) for k in key)
            if key_str in seen:
                seen[key_str].frequency += 1
            else:
                seen[key_str] = trace

        # Sort by frequency descending
        return sorted(seen.values(), key=lambda t: t.frequency, reverse=True)

    def _detect_anomalies(self, text: str, result: LogAnalysisResult) -> List[str]:
        """Detect patterns suggesting deeper systemic issues."""
        anomalies = []

        # High error rate
        if result.total_lines > 0:
            err_rate = result.error_count / max(result.total_lines, 1)
            if err_rate > 0.1:
                anomalies.append(
                    f"High error rate: {err_rate:.1%} of log lines are errors"
                )

        # Repeated same exception
        for err in result.top_errors:
            if err["count"] >= 5:
                anomalies.append(
                    f"Repeated exception: {err['exception_type']} appears {err['count']} times"
                )

        # Memory/resource keywords
        for keyword, desc in [
            ("memory leak", "Potential memory leak mentioned in logs"),
            ("out of memory", "OOM condition detected"),
            ("deadlock", "Potential deadlock mentioned"),
            ("connection refused", "Connection refused — possible service down"),
            ("timeout", "Timeout errors detected"),
            ("bucket count", "Rate limiter bucket growth detected (possible memory leak)"),
        ]:
            if keyword.lower() in text.lower():
                anomalies.append(desc)

        return anomalies


def extract_stack_trace(logs: str) -> Dict[str, Any]:
    """
    Public API: extract stack trace info from log string.
    Returns dict suitable for state['log_analysis'].
    """
    parser = LogParser()
    result = parser.parse(logs)
    return result.to_dict()
