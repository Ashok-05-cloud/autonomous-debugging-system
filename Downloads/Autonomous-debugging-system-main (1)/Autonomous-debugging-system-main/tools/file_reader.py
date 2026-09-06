"""
File Reader Tool
Safe file reading with encoding fallbacks, size limits, and path normalization.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

MAX_FILE_SIZE = 512 * 1024  # 512KB — avoid reading huge binaries
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".zip", ".tar", ".gz",
    ".whl", ".egg",
}
TEXT_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]


def read_file(
    path: str,
    max_bytes: int = MAX_FILE_SIZE,
    encoding: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Read a file safely.

    Returns: (content, error_message)
      - (content, None) on success
      - (None, error_msg) on failure
    """
    p = Path(path)

    # Existence check
    if not p.exists():
        return None, f"File not found: {path}"

    # Binary extension guard
    if p.suffix.lower() in BINARY_EXTENSIONS:
        return None, f"Binary file type skipped: {p.suffix}"

    # Size check
    try:
        size = p.stat().st_size
    except OSError as e:
        return None, f"Cannot stat file: {e}"

    if size > max_bytes:
        logger.warning(f"[FileReader] File too large ({size} bytes): {path} — reading first {max_bytes}")

    # Encoding trial
    encodings_to_try = [encoding] if encoding else TEXT_ENCODINGS

    for enc in encodings_to_try:
        try:
            with open(p, "r", encoding=enc, errors="replace") as f:
                content = f.read(max_bytes)
            if size > max_bytes:
                content += f"\n\n[... TRUNCATED: {size - max_bytes} bytes remaining ...]"
            return content, None
        except UnicodeDecodeError:
            continue
        except PermissionError as e:
            return None, f"Permission denied: {e}"
        except OSError as e:
            return None, f"OS error reading file: {e}"

    # All encodings failed
    return None, f"Cannot decode file with any of: {encodings_to_try}"


def find_files(
    root: str,
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    max_files: int = 500,
) -> List[str]:
    """
    Recursively find files under root.
    Returns list of relative path strings.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        logger.warning(f"[FileReader] Not a directory: {root}")
        return []

    exclude = set(exclude_dirs or ["__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build", ".tox"])
    exts = set(extensions or [".py", ".js", ".ts", ".java", ".go", ".rb", ".rs", ".cpp", ".c", ".h"])

    found = []
    try:
        for path in root_path.rglob("*"):
            if len(found) >= max_files:
                logger.warning(f"[FileReader] Max file count {max_files} reached")
                break
            # Skip excluded dirs
            if any(excl in path.parts for excl in exclude):
                continue
            if path.is_file() and path.suffix.lower() in exts:
                found.append(str(path))
    except PermissionError:
        pass

    return found


def read_file_lines(
    path: str,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Read specific line range from a file."""
    content, err = read_file(path)
    if err:
        return None, err

    lines = content.splitlines()
    total = len(lines)

    start = max(0, start_line - 1)
    end = min(total, end_line) if end_line else total

    return lines[start:end], None
