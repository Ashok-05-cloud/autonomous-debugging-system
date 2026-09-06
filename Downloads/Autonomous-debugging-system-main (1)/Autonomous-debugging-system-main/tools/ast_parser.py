"""
AST Parser Tool
Extracts function/class context from Python source files using AST.
Handles: missing files, syntax errors, dynamic code, nested functions.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class FunctionExtractor(ast.NodeVisitor):
    """Visits AST nodes to find function/method containing a target line."""

    def __init__(self, target_line: int):
        self.target_line = target_line
        self.candidates: List[ast.FunctionDef] = []
        self._class_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._check_function(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_function(self, node):
        start = node.lineno
        end = getattr(node, "end_lineno", None) or self._estimate_end(node)
        if start <= self.target_line <= end:
            # Attach class context
            node._class_context = ".".join(self._class_stack) if self._class_stack else None
            self.candidates.append(node)

    @staticmethod
    def _estimate_end(node) -> int:
        """Estimate end line for older Python without end_lineno."""
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                max_line = max(max_line, child.lineno)
        return max_line


def extract_function(
    file_path: str,
    line_number: int,
    context_lines: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Extract the function/method that contains `line_number` in `file_path`.

    Returns dict with:
      - function_name, class_name, start_line, end_line
      - source_code (the full function)
      - context_before/after (surrounding lines)
    Returns None if file not found or function cannot be identified.
    """
    path = Path(file_path)

    # ── File existence check ──────────────────────────────────────────────
    if not path.exists():
        # Try with common prefix stripping (container paths vs local)
        for strip_prefix in ("/opt/app/", "/app/", "/home/app/"):
            alt = Path(str(path).replace(strip_prefix, ""))
            if alt.exists():
                path = alt
                break
        else:
            logger.warning(f"[ASTParser] File not found: {file_path}")
            return None

    # ── Read file ─────────────────────────────────────────────────────────
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            source = path.read_text(encoding="latin-1")
            logger.warning(f"[ASTParser] Used latin-1 fallback for {file_path}")
        except Exception as e:
            logger.error(f"[ASTParser] Cannot read {file_path}: {e}")
            return None
    except Exception as e:
        logger.error(f"[ASTParser] File read error {file_path}: {e}")
        return None

    source_lines = source.splitlines()

    # ── Parse AST ─────────────────────────────────────────────────────────
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        logger.warning(f"[ASTParser] Syntax error in {file_path}: {e} — falling back to raw read")
        return _fallback_raw_extract(source_lines, line_number, file_path, context_lines)
    except Exception as e:
        logger.error(f"[ASTParser] AST parse error {file_path}: {e}")
        return _fallback_raw_extract(source_lines, line_number, file_path, context_lines)

    # ── Find enclosing function ───────────────────────────────────────────
    extractor = FunctionExtractor(line_number)
    extractor.visit(tree)

    if not extractor.candidates:
        logger.debug(f"[ASTParser] No function found at line {line_number} in {file_path}")
        # Return surrounding context instead
        return _fallback_raw_extract(source_lines, line_number, file_path, context_lines)

    # Pick innermost (last/most-nested) candidate
    node = extractor.candidates[-1]
    start = node.lineno - 1  # 0-indexed
    end = getattr(node, "end_lineno", node.lineno + 50)

    func_lines = source_lines[start:end]
    func_source = "\n".join(func_lines)

    # Dedent
    try:
        func_source = textwrap.dedent(func_source)
    except Exception:
        pass

    # Context lines around target
    target_idx = line_number - 1
    ctx_start = max(0, target_idx - context_lines)
    ctx_end = min(len(source_lines), target_idx + context_lines + 1)
    context_snippet = "\n".join(source_lines[ctx_start:ctx_end])

    # Check for dynamic/generated code markers
    is_dynamic = any(
        marker in func_source
        for marker in ("exec(", "eval(", "compile(", "__import__", "importlib")
    )

    result = {
        "function_name": node.name,
        "class_name": getattr(node, "_class_context", None),
        "file": str(path),
        "start_line": node.lineno,
        "end_line": end,
        "target_line": line_number,
        "source_code": func_source,
        "context_snippet": context_snippet,
        "is_dynamic": is_dynamic,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "args": [arg.arg for arg in node.args.args],
        "decorators": [
            ast.unparse(d) if hasattr(ast, "unparse") else str(d)
            for d in node.decorator_list
        ],
    }

    logger.debug(
        f"[ASTParser] Extracted {node.name}() "
        f"(lines {node.lineno}–{end}) from {file_path}"
    )
    return result


def _fallback_raw_extract(
    source_lines: List[str],
    line_number: int,
    file_path: str,
    context_lines: int = 10,
) -> Dict[str, Any]:
    """Return raw line context when AST extraction fails."""
    idx = line_number - 1
    start = max(0, idx - context_lines)
    end = min(len(source_lines), idx + context_lines + 1)
    snippet = "\n".join(source_lines[start:end])

    return {
        "function_name": None,
        "class_name": None,
        "file": file_path,
        "start_line": start + 1,
        "end_line": end,
        "target_line": line_number,
        "source_code": snippet,
        "context_snippet": snippet,
        "is_dynamic": False,
        "is_async": False,
        "args": [],
        "decorators": [],
        "_fallback": True,
        "_fallback_reason": "AST parse failed or function not found",
    }


def extract_class_methods(file_path: str, class_name: str) -> List[Dict[str, Any]]:
    """
    Extract all method signatures from a class.
    Useful for understanding class-level context.
    """
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception as e:
        logger.warning(f"[ASTParser] Cannot parse {file_path} for class extraction: {e}")
        return []

    methods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append({
                        "name": item.name,
                        "line": item.lineno,
                        "args": [arg.arg for arg in item.args.args],
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                    })
            break

    return methods


def get_imports(file_path: str) -> List[str]:
    """Extract all import statements from a Python file."""
    path = Path(file_path)
    if not path.exists():
        return []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except Exception:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(a.name for a in node.names)
            imports.append(f"from {module} import {names}")
    return imports
