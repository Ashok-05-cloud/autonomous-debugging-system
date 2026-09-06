"""
LLM Client - Ollama interface with full fallback support.

Uses Ollama REST API with LLaMA 3.2.
If Ollama is unavailable, falls back to heuristic analysis.
All outputs are structured JSON with strict temperature=0.1 for determinism.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))


class LLMClient:
    """
    Thin wrapper around Ollama REST API.
    Enforces structured JSON output.
    Handles connection errors, timeouts, malformed responses.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Ollama server is reachable and model is loaded."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            if resp.status_code == 200:
                tags = resp.json().get("models", [])
                model_names = [t.get("name", "").split(":")[0] for t in tags]
                self._available = any(
                    self.model.split(":")[0] in n for n in model_names
                )
                if not self._available:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {model_names}. Will use fallbacks."
                    )
            else:
                self._available = False
        except Exception as e:
            logger.warning(f"Ollama not reachable at {self.base_url}: {e}")
            self._available = False

        return self._available

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """
        Send prompt to Ollama, return parsed JSON response dict.
        Returns None on complete failure (caller handles fallback).

        Enforces JSON output via system prompt instruction.
        """
        if not self.is_available():
            logger.debug("LLM not available, skipping generate()")
            return None

        full_system = (
            "You are a senior software engineer and debugging expert. "
            "You MUST respond with valid JSON only. "
            "No markdown, no explanation outside the JSON structure. "
            "If uncertain about any field, set it to null. "
            "Never invent code paths or filenames not mentioned in the input. "
        )
        if system_prompt:
            full_system += f"\n\nAdditional context: {system_prompt}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": full_system,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
            "format": "json",  # Ollama structured output
        }

        for attempt in range(1, retries + 2):
            try:
                logger.debug(f"LLM request attempt {attempt}/{retries + 1}")
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "")

                # Attempt JSON parse
                parsed = self._safe_json_parse(raw)
                if parsed is not None:
                    return parsed

                logger.warning(f"LLM returned non-JSON on attempt {attempt}: {raw[:200]}")

            except requests.exceptions.Timeout:
                logger.warning(f"LLM timeout on attempt {attempt}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"LLM connection error on attempt {attempt}: {e}")
                self._available = False
                return None
            except requests.exceptions.HTTPError as e:
                logger.warning(f"LLM HTTP error {e.response.status_code} on attempt {attempt}")
            except Exception as e:
                logger.warning(f"LLM unexpected error on attempt {attempt}: {e}")

            if attempt <= retries:
                time.sleep(2 ** attempt)

        logger.error("All LLM attempts failed")
        return None

    @staticmethod
    def _safe_json_parse(text: str) -> Optional[Dict]:
        """Parse JSON from LLM output, handling common formatting issues."""
        if not text or not text.strip():
            return None

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last fence lines
            inner = "\n".join(
                l for l in lines
                if not l.strip().startswith("```")
            )
            text = inner.strip()

        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Try to extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    def generate_code(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        timeout: int = 60,
    ) -> Optional[str]:
        """
        Generate free-form code (not forced JSON).
        Returns raw string output or None on failure.
        Used by ReproAgent to get Python code directly.
        """
        if not self.is_available():
            return None

        full_system = (
            "You are an expert Python developer. "
            "Output ONLY valid Python code with no markdown fences, no explanation. "
            "Start directly with the first line of Python (shebang or import or comment). "
            "The code must be self-contained and runnable with: python script.py"
        )
        if system_prompt:
            full_system += f"\n\n{system_prompt}"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": full_system,
            "stream": False,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": max_tokens,
                "top_p": 0.9,
            },
            # No "format": "json" — we want raw Python
        }

        for attempt in range(1, 3):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
                resp.raise_for_status()
                raw = resp.json().get("response", "").strip()

                # Strip markdown fences if LLM still added them
                if raw.startswith("```"):
                    lines = raw.split("\n")
                    raw = "\n".join(
                        l for l in lines if not l.strip().startswith("```")
                    ).strip()

                if raw and len(raw) > 30:
                    return raw

            except requests.exceptions.Timeout:
                logger.warning(f"[LLMClient] generate_code timeout on attempt {attempt}")
            except requests.exceptions.ConnectionError:
                self._available = False
                return None
            except Exception as e:
                logger.warning(f"[LLMClient] generate_code error on attempt {attempt}: {e}")

            if attempt < 2:
                import time as _time
                _time.sleep(1)

        return None
