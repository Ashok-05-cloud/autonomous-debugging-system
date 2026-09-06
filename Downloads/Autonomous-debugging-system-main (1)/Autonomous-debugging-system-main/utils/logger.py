"""
Centralized logging setup.
Uses loguru for structured, colored output with file rotation.
"""
import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "outputs/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure root logger
_configured = False

def _configure():
    global _configured
    if _configured:
        return

    fmt = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
