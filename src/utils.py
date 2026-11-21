"""Small shared helpers: config loading, git metadata, logging and retry."""
from __future__ import annotations

import csv
import functools
import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator, TypeVar

import yaml

from src.schema import SelectionRecord

T = TypeVar("T")


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    """Read a YAML config file and return it as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return cfg


def git_hash() -> str:
    """Short commit hash (suffix ``-dirty`` if the tree has changes), or ``no-git``."""
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger (idempotent)."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def retry(times: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: tuple[type[BaseException], ...] = (Exception,)) -> Callable:
    """Decorator: retry a flaky call (e.g. a download or API request) with exponential backoff."""
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            wait = delay
            for attempt in range(1, times + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions:
                    if attempt == times:
                        raise
                    logging.getLogger(__name__).warning(
                        "%s failed (attempt %d/%d), retrying in %.1fs", fn.__name__, attempt, times, wait)
                    time.sleep(wait)
                    wait *= backoff
            raise RuntimeError("unreachable")
        return wrapper
    return deco


def read_selection_log(path: str | Path = "configs/selection_log.csv") -> Iterator[SelectionRecord]:
    """Yield validated rows of the data-selection log."""
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield SelectionRecord(**row)
