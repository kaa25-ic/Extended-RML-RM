"""Path utilities for integrating the inherited baseline code."""

from __future__ import annotations

import sys
from pathlib import Path


def code_root() -> Path:
    """Return the root directory of the project code repository."""
    return Path(__file__).resolve().parents[2]


def legacy_root() -> Path:
    """Return the root directory of the inherited baseline code."""
    return code_root() / "legacy"


def legacy_python_root() -> Path:
    """Return the Python package root of the inherited baseline."""
    return legacy_root() / "rml_reward_machines"


def legacy_rml_root() -> Path:
    """Return the Prolog monitor root of the inherited baseline."""
    return legacy_root() / "RML"


def ensure_legacy_python_path() -> None:
    """Prepend the inherited baseline Python root to ``sys.path`` if needed."""
    legacy_path = str(legacy_python_root())
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
