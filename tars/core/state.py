from __future__ import annotations

from enum import Enum

# Compatibility shim:
# server.py expects: from tars.core.state import ReasoningMode
# If a canonical ReasoningMode exists elsewhere, re-export it. Otherwise define a safe fallback.

try:
    from tars.core.chat import ReasoningMode as ReasoningMode  # type: ignore
except Exception:
    class ReasoningMode(str, Enum):
        ANALYST = "analyst"
        CRITIC = "critic"
        SYNTHESIZER = "synthetic"
