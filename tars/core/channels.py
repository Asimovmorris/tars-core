from __future__ import annotations

from enum import Enum

# Compatibility shim:
# server.py expects: from tars.core.channels import ResponseChannel
# If a canonical ResponseChannel exists elsewhere, re-export it. Otherwise define a safe fallback.

try:
    # Most likely place (based on your project): core/chat.py
    from tars.core.chat import ResponseChannel as ResponseChannel  # type: ignore
except Exception:
    class ResponseChannel(str, Enum):
        VOICE = "VOICE"
        TEXT = "TEXT"
