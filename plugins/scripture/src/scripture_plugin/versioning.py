from __future__ import annotations

import hashlib
import json
from typing import Any


def configured_version(value: Any) -> str:
    """Make evaluator version IDs reflect plugin-owned metadata/config snapshots."""
    if hasattr(value, "profiles"):
        payload = {
            name: {"voice_profile": profile.voice_profile, "books": profile.books}
            for name, profile in sorted(value.profiles.items())
        }
    elif hasattr(value, "entries"):
        payload = [entry.to_prompt_dict() for entry in value.entries]
    else:
        payload = value
    digest = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=list, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:10]
    return f"1.0.0+cfg.{digest}"
