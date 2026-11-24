"""Per-request metrics log: one JSON line per answered question in logs/requests.jsonl.

This is the raw material for LLM observability (latency, tokens, cost per request).
The API key is never part of a record.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_request(logs_dir: Path, record: dict[str, Any]) -> None:
    """Append `record` (plus a UTC timestamp) to logs/requests.jsonl."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    line = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    with (logs_dir / "requests.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
