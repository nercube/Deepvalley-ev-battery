# src/utils/audit_logger.py

import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "audit.log"


def log_event(action: str, status: str, details: dict | None = None):
    """
    Append an audit event to audit.log
    """
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "status": status,
        "details": details or {}
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
