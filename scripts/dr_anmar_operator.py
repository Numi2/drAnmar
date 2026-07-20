# Copyright (c) 2026, Dr.Anmar Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Single-operator browser lease for shared Dr.Anmar workstations."""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from dataclasses import dataclass, field


OPERATOR_HEADER = "x-dr-anmar-operator"
ACCESS_COOKIE = "dr_anmar_access"
_OPERATOR_ID = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")


@dataclass
class OperatorLease:
    """Allow observation from many browsers but mutation from one live session."""

    ttl_seconds: float = 30.0
    _operator_id: str | None = None
    _last_seen: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def claim(self, operator_id: str | None) -> tuple[bool, str]:
        if not operator_id or not _OPERATOR_ID.fullmatch(operator_id):
            return False, "This browser has no valid Dr.Anmar operator identity. Reload Doctor Studio."
        now = time.monotonic()
        with self._lock:
            expired = self._operator_id is None or now - self._last_seen > self.ttl_seconds
            if expired or self._operator_id == operator_id:
                self._operator_id = operator_id
                self._last_seen = now
                return True, ""
            remaining = max(1, round(self.ttl_seconds - (now - self._last_seen)))
        return False, f"Another browser is controlling this workstation. It releases automatically in {remaining}s."

    def release(self, operator_id: str | None) -> bool:
        with self._lock:
            if operator_id and operator_id == self._operator_id:
                self._operator_id = None
                self._last_seen = 0.0
                return True
        return False

    def status(self) -> dict[str, float | bool]:
        now = time.monotonic()
        with self._lock:
            remaining = max(0.0, self.ttl_seconds - (now - self._last_seen)) if self._operator_id else 0.0
            active = self._operator_id is not None and remaining > 0.0
            if not active:
                self._operator_id = None
                self._last_seen = 0.0
        return {"active": active, "expires_in_s": round(remaining, 1), "ttl_s": self.ttl_seconds}


def configured_access_token() -> str | None:
    value = os.environ.get("DR_ANMAR_ACCESS_TOKEN", "").strip()
    return value or None


def access_cookie_value(token: str) -> str:
    return hashlib.sha256(f"dr-anmar-access-v1:{token}".encode()).hexdigest()


def access_is_authorized(cookie_value: str | None, token: str | None = None) -> bool:
    configured = configured_access_token()
    if configured is None:
        return True
    candidate = access_cookie_value(token) if token is not None else (cookie_value or "")
    return hmac.compare_digest(candidate, access_cookie_value(configured))
