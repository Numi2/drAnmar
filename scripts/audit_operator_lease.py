#!/usr/bin/env python3
"""Deterministic source-level gate for shared-workstation ownership."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from dr_anmar_operator import OperatorLease, access_cookie_value, access_is_authorized  # noqa: E402


def main() -> None:
    lease = OperatorLease(ttl_seconds=0.02)
    first = "browser-11111111-1111-4111-8111-111111111111"
    second = "browser-22222222-2222-4222-8222-222222222222"
    assert lease.claim(first)[0]
    assert lease.claim(first)[0]
    assert not lease.claim(second)[0]
    assert lease.release(first)
    assert lease.claim(second)[0]
    time.sleep(0.03)
    assert lease.claim(first)[0]
    previous = os.environ.get("DR_ANMAR_ACCESS_TOKEN")
    try:
        os.environ["DR_ANMAR_ACCESS_TOKEN"] = "unit-test-secret"
        assert access_is_authorized(access_cookie_value("unit-test-secret"))
        assert not access_is_authorized(access_cookie_value("wrong-secret"))
    finally:
        if previous is None:
            os.environ.pop("DR_ANMAR_ACCESS_TOKEN", None)
        else:
            os.environ["DR_ANMAR_ACCESS_TOKEN"] = previous
    print("Operator lease: exclusivity, release, expiry, and opt-in access token verified")


if __name__ == "__main__":
    main()
