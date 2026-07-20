#!/usr/bin/env python3
"""Fail closed when the Dr.Anmar evidence ledger becomes ambiguous."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/VALIDATION_GATES.json"
STATES = {"source_ready", "pending_live", "blocked_host", "blocked_dependency", "blocked_external", "validated"}
PRIORITIES = {"P0", "P1", "P2"}


def main() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise SystemExit("Validation ledger has no gates")
    identifiers: set[str] = set()
    for gate in gates:
        identifier = gate.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise SystemExit(f"Invalid or duplicate validation gate: {identifier!r}")
        identifiers.add(identifier)
        if gate.get("state") not in STATES or gate.get("priority") not in PRIORITIES:
            raise SystemExit(f"Invalid state or priority for {identifier}")
        if not gate.get("owner") or not gate.get("requirement"):
            raise SystemExit(f"Gate {identifier} is missing an owner or requirement")
        evidence = gate.get("evidence")
        if not isinstance(evidence, list):
            raise SystemExit(f"Gate {identifier} evidence must be a list")
        if gate["state"] == "validated" and not evidence:
            raise SystemExit(f"Validated gate {identifier} has no evidence")
        for item in evidence:
            if not isinstance(item, str) or item.startswith("/") or not (ROOT / item).exists():
                raise SystemExit(f"Gate {identifier} references missing or unsafe evidence: {item!r}")
    if payload.get("clinical_ready") and any(gate["state"] != "validated" for gate in gates):
        raise SystemExit("Clinical-ready claim is inconsistent with unresolved validation gates")
    print(f"Validation ledger: {len(gates)} explicit gates; runtime_ready={payload.get('runtime_ready')}; clinical_ready={payload.get('clinical_ready')}")


if __name__ == "__main__":
    main()
