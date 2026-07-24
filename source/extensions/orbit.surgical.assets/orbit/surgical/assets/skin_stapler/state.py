"""Discrete magazine state handling.

State must be selected while the USD is spawned, before Isaac creates physics
views. Trigger and pusher motion are joint state, not USD variants.
"""
from __future__ import annotations

from enum import Enum


class StaplerState(str, Enum):
    """Discrete USD variant states supported by the stapler assets.

    ``str`` + ``Enum`` is used instead of :class:`enum.StrEnum` so the helper
    remains importable in Python 3.10 environments as well as the newer Python
    runtimes used by Isaac Sim 5.1 and 6.0.
    """

    LOADED = "loaded"
    EMPTY = "empty"

    def __str__(self) -> str:
        return self.value


def coerce_state(value: str | StaplerState) -> StaplerState:
    if isinstance(value, StaplerState):
        return value
    try:
        return StaplerState(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in StaplerState)
        raise ValueError(f"Unsupported stapler state {value!r}; expected one of: {allowed}") from exc


def variant_selection(value: str | StaplerState) -> dict[str, str]:
    """Return an Isaac ``UsdFileCfg.variants`` mapping."""
    return {"state": coerce_state(value).value}


def semantic_state_label(value: str | StaplerState) -> tuple[str, str]:
    return ("state", coerce_state(value).value)
