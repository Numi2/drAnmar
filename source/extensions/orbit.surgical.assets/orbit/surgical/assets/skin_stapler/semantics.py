"""Semantic-tag helpers with current and legacy Isaac fallbacks."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .state import StaplerState, coerce_state

# Semantic types are intentionally unique. Some legacy semantic APIs replace a
# previous value when the same type is authored more than once.
DEVICE_TAGS: tuple[tuple[str, str], ...] = (
    ("class", "skin_stapler"),
    ("device_type", "surgical_closure_device"),
    ("workflow_handover", "handover"),
    ("workflow_closure", "closure"),
    ("workflow_disposal", "disposal"),
)
STAPLE_TAGS: tuple[tuple[str, str], ...] = (
    ("class", "skin_staple"),
    ("workflow_closure", "closure"),
    ("workflow_disposal", "disposal"),
)


def semantic_tags(state: str | StaplerState | None = None) -> list[tuple[str, str]]:
    """Return spawn-time semantic tags for a stapler instance."""
    tags = list(DEVICE_TAGS)
    if state is not None:
        tags.append(("state", coerce_state(state).value))
    return tags


def staple_semantic_tags() -> list[tuple[str, str]]:
    return list(STAPLE_TAGS)


def apply_semantic_labels(prim: Any, labels: Iterable[str], *, instance_name: str = "class") -> str:
    """Apply labels through the current API, then the supported legacy API.

    Returns the route used. Import failures deliberately catch both
    ``ImportError`` and ``ModuleNotFoundError``. Runtime errors from an imported
    API are not hidden because they indicate a real integration problem.
    """
    normalized = tuple(dict.fromkeys(str(label) for label in labels))
    if not normalized:
        raise ValueError("At least one semantic label is required")

    try:
        from isaaclab.sim.utils.semantics import add_labels  # type: ignore

        add_labels(prim, labels=list(normalized), instance_name=instance_name)
        return "isaaclab.sim.utils.semantics.add_labels"
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        from isaacsim.core.utils.semantics import add_update_semantics  # type: ignore

        for index, label in enumerate(normalized):
            suffix = instance_name if index == 0 else f"{instance_name}_{index}"
            add_update_semantics(prim, semantic_label=label, type_label=suffix)
        return "isaacsim.core.utils.semantics.add_update_semantics"
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Neither the current Isaac Lab semantic helper nor the supported "
            "legacy Isaac Sim semantic API is importable."
        ) from exc
