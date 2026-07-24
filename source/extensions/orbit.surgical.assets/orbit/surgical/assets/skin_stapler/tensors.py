"""Compatibility helpers for Torch tensors, Warp arrays, and Isaac proxy arrays."""
from __future__ import annotations

from typing import Any


def unwrap_tensor(value: Any) -> Any:
    """Return a Torch-compatible value when the runtime exposes one.

    Isaac Lab 2.x commonly returns ``torch.Tensor`` directly. Isaac Lab 3.x may
    return a proxy exposing ``.torch`` or a native Warp array. Imports remain
    lazy so this helper can be imported outside an Isaac runtime.
    """

    if hasattr(value, "torch"):
        candidate = value.torch
        return candidate() if callable(candidate) else candidate

    try:
        import warp as wp  # type: ignore

        if isinstance(value, wp.array):
            return wp.to_torch(value)
    except (ImportError, ModuleNotFoundError, TypeError):
        pass

    return value
