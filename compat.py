"""Compatibility shims (e.g. torch.load across PyTorch versions)."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, BinaryIO, Optional, Union

import torch

_TORCH_LOAD_HAS_WEIGHTS_ONLY = "weights_only" in inspect.signature(torch.load).parameters


def torch_load(
    f: Union[str, Path, BinaryIO],
    map_location: Any = None,
    *,
    weights_only: Optional[bool] = False,
) -> Any:
    """torch.load with optional weights_only (PyTorch >= 2.0)."""
    if _TORCH_LOAD_HAS_WEIGHTS_ONLY:
        return torch.load(f, map_location=map_location, weights_only=weights_only)
    return torch.load(f, map_location=map_location)
