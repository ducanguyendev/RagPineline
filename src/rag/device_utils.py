"""Portable accelerator helpers for indexing and diagnostics."""

from __future__ import annotations

import platform
from typing import Any

import torch


def is_xpu_available() -> bool:
    """Return whether this PyTorch build can use an Intel XPU device."""
    try:
        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    except (AttributeError, RuntimeError):
        return False


def is_cuda_available() -> bool:
    """Return whether this PyTorch build can use a CUDA device."""
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def _parse_device_index(device: str, accelerator: str) -> int:
    if device == accelerator:
        return 0

    prefix = f"{accelerator}:"
    if not device.startswith(prefix):
        raise ValueError(f"Unsupported device: {device}")

    raw_index = device[len(prefix) :]
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"Invalid {accelerator.upper()} device index: {device}") from exc

    if index < 0:
        raise ValueError(f"Invalid {accelerator.upper()} device index: {device}")
    return index


def resolve_device(preferred: str = "auto") -> str:
    """Resolve ``auto``, CPU, CUDA, or XPU preferences to a concrete device."""
    requested = (preferred or "auto").strip().lower()

    if requested == "auto":
        if is_xpu_available():
            return "xpu:0"
        if is_cuda_available():
            return "cuda:0"
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if requested == "xpu" or requested.startswith("xpu:"):
        index = _parse_device_index(requested, "xpu")
        if not is_xpu_available():
            raise RuntimeError("XPU was requested but is not available in this PyTorch build.")
        if index >= torch.xpu.device_count():
            raise RuntimeError(
                f"XPU device index {index} is out of range; "
                f"available devices: {torch.xpu.device_count()}."
            )
        return f"xpu:{index}"

    if requested == "cuda" or requested.startswith("cuda:"):
        index = _parse_device_index(requested, "cuda")
        if not is_cuda_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch build.")
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {index} is out of range; "
                f"available devices: {torch.cuda.device_count()}."
            )
        return f"cuda:{index}"

    raise ValueError(
        f"Unsupported device preference '{preferred}'. "
        "Use auto, cpu, cuda, cuda:N, xpu, or xpu:N."
    )


def get_device_name(device: str) -> str:
    """Return a human-readable name for a resolved device."""
    resolved = device.strip().lower()
    if resolved == "cpu":
        return platform.processor() or "CPU"
    if resolved.startswith("xpu"):
        index = _parse_device_index(resolved, "xpu")
        return str(torch.xpu.get_device_name(index))
    if resolved.startswith("cuda"):
        index = _parse_device_index(resolved, "cuda")
        return str(torch.cuda.get_device_name(index))
    raise ValueError(f"Unsupported device: {device}")


def get_device_properties(device: str) -> dict[str, Any]:
    """Return serializable device properties when the backend exposes them."""
    resolved = device.strip().lower()
    if resolved == "cpu":
        return {"name": get_device_name(resolved)}

    if resolved.startswith("xpu"):
        index = _parse_device_index(resolved, "xpu")
        properties = torch.xpu.get_device_properties(index)
    elif resolved.startswith("cuda"):
        index = _parse_device_index(resolved, "cuda")
        properties = torch.cuda.get_device_properties(index)
    else:
        raise ValueError(f"Unsupported device: {device}")

    result: dict[str, Any] = {"name": get_device_name(resolved)}
    for attribute in (
        "total_memory",
        "driver_version",
        "max_compute_units",
        "multi_processor_count",
        "has_fp16",
        "is_integrated_gpu",
    ):
        if hasattr(properties, attribute):
            value = getattr(properties, attribute)
            if isinstance(value, (str, int, float, bool)):
                result[attribute] = value
    return result


def clear_device_cache(device: str) -> None:
    """Release unused accelerator cache; CPU is intentionally a no-op."""
    resolved = device.strip().lower()
    try:
        if resolved.startswith("xpu") and is_xpu_available():
            torch.xpu.empty_cache()
        elif resolved.startswith("cuda") and is_cuda_available():
            torch.cuda.empty_cache()
    except (AttributeError, RuntimeError):
        # Cache cleanup must never mask the original accelerator failure.
        return


def synchronize_device(device: str) -> None:
    """Wait for queued accelerator work so performance timings are accurate."""
    resolved = device.strip().lower()
    if resolved.startswith("xpu") and is_xpu_available():
        torch.xpu.synchronize()
    elif resolved.startswith("cuda") and is_cuda_available():
        torch.cuda.synchronize()
