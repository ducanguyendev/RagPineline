"""Report the accelerator that will be used for index builds."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.device_utils import (  # noqa: E402
    get_device_name,
    get_device_properties,
    is_cuda_available,
    is_xpu_available,
    resolve_device,
)


def _format_property(name: str, value: object) -> str:
    if name == "total_memory" and isinstance(value, int):
        return f"{value / (1024**2):.0f} MB"
    return str(value)


def main() -> None:
    load_dotenv(ROOT / ".env")

    xpu_available = is_xpu_available()
    cuda_available = is_cuda_available()

    print(f"Python version : {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available  : {xpu_available}")
    print(f"CUDA available : {cuda_available}")

    if xpu_available:
        print(f"XPU devices    : {torch.xpu.device_count()}")
        for index in range(torch.xpu.device_count()):
            device = f"xpu:{index}"
            print(f"  {device}          : {get_device_name(device)}")
            for key, value in get_device_properties(device).items():
                if key != "name":
                    print(f"    {key}: {_format_property(key, value)}")

    if cuda_available:
        print(f"CUDA devices   : {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            device = f"cuda:{index}"
            print(f"  {device}         : {get_device_name(device)}")

    selected = resolve_device(os.getenv("INDEX_DEVICE", "auto"))
    print(f"Selected indexing device: {selected}")


if __name__ == "__main__":
    main()
