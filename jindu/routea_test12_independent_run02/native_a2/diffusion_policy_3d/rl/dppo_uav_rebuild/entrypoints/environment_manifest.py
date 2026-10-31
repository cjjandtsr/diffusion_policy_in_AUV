"""Create a secret-safe manifest of behavior-relevant runtime state."""

from __future__ import annotations

import json
import locale
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, Mapping

import numpy as np
import torch


SAFE_ENV_KEYS = frozenset(
    {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "LC_ALL",
        "LANG",
        "TZ",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "TORCH_DISTRIBUTED_DEBUG",
    }
)


def collect_safe_environment(environ: Mapping[str, str]) -> Dict[str, Any]:
    """Select from an empty mapping; never copy then redact the environment."""

    result: Dict[str, Any] = {
        key: environ[key] for key in sorted(SAFE_ENV_KEYS) if key in environ
    }
    for key in sorted(environ):
        if key.startswith(("NCCL_", "TORCH_NCCL_")):
            result[key] = environ[key]
    result["MASTER_ADDR_present"] = "MASTER_ADDR" in environ
    if "MASTER_PORT" in environ:
        try:
            port = int(environ["MASTER_PORT"])
        except (TypeError, ValueError) as exc:
            raise ValueError("MASTER_PORT must be an integer in [1, 65535]") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MASTER_PORT must be an integer in [1, 65535]")
        result["MASTER_PORT"] = port
    return result


def _cuda_device_record(requested_device: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if record["available"] and str(requested_device).startswith("cuda"):
        device = torch.device(requested_device)
        properties = torch.cuda.get_device_properties(device)
        record["selected"] = {
            "index": int(device.index if device.index is not None else 0),
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    else:
        record["selected"] = None
    return record


def build_environment_manifest(*, device: str, dtype: str) -> Dict[str, Any]:
    """Capture versions and deterministic flags without serializing secrets."""

    preferred_encoding = locale.getpreferredencoding(False)
    return {
        "schema_version": 1,
        "requested_device": str(device),
        "requested_dtype": str(dtype),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "numpy": {"version": np.__version__},
        "torch": {
            "version": torch.__version__,
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "cuda": _cuda_device_record(str(device)),
        "locale": {
            "preferred_encoding": preferred_encoding,
            "timezone_names": list(time.tzname),
        },
        "environment": collect_safe_environment(os.environ),
    }


def write_environment_manifest(path, *, device: str, dtype: str) -> Dict[str, Any]:
    """Write the canonical runtime manifest and return the same value."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = build_environment_manifest(device=device, dtype=dtype)
    output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
