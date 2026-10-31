"""Fail-closed serialization, hashing, and RNG contracts for Route A diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch


DIAGNOSTIC_SCHEMA_VERSION = "routea_causal_diagnostic_v1"
DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024


class DiagnosticFailure(RuntimeError):
    """A machine-readable diagnostic refusal with a stable process exit code."""

    def __init__(self, status: str, detail: str, exit_code: int = 2):
        super().__init__(f"{status}: {detail}")
        self.status = str(status)
        self.detail = str(detail)
        self.exit_code = int(exit_code)


def _is_numeric_dtype(dtype: np.dtype) -> bool:
    return not dtype.hasobject and np.issubdtype(dtype, np.number)


def _validated_numeric_array(value: Any, *, status: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as exc:
        raise DiagnosticFailure(status, f"cannot convert value to ndarray: {exc}") from exc
    if not _is_numeric_dtype(array.dtype):
        raise DiagnosticFailure(status, f"non-numeric or object dtype: {array.dtype}")
    if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
        raise DiagnosticFailure(status, f"non-finite values in dtype {array.dtype}")
    return np.ascontiguousarray(array)


def canonical_array_sha256(value: Any) -> str:
    """Hash a numeric array by dtype string, shape, and C-order bytes."""

    array = _validated_numeric_array(value, status="INVALID_NUMERIC_ARTIFACT")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_tree_node(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = _validated_numeric_array(value, status="INVALID_CANONICAL_VALUE")
        return {
            "type": "array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": canonical_array_sha256(array),
        }
    if isinstance(value, np.generic):
        return _canonical_tree_node(value.item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise DiagnosticFailure(
                "INVALID_CANONICAL_VALUE", "mapping keys must all be strings"
            )
        return {
            "type": "mapping",
            "items": [
                [key, _canonical_tree_node(value[key])] for key in sorted(value)
            ],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_canonical_tree_node(v) for v in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_canonical_tree_node(v) for v in value]}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DiagnosticFailure("INVALID_CANONICAL_VALUE", "non-finite float")
        return value
    raise DiagnosticFailure(
        "INVALID_CANONICAL_VALUE", f"unsupported value type: {type(value).__name__}"
    )


def canonical_tree_sha256(value: Any) -> str:
    node = _canonical_tree_node(value)
    encoded = json.dumps(
        node, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise DiagnosticFailure(
                "INVALID_JSON_ARTIFACT", "non-scalar tensor is not valid JSON"
            )
        return _json_ready(value.detach().cpu().item())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise DiagnosticFailure(
                "INVALID_JSON_ARTIFACT", "JSON object keys must be strings"
            )
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DiagnosticFailure("INVALID_JSON_ARTIFACT", "non-finite float")
        return value
    raise DiagnosticFailure(
        "INVALID_JSON_ARTIFACT", f"unsupported JSON type: {type(value).__name__}"
    )


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path.parent), prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_strict_json(path: Path | str, payload: Any) -> Dict[str, Any]:
    target = Path(path)
    ready = _json_ready(payload)
    try:
        encoded = (
            json.dumps(
                ready,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiagnosticFailure("INVALID_JSON_ARTIFACT", str(exc)) from exc
    _atomic_replace_bytes(target, encoded)
    return {
        "path": str(target),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticFailure("INVALID_JSON_ARTIFACT", f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise DiagnosticFailure("INVALID_JSON_ARTIFACT", f"non-finite constant: {value}")


def load_strict_json(
    path: Path | str, max_bytes: int = DEFAULT_MAX_JSON_BYTES
) -> Any:
    target = Path(path)
    size = target.stat().st_size
    if size > int(max_bytes):
        raise DiagnosticFailure(
            "ARTIFACT_TOO_LARGE", f"{target} has {size} bytes; limit={max_bytes}"
        )
    try:
        text = target.read_text(encoding="utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except DiagnosticFailure:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticFailure("INVALID_JSON_ARTIFACT", f"{target}: {exc}") from exc


def _validate_npz_arrays(
    arrays: Mapping[str, Any], *, max_total_bytes: int
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not arrays or not all(isinstance(name, str) and name for name in arrays):
        raise DiagnosticFailure(
            "INVALID_NUMERIC_ARTIFACT", "NPZ requires non-empty string array names"
        )
    validated: Dict[str, np.ndarray] = {}
    metadata: Dict[str, Any] = {}
    total = 0
    for name in sorted(arrays):
        array = _validated_numeric_array(
            arrays[name], status="INVALID_NUMERIC_ARTIFACT"
        )
        total += int(array.nbytes)
        if total > int(max_total_bytes):
            raise DiagnosticFailure(
                "ARTIFACT_TOO_LARGE",
                f"uncompressed NPZ arrays use {total} bytes; limit={max_total_bytes}",
            )
        validated[name] = array
        metadata[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "sha256": canonical_array_sha256(array),
        }
    return validated, {"total_nbytes": total, "arrays": metadata}


def write_numeric_npz(
    path: Path | str,
    arrays: Mapping[str, Any],
    *,
    max_total_bytes: int,
) -> Dict[str, Any]:
    target = Path(path)
    validated, manifest = _validate_npz_arrays(
        arrays, max_total_bytes=max_total_bytes
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=str(target.parent), prefix=f".{target.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **validated)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    payload = target.read_bytes()
    return {
        "schema": "routea_numeric_npz_manifest_v1",
        "path": str(target),
        "file_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        **manifest,
    }


def load_numeric_npz(
    path: Path | str,
    *,
    expected_shapes: Optional[Mapping[str, tuple[int, ...]]] = None,
    max_total_bytes: int,
) -> Dict[str, np.ndarray]:
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError) as exc:
        raise DiagnosticFailure("INVALID_NUMERIC_ARTIFACT", f"{target}: {exc}") from exc
    validated, _ = _validate_npz_arrays(arrays, max_total_bytes=max_total_bytes)
    if expected_shapes is not None:
        for name, shape in expected_shapes.items():
            if name not in validated or tuple(validated[name].shape) != tuple(shape):
                actual = None if name not in validated else tuple(validated[name].shape)
                raise DiagnosticFailure(
                    "NPZ_SHAPE_MISMATCH",
                    f"{name}: expected={tuple(shape)} actual={actual}",
                )
    return validated


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise DiagnosticFailure(
            "INVALID_RNG_STATE", f"expected keys={sorted(required)} actual={sorted(state)}"
        )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].clone())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise DiagnosticFailure(
                "INVALID_RNG_STATE", "CUDA RNG state supplied but CUDA is unavailable"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise DiagnosticFailure(
                "INVALID_RNG_STATE",
                f"CUDA RNG count={len(cuda_states)} devices={torch.cuda.device_count()}",
            )
        torch.cuda.set_rng_state_all([item.clone() for item in cuda_states])


def rng_state_sha256(state: Mapping[str, Any]) -> str:
    return canonical_tree_sha256(state)


def _file_record(repo_root: Path, path: Path) -> Dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_diagnostic_source_manifest(repo_root: Path | str) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "scripts" / "routea_causal_diagnostics"
    candidates = list(package.rglob("*.py")) if package.is_dir() else []
    cli = root / "scripts" / "probe_routea_goal_sensitivity.py"
    if cli.is_file():
        candidates.append(cli)
    paths = sorted(
        {path.resolve() for path in candidates if "__pycache__" not in path.parts}
    )
    if not paths:
        raise DiagnosticFailure(
            "DIAGNOSTIC_SOURCE_MISSING", f"no diagnostic Python source under {root}"
        )
    records = [_file_record(root, path) for path in paths]
    content_sha256 = canonical_tree_sha256(
        [[row["path"], row["sha256"], row["bytes"]] for row in records]
    )
    return {
        "schema": "routea_causal_diagnostic_source_manifest_v1",
        "content_sha256": content_sha256,
        "file_count": len(records),
        "files": records,
    }
