#!/usr/bin/env python3
"""Q3: read the E5 checkpoint normalizer (action scale/offset, 11-dim).

Read-only diagnostic; does not modify any checkpoint or tracked file.
Verifies the checkpoint SHA256 against the P2-bound hash (c1fca4ac...)
before extracting normalizer tensors.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

CKPT = Path(sys.argv[1] if len(sys.argv) > 1 else
             "/home/cjj/AUV/Improved-3D-Diffusion-Policy-main/Improved-3D-Diffusion-Policy/"
             "data/outputs/2026.08.16/08.53.26_e5_lambda0p5/checkpoints/latest.ckpt")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else
           "/home/cjj/AUV/plan/q2q3q4_20261025/q3_e5_normalizer_result.json")
P2_BOUND_SHA = "c1fca4ac2735928b18fa7a9c069f90ad181eb84092acee379b073dc6a61bfc9a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_tensors(obj, prefix=""):
    """Recursively collect (flat_key -> tensor) from the loaded object."""
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(collect_tensors(value, "{}.{}".format(prefix, key) if prefix else str(key)))
    elif hasattr(obj, "items"):  # nn.ParameterDict / OrderedDict
        for key, value in obj.items():
            out.update(collect_tensors(value, "{}.{}".format(prefix, key) if prefix else str(key)))
    elif torch.is_tensor(obj):
        out[prefix] = obj
    return out


def main():
    print("checkpoint:", CKPT)
    actual = sha256_file(CKPT)
    print("sha256:", actual)
    print("P2 bound match:", actual == P2_BOUND_SHA)

    print("loading checkpoint (1.1G, may take ~1 min) ...", flush=True)
    payload = torch.load(str(CKPT), map_location="cpu")
    tensors = collect_tensors(payload)
    print("collected tensor keys:", len(tensors))

    action = {k: v for k, v in tensors.items()
              if k.endswith("normalizer.params_dict.action.scale")
              or k.endswith("normalizer.params_dict.action.offset")
              or "normalizer.params_dict.action.input_stats" in k}
    result = {
        "checkpoint": str(CKPT),
        "sha256": actual,
        "p2_bound_sha256": P2_BOUND_SHA,
        "sha256_match_p2_bound": actual == P2_BOUND_SHA,
        "action": {},
    }
    for key, tensor in sorted(action.items()):
        values = tensor.detach().cpu().numpy().tolist()
        short = key.split("normalizer.params_dict.")[-1]
        result["action"][short] = values
        print(short, "shape", list(tensor.shape))
        print("  ", [round(float(x), 10) for x in values])

    scale = result["action"].get("action.scale")
    offset = result["action"].get("action.offset")
    if scale is not None and len(scale) >= 10:
        import math
        print("delta_yaw scale (ch9):", scale[9])
        print("0.1 normalized dyaw = %g rad = %g deg" %
              (0.1 / scale[9], math.degrees(0.1 / scale[9])))
    if offset is not None:
        print("delta_yaw offset (ch9):", offset[9])

    # P2 cross-check for the four known values
    p2_known = {
        "scale[0] dX": 4.2736897468566895,
        "scale[1] dY": 4.796350955963135,
        "scale[2] dZ": 27.20646095275879,
        "offset[9] dYaw": 0.00027647707611322403,
    }
    checks = {}
    if scale is not None:
        checks["scale0_dx"] = bool(abs(scale[0] - p2_known["scale[0] dX"]) < 1e-9)
        checks["scale1_dy"] = bool(abs(scale[1] - p2_known["scale[1] dY"]) < 1e-9)
        checks["scale2_dz"] = bool(abs(scale[2] - p2_known["scale[2] dZ"]) < 1e-9)
    if offset is not None:
        checks["offset9_dyaw"] = bool(abs(offset[9] - p2_known["offset[9] dYaw"]) < 1e-9)
    result["p2_cross_check"] = checks
    print("P2 cross-check:", checks)

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
