"""Legacy entry-point guard + canonical entry preflight (R3.3).

The legacy side is the official core's boundary: formal training runs
only through the A-Official-Core path (dppo_uav), never through a legacy
E-series entry point. The canonical side is the preflight every formal
entry (train/eval) must pass BEFORE creating env/optimizer/policy:
git clean/commit, vendor hashes, BC/manifest/contract hashes,
projection=False, train/eval split, formal numerical mode.

Classification (closure item 9):
- legacy E-series training scripts -> frozen historical (guarded);
- the dppo_uav official core -> canonical (never guarded).

The heavy imports (torch/train_loop/integration/contracts) are
function-local: the legacy scripts only import guard_formal_mode, and
their import chain must not change.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

HISTORICAL_INVALID_FOR_FORMAL_TRAINING = (
    "frozen historical — invalid for formal training"
)

# Legacy pre-official-core DPPO training entry points (frozen historical).
HISTORICAL_FORMAL_ENTRYPOINTS: Set[str] = {
    "scripts/train_ddpm_ppo.py",
    "scripts/train_ddpm_ppo_meanlogp.py",
    "scripts/train_ddpm_ppo_meanlogp_1.py",
    "scripts/train_ddpm_ppo_meanlogp_2.py",
    "scripts/train_ddpm_ppo_meanlogp_3.py",
    "scripts/train_ddpm_ppo_meanlogp_3_e039.py",
    "scripts/train_ddpm_ppo_meanlogp_3_e040.py",
}

# Eval/regression-only worlds (MasterPlan [FACT]) — the formal TRAIN pool
# must never contain them (the eval side may).
EVAL_ONLY_WORLDS: frozenset = frozenset({"w4", "w8", "w11", "w12", "w18"})

# The byte-frozen official DPPO oracle (401 vendor files).
VENDOR_DIR_NAME = "third_party/dppo_official_cc7234a"


def guard_formal_mode(entrypoint: str, mode: str) -> None:
    """Reject formal mode through a legacy E-series entry point.

    Args:
        entrypoint: repository-relative path of the entry script.
        mode: the entry's requested mode ("smoke"/"dry_run"/"formal").

    Raises:
        RuntimeError: when mode == "formal" and the entrypoint is a
            historical legacy script.
    """
    if mode == "formal" and entrypoint in HISTORICAL_FORMAL_ENTRYPOINTS:
        raise RuntimeError(
            f"{entrypoint}: {HISTORICAL_INVALID_FOR_FORMAL_TRAINING} — "
            "formal training runs only through the A-Official-Core path"
        )


def _repo_root() -> Path:
    """Repository root (this module: <root>/diffusion_policy_3d/rl/dppo_uav)."""
    root = Path(__file__).resolve().parents[3]
    if not (root / ".git").exists():
        raise RuntimeError(f"repository root not found above {root}")
    return root


def file_sha256(path: Path) -> str:
    """sha256 hexdigest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def world_assets_hash_of(worlds_dir: Path, world_uids: List[str]) -> str:
    """Aggregate sha256 over every admitted world's asset files.

    Each admitted world binds BOTH its scene file (<uid>.world) and its
    obstacle geometry (<uid>.world.obstacles.json) — a missing file is a
    hard failure (preflight must never hash around a vanished asset).
    Deterministic aggregation: sorted per-world lines
    "<uid>|<scene_sha>|<obstacles_sha>", then sha256 over the joined
    blob (Gate D reopening 2026-08-22: the external world assets live
    outside this repo — without content hashing, swapping a world file
    mid-training is invisible to every identity axis).
    """
    lines: List[str] = []
    for uid in sorted(world_uids):
        scene = worlds_dir / f"{uid}.world"
        obstacles = worlds_dir / f"{uid}.world.obstacles.json"
        if not scene.is_file():
            raise RuntimeError(f"admitted world scene missing: {scene}")
        if not obstacles.is_file():
            raise RuntimeError(
                f"admitted world obstacles missing: {obstacles}")
        lines.append(f"{uid}|{file_sha256(scene)}|{file_sha256(obstacles)}")
    if not lines:
        raise RuntimeError(f"no admitted worlds under {worlds_dir}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def official_source_file_hashes(
    repo_root: Optional[Path] = None,
) -> Dict[str, str]:
    """sha256 of every git-tracked file, repo-relative keys.

    The code-hash axis of the runtime identity (schema v2) — computed
    from ``git ls-files`` so the set is exactly the committed source. A
    file that vanished from disk is a tree inconsistency, never something
    to hash around: hard-fails.
    """
    root = repo_root or _repo_root()
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {out.stderr.strip()}")
    hashes: Dict[str, str] = {}
    for rel in out.stdout.splitlines():
        path = root / rel
        if not path.is_file():
            raise RuntimeError(
                f"git-tracked file missing from disk: {rel} — the source "
                "set is git's; a vanished file is a tree inconsistency"
            )
        hashes[rel] = file_sha256(path)
    return hashes


def verify_vendor_hashes(repo_root: Optional[Path] = None) -> None:
    """``sha256sum -c SHA256SUMS`` over the frozen vendor snapshot.

    The vendor dir (401 files) is byte-frozen; a single failing entry is
    a hard preflight failure.
    """
    root = repo_root or _repo_root()
    vendor = root / VENDOR_DIR_NAME
    sums = vendor / "SHA256SUMS"
    if not sums.is_file():
        raise RuntimeError(f"vendor SHA256SUMS not found: {sums}")
    out = subprocess.run(
        ["sha256sum", "-c", sums.name], cwd=vendor,
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        failed = [
            ln for ln in out.stdout.splitlines()
            if ln and not ln.endswith("OK")
        ]
        raise RuntimeError(
            "vendor hash verification FAILED: "
            + "; ".join(failed[:5])
            + (f" (+{len(failed) - 5} more)" if len(failed) > 5 else "")
        )


def validate_numerical_mode() -> Dict[str, bool]:
    """Apply the formal numerical mode and assert the backends took it.

    The formal entry hard-wires FORMAL_NUMERICAL_MODE (TF32 off +
    deterministic) — there is no escape hatch; this validates that the
    mode the entry runs under is the one recorded in checkpoints/reports.
    """
    import torch

    from diffusion_policy_3d.rl.dppo_uav.numerics import configure_numerical_mode
    from diffusion_policy_3d.rl.dppo_uav.train_loop import FORMAL_NUMERICAL_MODE

    resolved = configure_numerical_mode(FORMAL_NUMERICAL_MODE)
    got = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", ""),
    }
    if got != resolved:
        raise RuntimeError(
            f"formal numerical mode not in effect: resolved {resolved}, "
            f"backends {got}"
        )
    return resolved


def require_fixed_python_hashseed() -> str:
    """Formal-entry preflight (RULING-7 P0-2): a fixed non-empty
    PYTHONHASHSEED must be set in the launch environment.

    Python's str hash() is randomized per interpreter when PYTHONHASHSEED
    is unset or "random". After the env RNG fix (stable blake2b over the
    task payload) no production path depends on hash(), but the preflight
    still pins the seed so ANY residual str-hash dependence — or a future
    regression — stays reproducible across processes, and the recorded
    environment_rng_state.python_hash_seed can only prove cross-process
    reproducibility when it is non-empty.

    Raises:
        RuntimeError: PYTHONHASHSEED unset or "random".

    Returns:
        The fixed seed value (the value the checkpoint records).
    """
    seed = os.environ.get("PYTHONHASHSEED", "")
    if not seed or seed == "random":
        raise RuntimeError(
            "formal entry requires a fixed non-empty PYTHONHASHSEED "
            f"(got {seed!r}) — set PYTHONHASHSEED=<decimal> in the launch "
            "environment; unset/random leaves str hash() process-random "
            "and cannot reproduce the env RNG cross-process"
        )
    return seed


def assert_no_eval_only_worlds(pool_uids: Sequence[str]) -> None:
    """A world reserved for eval/regression must never enter the train pool."""
    leak = sorted(u for u in pool_uids if u in EVAL_ONLY_WORLDS)
    if leak:
        raise RuntimeError(
            "train/eval split violated: eval-only world(s) in the formal "
            f"train pool: {leak} — w4/w8/w11/w12/w18 are eval/regression "
            "only (MasterPlan [FACT])"
        )


def validate_train_eval_split(
    manifest_path: str,
    worlds_train_dir: str,
) -> List[str]:
    """The formal train pool (manifest rl_train buckets) is eval-clean.

    Returns the pool's world uids (also proving the pool is non-empty).
    """
    from diffusion_policy_3d.rl.dppo_uav.integration import (
        build_routea_train_pool,
    )

    pool = build_routea_train_pool(manifest_path, worlds_train_dir)
    uids = [uid for uid, _ in pool]
    if not uids:
        raise RuntimeError(
            "train/eval split: formal train pool is empty "
            f"(manifest {manifest_path})"
        )
    assert_no_eval_only_worlds(uids)
    return uids


def validate_official_environment(
    *,
    manifest_path: str,
    bc_checkpoint_path: str,
    worlds_train_dir: str,
    repo_root: Optional[Path] = None,
    allowed_untracked_prefixes: Sequence[str] = ("temp",),
) -> Dict[str, str]:
    """Preflight the canonical formal entries (R3.3 Step 3).

    Every check runs BEFORE any env/optimizer/policy construction; the
    first failure raises RuntimeError (the entries exit non-zero). Checks:
    git clean (no tracked modifications; untracked only under the allowed
    prefixes, temp/ scratch by default) and commit, vendor SHA256SUMS,
    BC/manifest file presence + hash, contract projection=False,
    train/eval split, formal numerical mode. Relative BC/manifest/worlds
    paths resolve against the repo root (the policy loader's semantics),
    never against CWD.

    Returns the runtime facts the entry records:
    {official_dppo_commit, project_git_commit, project_tree_hash,
     contract_hash, manifest_hash, bc_checkpoint_hash,
     world_assets_hash, source_file_hashes}.
    """
    root = repo_root or _repo_root()
    from diffusion_policy_3d.rl.dppo_uav.contracts import RLSystemContract
    from diffusion_policy_3d.rl.dppo_uav.train_loop import (
        OFFICIAL_DPPO_COMMIT,
    )

    # git clean: porcelain must show no tracked modification, and no
    # untracked file outside the allowed scratch prefixes.
    out = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root,
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git status failed: {out.stderr.strip()}")
    lines = [ln for ln in out.stdout.splitlines() if ln]
    modified = [ln for ln in lines if not ln.startswith("?? ")]
    untracked = [ln[3:] for ln in lines if ln.startswith("?? ")]
    bad_untracked = [
        p for p in untracked
        if not any(
            p == prefix or p.startswith(prefix + "/")
            for prefix in allowed_untracked_prefixes
        )
    ]
    if modified or bad_untracked:
        details = []
        if modified:
            details.append(f"{len(modified)} tracked modification(s)")
        if bad_untracked:
            details.append(
                f"{len(bad_untracked)} untracked file(s) outside "
                f"{tuple(allowed_untracked_prefixes)}/ "
                f"(e.g. {bad_untracked[0]})"
            )
        raise RuntimeError(
            "git tree not clean — " + "; ".join(details)
        )
    # git commit: record the exact commit/tree the run binds to.
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root,
        capture_output=True, text=True,
    )
    if head.returncode != 0 or tree.returncode != 0:
        raise RuntimeError("git rev-parse failed")
    commit = head.stdout.strip()
    tree_hash = tree.stdout.strip()
    # vendor hashes: the 401 frozen official files.
    verify_vendor_hashes(root)
    # BC/manifest paths may be repo-relative (the policy loader resolves
    # against the project root, not CWD) — resolve them the same way here,
    # so the preflight never checks a CWD-relative phantom while the
    # loader reads a root-relative file.
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else root / path

    bc_path = _resolve(bc_checkpoint_path)
    manifest = _resolve(manifest_path)
    worlds_dir = _resolve(worlds_train_dir)
    if not bc_path.is_file():
        raise RuntimeError(f"BC checkpoint not found: {bc_path}")
    if not manifest.is_file():
        raise RuntimeError(f"admission manifest not found: {manifest}")
    # Contract: obstacle projection is policy-external trajectory rewriting
    # (策略边界) — the formal core rejects it at every boundary.
    contract = RLSystemContract()
    if contract.use_obstacle_projection:
        raise RuntimeError(
            "contract use_obstacle_projection must be False for formal "
            "training/eval (§7.4) — obstacle projection is policy-external"
        )
    # Train/eval split: the formal train pool is eval-clean and non-empty.
    train_uids = validate_train_eval_split(str(manifest), str(worlds_dir))
    # Gate D reopening (2026-08-22): the external world assets are part
    # of the training identity — the aggregate hash over every admitted
    # world's scene + obstacle files hard-fails on a missing asset and
    # binds the exact file contents (swapping a world file after
    # preflight must be caught at resume/load).
    world_hash = world_assets_hash_of(worlds_dir, train_uids)
    # Formal numerical mode in effect.
    validate_numerical_mode()
    # RULING-7 P0-2: the env episode RNG is a stable blake2b of the task
    # payload — but the recorded environment_rng_state.python_hash_seed
    # can only prove cross-process reproducibility when the seed is
    # fixed and non-empty. Every formal entry must pin it.
    require_fixed_python_hashseed()
    from diffusion_policy_3d.rl.dppo_uav.checkpoint import contract_hash_of

    return {
        "official_dppo_commit": OFFICIAL_DPPO_COMMIT,
        "project_git_commit": commit,
        "project_tree_hash": tree_hash,
        "contract_hash": contract_hash_of(contract),
        "manifest_hash": file_sha256(manifest),
        "bc_checkpoint_hash": file_sha256(bc_path),
        "world_assets_hash": world_hash,
        "source_file_hashes": official_source_file_hashes(root),
    }
