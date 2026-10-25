"""R2.6 contracts (guide Task R2.6 Step 1): production caller same-path.

Static (AST) assertions that the G3 gate audit and the train loop consume
the SAME stateful EpisodeTaskSampler — no independently rebuilt role
RNG/cursor in either path — and that the G3 CLI defaults to the production
manifest/worlds via the DPPO_MANIFEST / DPPO_WORLDS_TRAIN env-var
convention (same as tests/dppo_parity/conftest.py).

RED contract (before the G3 CLI rework): the CLI requires --manifest and
--worlds-dir, so `test_g3_cli_defaults_to_production_paths` fails.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_AUDIT = REPO_ROOT / "diffusion_policy_3d/rl/dppo_uav/gate_audit.py"
TRAIN_LOOP = REPO_ROOT / "diffusion_policy_3d/rl/dppo_uav/train_loop.py"
G3_CLI = REPO_ROOT / "scripts/g3_10k_reset_gate_audit.py"


def _parse(rel_path: str) -> ast.Module:
    with open(REPO_ROOT / rel_path, "r", encoding="utf-8") as f:
        return ast.parse(f.read(), filename=rel_path)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _has_call(func: ast.FunctionDef, callee_name: str) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == callee_name:
                return True
            if isinstance(f, ast.Attribute) and f.attr == callee_name:
                return True
    return False


def _has_attr_call(func: ast.FunctionDef, attr: str) -> bool:
    return any(
        isinstance(node, ast.Attribute) and node.attr == attr
        for node in ast.walk(func)
    )


def _has_name(tree: ast.AST, name: str) -> bool:
    return any(isinstance(node, ast.Name) and node.id == name
               for node in ast.walk(tree))


def _has_imported(tree: ast.AST, name: str) -> bool:
    """True if `name` is imported anywhere in the module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == name or a.name.startswith(name + ".")
                   for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == name for a in node.names):
                return True
    return False


# ═══════════════════════════════════════════════════════════════════
# gate audit consumes the shared EpisodeTaskSampler factory
# ═══════════════════════════════════════════════════════════════════

def test_gate_audit_constructs_shared_sampler():
    """run_reset_gate_audit builds the sampler with the SAME factory call
    the train loop uses (EpisodeTaskSampler(task_specs, targets,
    base_seed=...)) and samples roles only through next_attempt()."""
    module = _parse("diffusion_policy_3d/rl/dppo_uav/gate_audit.py")
    fn = _function(module, "run_reset_gate_audit")
    assert _has_call(fn, "EpisodeTaskSampler")
    assert _has_attr_call(fn, "next_attempt")


def test_gate_audit_has_no_independent_role_rng():
    """No independently seeded role RNG/cursor in the gate audit path —
    role selection is the sampler's job (guide: '禁止 G3 内独立写 role
    RNG/cursor')."""
    module = _parse("diffusion_policy_3d/rl/dppo_uav/gate_audit.py")
    fn = _function(module, "run_reset_gate_audit")
    for rng_name in ("default_rng", "RandomState", "randint", "random",
                     "rng"):
        assert not _has_name(fn, rng_name), f"independent {rng_name} in gate path"
    assert not _has_imported(fn, "random")


# ═══════════════════════════════════════════════════════════════════
# train loop consumes the INJECTED sampler, never rebuilds one
# ═══════════════════════════════════════════════════════════════════

def test_train_loop_injects_sampler_without_rebuilding():
    """run_e2e_batch takes task_sampler as an argument and samples through
    it; it never constructs its own EpisodeTaskSampler."""
    module = _parse("diffusion_policy_3d/rl/dppo_uav/train_loop.py")
    fn = _function(module, "run_e2e_batch")
    arg_names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "task_sampler" in arg_names
    assert not _has_call(fn, "EpisodeTaskSampler")
    assert _has_attr_call(fn, "next_attempt")


# ═══════════════════════════════════════════════════════════════════
# G3 CLI never samples roles itself; production paths by default
# ═══════════════════════════════════════════════════════════════════

def test_g3_cli_never_samples_roles_independently():
    """The CLI's only admission-path entry is run_reset_gate_audit (which
    uses the shared sampler); the CLI itself builds no sampler and owns no
    role RNG/cursor."""
    module = _parse("scripts/g3_10k_reset_gate_audit.py")
    assert not _has_call(module, "EpisodeTaskSampler")
    assert not _has_name(module, "next_attempt")
    for rng_name in ("default_rng", "RandomState", "randint", "random",
                     "rng"):
        assert not _has_name(module, rng_name)
    assert _has_call(module, "run_reset_gate_audit")


def test_g3_cli_defaults_to_production_paths():
    """Guide Step 2 runs the CLI with ONLY --n-resets/--seed/--output, so
    --manifest and --worlds-dir must default to the production paths via
    the DPPO_MANIFEST / DPPO_WORLDS_TRAIN env-var convention."""
    module = _parse("scripts/g3_10k_reset_gate_audit.py")
    source = open(G3_CLI, "r", encoding="utf-8").read()
    assert "DPPO_MANIFEST" in source
    assert "DPPO_WORLDS_TRAIN" in source
    # run()'s keyword-only manifest/worlds_dir must have defaults (no longer
    # required args) and the env-var fallback must be applied in run().
    fn = _function(module, "run")
    kw_names = [a.arg for a in fn.args.kwonlyargs]
    assert "manifest" in kw_names and "worlds_dir" in kw_names
    kw_defaults = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert kw_defaults["manifest"] is not None
    assert kw_defaults["worlds_dir"] is not None
    assert _has_attr_call(fn, "get")  # os.environ.get fallback in run()
