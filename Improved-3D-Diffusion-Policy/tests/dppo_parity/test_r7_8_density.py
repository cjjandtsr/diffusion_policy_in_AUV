"""R7-8 Step 6: chunked offline-density build from the frozen EGO zarr
dataset (user ruling 2026-08-24 — real density inside R7-8, never the
synthetic placeholder; R7-5 EXIT=137 lesson: the 4.8 GB dataset must be
read in chunks, never materialized at once).

The builder emits the reward_v13 schema with "x,y,z" cell keys — the
exact schema reward13_density_to_capability bridges into the capability
OfflineDensity, so the two sides must agree on every cell.
"""

import numpy as np
import pytest
import zarr

from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
    build_density_from_zarr,
)
from diffusion_policy_3d.rl.dppo_uav.r7_bank_builder import (
    reward13_density_to_capability,
)

# 13 state columns per the EGO dataset convention (pos 0-2, vel 3-5,
# speed 6, yaw cos/sin 7-8, rest unused).
_N_COLS = 13


def _zarr_dataset(tmp_path, n_rows=10, episode_ends=(5, 10)):
    """A tiny frozen-EGO-shaped zarr dataset: state (N, 13) + ends."""
    path = tmp_path / "mini_ego.zarr"
    root = zarr.open(str(path), mode="w")
    rng = np.random.RandomState(0)
    state = rng.uniform(-8.0, 8.0, size=(n_rows, _N_COLS)).astype(
        np.float32)
    state[:, 2] = 2.0  # flight altitude
    root.create_array("data/state", data=state)
    root.create_array("meta/episode_ends",
                      data=np.asarray(episode_ends, dtype=np.int64))
    return str(path)


def _full_reference(path, bin_size=(1.0, 0.5, 0.5)):
    """The unfrozen full-load reference: goals broadcast per episode,
    one compute_support_cell call per row (the old tools semantics)."""
    from diffusion_policy_3d.env.reward_v13 import (
        compute_support_cell,
    )
    import zarr as _z

    root = _z.open(path, mode="r")
    state = root["data/state"][:]
    episode_ends = root["meta/episode_ends"][:]
    positions = state[:, 0:3]
    goals = []
    prev_end = 0
    for ep_end in episode_ends:
        if 0 < int(ep_end) <= len(positions):
            goals.append(positions[int(ep_end) - 1])
        prev_end = int(ep_end)
    counts = {}
    for i, pos in enumerate(positions):
        goal = goals[min(_episode_of(i, episode_ends), len(goals) - 1)]
        cell = compute_support_cell(pos, goal, bin_size)
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def _episode_of(row_idx, episode_ends):
    """Episode index for a row (searchsorted side='right' semantics)."""
    import numpy as _np

    ends = _np.asarray(episode_ends, dtype=int)
    idx = _np.searchsorted(ends, row_idx, side="right")
    return int(min(idx, len(ends) - 1))


# ═══════════════════════════════════════════════════════════════════
# chunked build — correctness and the R7-8 schema contract
# ═══════════════════════════════════════════════════════════════════

def test_build_density_from_zarr_matches_full_reference(tmp_path):
    """The chunked builder's cell counts equal the unfrozen full-load
    computation on the same tiny dataset."""
    path = _zarr_dataset(tmp_path)
    density = build_density_from_zarr(path, chunk_rows=3)
    assert density["total"] == 10
    reference = _full_reference(path)
    assert {tuple(map(int, k.split(","))): v
            for k, v in density["cell_counts"].items()} == reference


def test_build_density_chunk_size_invariant(tmp_path):
    """Different chunk sizes give byte-identical cell counts."""
    path = _zarr_dataset(tmp_path)
    a = build_density_from_zarr(path, chunk_rows=2)
    b = build_density_from_zarr(path, chunk_rows=100)
    assert a == b


def test_build_density_schema_is_reward13_capability_compatible(tmp_path):
    """The emitted cell_counts keys are literal "x,y,z" strings — the
    exact schema reward13_density_to_capability bridges (the old tools
    tuple-repr "(x, y, z)" keys were incompatible and are never
    emitted)."""
    path = _zarr_dataset(tmp_path)
    density = build_density_from_zarr(path, chunk_rows=2)
    for key in density["cell_counts"]:
        parts = key.split(",")
        assert len(parts) == 3
        for part in parts:
            int(part)  # must parse — ValueError otherwise
    capability = reward13_density_to_capability(density)
    assert capability.counts  # non-empty, validated
    assert capability.bin_size == (1.0, 0.5, 0.5)


def test_build_density_missing_dataset_rejected(tmp_path):
    """A missing dataset path is a hard error — never an empty density."""
    with pytest.raises(ValueError, match="not found"):
        build_density_from_zarr(str(tmp_path / "nope.zarr"))


def test_build_density_missing_keys_rejected(tmp_path):
    """A zarr without data/state or meta/episode_ends is a hard error."""
    path = tmp_path / "bad.zarr"
    root = zarr.open(str(path), mode="w")
    root.create_array("data/state", data=np.zeros((4, _N_COLS),
                                                  dtype=np.float32))
    with pytest.raises(ValueError, match="episode_ends"):
        build_density_from_zarr(str(path))
