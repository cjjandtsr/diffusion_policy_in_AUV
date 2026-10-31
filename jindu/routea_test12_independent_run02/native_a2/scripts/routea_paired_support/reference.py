"""Reuse the accepted frozen bank and validate each compared raw trajectory."""
from pathlib import Path
import re

import numpy as np

from .bank import check_bank
from .core import canonical_hash, file_hash, read_json
from .episodes import action_seed


BANK_HASH = "863bf1fb2a41c991b73cc92493aa7d8732c711b4e48c6a8d063bd2dd8c59996a"
CATALOGUE_FILE = "delivery/scripts/routea_paired_support/policy_catalog_v1.json"
_SHA = re.compile(r"^[0-9a-f]{64}$")
COMMON_PROTOCOL_SOURCES = {
    'diffusion_policy_3d/env/kinematic_uav_env.py': '8f840e471d47b00f8b13f1a946a1aa6c3447aac7908cb3c3f6fc144544c2629a',
    'diffusion_policy_3d/env/reward_v2_support/reward.py': 'b6d914ac7a935f75fde42ed80aa1e82911caa20fba2d8b2f2d4868b762728efc',
    'diffusion_policy_3d/rl/dppo_uav_rebuild/rewards/v2/facade.py': '9096a0d4ffec10165a09f4d8fc5fcdca06cdd3690243835a425e42c61395fb2c',
    'diffusion_policy_3d/env/reward_v2_support/environment.py': '65518bf7ba4e9a5017f061b53d38a521b8f956787a15c81b66ec9dfe8abbf77b',
    'diffusion_policy_3d/env/reward_v2_support/first_event.py': '2f979d885d1feef5acd5f1c7818d624bfff637bb0981053192a989df90d0bc15',
}


def source_compatible(name, current, original, allow_space_expansion):
    return current == original or (allow_space_expansion
        and COMMON_PROTOCOL_SOURCES.get(name) == current)


def check_reference(root, reference, allow_space_expansion=False):
    """Check frozen inputs; explicitly allow the evaluation space-limit change."""
    root, reference = Path(root), Path(reference)
    bank = check_bank(reference / "bank/bank.json")
    if bank["content_hash"] != BANK_HASH:
        raise ValueError("reference bank differs from accepted frozen task bank")
    cat = read_json(reference / CATALOGUE_FILE)
    if (cat["content_hash"] != bank["catalogue_hash"]
            or canonical_hash({k: v for k, v in cat.items() if k != "content_hash"}) != cat["content_hash"]):
        raise ValueError("reference catalogue differs from accepted frozen bank")
    critical = set(cat["actor_compatibility_paths"]) | {
        "diffusion_policy_3d/env/kinematic_uav_env.py",
        "diffusion_policy_3d/env/collision_geom.py",
        "diffusion_policy_3d/env/traj_utils.py",
        "diffusion_policy_3d/model/common/synth_pointcloud.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/environments/kinematic_adapter.py",
        "diffusion_policy_3d/rl/dppo_uav_rebuild/rewards/v2/facade.py",
    }
    critical.update(name for name in bank["evaluation_sources"]
                    if name.startswith("diffusion_policy_3d/env/reward_v2_support/"))
    for name in sorted(critical):
        if not source_compatible(name, file_hash(root / name),
                                 bank["evaluation_sources"][name], allow_space_expansion):
            raise ValueError("reference actor/observation/physics source changed: " + name)
    return bank


def check_episode_identity(trace, entry, task, world, bank, cat, name, baseline):
    """Bind a load_commit result to the task and original BC actor observation."""
    from diffusion_policy_3d.env.traj_utils import action_to_world, POS

    header, rows, summary = trace
    descriptor = cat["policies"][name]
    use_base = descriptor["use_base_policy"]
    identity = (cat["common"]["bc_identity_hash"] if use_base
                else descriptor["runtime_identity_hash"])
    checks = {
        "bank": header["bank_hash"] == bank["content_hash"],
        "task": header["task"] == task["payload"],
        "pair": header["initial_hash"] == entry["initial_hash"],
        "reset_seed": header["reset_seed"] == entry["reset_seed"],
        "sampling_seeds": all(r["action_seed"] == action_seed(entry["task_id"], entry["repeat"], i)
                              for i, r in enumerate(rows)),
        "policy": header["policy"] == name and summary["policy"] == name,
        "branch": header["use_base_policy"] is use_base,
        "policy_identity": header["policy_identity_hash"] == identity,
        "reward_version": header["reward_version"] == "v2",
        "space_termination": header.get("terminate_at_space_limits", True)
                             == cat.get("terminate_at_space_limits", True)
                             == summary.get("terminate_at_space_limits", True),
        "geometry": header["geometry"] == world["geometry"]
                    and header["obstacle_sha256"] == world["obstacle_sha256"],
        "key": summary["task_id"] == entry["task_id"] and summary["repeat"] == entry["repeat"],
        "summary_pair": summary["initial_hash"] == entry["initial_hash"]
                        and summary["bank_hash"] == bank["content_hash"],
        "header_key": header["task_id"] == entry["task_id"] and header["repeat"] == entry["repeat"],
        "role": summary["role"] == task["role"] and summary["world_id"] == task["world_id"]
                and header["role"] == task["role"] and header["world_id"] == task["world_id"],
        "frames": header["position_frame"] == "world"
                  and header["policy_position_frame"] == "odom = world - spawn_offset"
                  and header["action_dtype"] == "float32",
        "spawn_offset": header["spawn_offset"] == world["spawn_offset"],
        "initial_metadata": header["initial_pointcloud_metadata"] == entry["initial_pointcloud_metadata"],
    }
    if not rows or not all(checks.values()):
        raise ValueError("raw header/pair mismatch: " + str(checks))
    first = rows[0]
    pre = first["pre_state"]
    if (not np.array_equal(np.asarray(pre["position"], np.float32),
                           np.asarray(task["payload"]["start_pos"], np.float32))
            or not np.array_equal(np.asarray(pre["velocity"], np.float32), np.zeros(3, np.float32))
            or abs(pre["yaw"] - task["payload"]["start_yaw"]) > 1e-10):
        raise ValueError("initial physical reset differs from frozen task")
    provenance = first["observation_provenance"]
    meta = entry["initial_pointcloud_metadata"]
    if (baseline[0]["initial_hash"] != entry["initial_hash"]
            or provenance != baseline[1][0]["observation_provenance"]
            or first["pointcloud_metadata_before"] != meta
            or provenance["raw_frame_hash"] != meta[-1]["raw_array_hash"]
            or provenance["episode_identity_hash"] != meta[-1]["episode_identity_hash"]
            or provenance["decision_index"] != meta[-1]["decision_index"]):
        raise ValueError("initial actor observation/provenance differs from original BC")
    for row in rows:
        proof = row["observation_provenance"]
        before = row["pointcloud_metadata_before"]
        if (not before or proof["normalizer_identity_hash"] != provenance["normalizer_identity_hash"]
                or proof["episode_identity_hash"] != provenance["episode_identity_hash"]
                or proof["raw_frame_hash"] != before[-1]["raw_array_hash"]
                or proof["decision_index"] != before[-1]["decision_index"]
                or any(not _SHA.fullmatch(str(proof[k])) for k in (
                    "raw_frame_hash", "raw_history_hash", "normalizer_identity_hash",
                    "normalized_history_hash", "policy_input_hash", "episode_identity_hash"))):
            raise ValueError("actor-call observation provenance mismatch")
        state = row["pre_state"]
        position = np.asarray(state["position"], np.float32)
        waypoints = action_to_world(np.asarray(row["policy_action_8x11"], np.float32), position,
                                    np.asarray(state["velocity"], np.float32), state["yaw"])
        planned = np.vstack((position, waypoints[:, POS]))
        saved = np.asarray(row["planned_policy_path"], np.float32)
        if saved.shape != (9, 3) or not np.allclose(planned, saved, atol=1e-6, rtol=0.):
            raise ValueError("saved H8 planned path disagrees with actual policy action")
