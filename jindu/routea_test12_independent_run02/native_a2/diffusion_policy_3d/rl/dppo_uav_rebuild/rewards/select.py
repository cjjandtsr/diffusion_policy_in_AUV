"""Select an explicit reward version before loading a GPU policy."""

import hashlib
import json
from pathlib import Path

from .v13.facade import RewardV13Facade


def select_reward(config, repo_root):
    version = str(getattr(config, "reward_version", "v13"))
    if version == "v13":
        return RewardV13Facade()
    if version == "v2":
        from .v2.facade import RewardV2Facade
        protocol = config.get("task_protocol")
        return RewardV2Facade(space_cost_budget=None if protocol is None else protocol.space_cost_budget)
    if version != "v17":
        raise ValueError("unsupported reward_version: {}".format(version))
    from .v17.facade import RewardV17Facade
    from omegaconf import OmegaConf
    from diffusion_policy_3d.env.reward_v17 import resolve_config
    from diffusion_policy_3d.env.reward_v17_support.density import validate_density
    resolved = resolve_config(OmegaConf.to_container(config.reward, resolve=True))
    if resolved["gamma_pbrs"] != float(config.dppo.gamma_environment):
        raise ValueError("PBRS gamma must equal environment/GAE gamma")
    if int(config.environment.max_episode_steps) != resolved["max_episode_steps"]:
        raise ValueError("reward calibration episode horizon mismatch")
    expected_sigma = .3 if bool(config.environment.disturb_enable) else 0.
    if resolved["sigma_z"] != expected_sigma:
        raise ValueError("height free-band scale disagrees with actual factory execution noise")
    # Freeze files by bytes as well as semantic contents. No implicit density load.
    artifacts = {}
    for name in ("reward_calibration", "reward_density"):
        entry = getattr(config, name, None)
        if entry is None:
            raise ValueError("v17 requires {} provenance".format(name))
        path = Path(repo_root) / str(entry.path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(entry.sha256):
            raise ValueError("{} SHA256 mismatch".format(name))
        artifacts[name] = json.loads(payload.decode("utf-8"))
    calibration, density = artifacts["reward_calibration"], validate_density(artifacts["reward_density"])
    if calibration.get("status") != "resolved" or calibration.get("selected_config") != resolved:
        raise ValueError("v17 requires resolved calibration matching every reward field")
    if calibration.get("density_hash") != density["content_hash"]:
        raise ValueError("calibration/density identity mismatch")
    facade = RewardV17Facade()
    if calibration.get("reward_source_sha256") != facade.identity.source_sha256:
        raise ValueError("reward source changed after offline calibration")
    facade.resolved_config, facade.offline_density = resolved, density
    return facade
