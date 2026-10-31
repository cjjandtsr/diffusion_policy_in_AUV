"""Source identity for the reused project Reward v1.3 implementation."""

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path


@dataclass(frozen=True)
class RewardIdentity:
    source_path: str
    source_sha256: str
    implementation: str = "diffusion_policy_3d.env.reward_v13"


def identify_reward_module(module) -> RewardIdentity:
    path = Path(inspect.getfile(module)).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return RewardIdentity(str(path), digest)

