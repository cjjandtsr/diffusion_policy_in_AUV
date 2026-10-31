"""Immutable episode task facts and order-independent seed identity."""

from dataclasses import dataclass
import hashlib
import json
import re

from .terminal_contract import TransitionFacts


EVAL_ONLY_WORLD_IDS = frozenset(("w4", "w8", "w11", "w12", "w18"))
TRAINABLE_ROLES = (
    "anticipatory", "dense_recovery", "support_expansion", "yaw_lateral_recovery"
)
TRAINING_ROLES = frozenset(TRAINABLE_ROLES)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_task_identity(entry, obstacle_sha256: str) -> str:
    """Hash only the reviewed semantic task fields, never manifest position."""

    if not _SHA256.fullmatch(str(obstacle_sha256)):
        raise ValueError("obstacle_sha256 must be lowercase sha256")
    task_id = entry.get("task_id") or entry.get("admission_id")
    world_id = entry.get("world_id") or entry.get("world_uid")
    payload = {
        "task_id": task_id,
        "world_id": world_id,
        "split": entry.get("split"),
        "role": entry.get("role") or entry.get("role_active"),
        "obstacle_sha256": obstacle_sha256,
        "start_goal_spec": entry.get("start_goal_spec", {}),
        "admission_schema_version": entry.get(
            "admission_schema_version", entry.get("schema_version")
        ),
    }
    if any(payload[key] in (None, "") for key in (
        "task_id", "world_id", "split", "role", "admission_schema_version"
    )):
        raise ValueError("task identity fields must be explicit")
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("task identity must be canonical finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def derive_task_seed(run_seed: int, task_identity_hash: str) -> int:
    if isinstance(run_seed, bool) or int(run_seed) < 0:
        raise ValueError("run_seed must be a non-negative integer")
    if not _SHA256.fullmatch(str(task_identity_hash)):
        raise ValueError("task_identity_hash must be lowercase sha256")
    preimage = str(int(run_seed)).encode("ascii") + b"\x00" + bytes.fromhex(task_identity_hash)
    digest = hashlib.sha256(preimage).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


@dataclass(frozen=True)
class EpisodeTaskSpec:
    task_id: str
    world_id: str
    obstacle_json: str
    role: str
    seed: int
    training_eligible: bool
    historical_eval_only: bool = False
    identity_hash: str = ""
    seed_preimage: str = ""
    seed_digest: str = ""

    def __post_init__(self):
        if not self.task_id or not self.obstacle_json or self.seed < 0:
            raise ValueError("task_id, obstacle_json and non-negative seed are required")
        if self.role not in TRAINING_ROLES:
            raise ValueError("unknown Route A training role: {}".format(self.role))
        forbidden = self.historical_eval_only or self.world_id in EVAL_ONLY_WORLD_IDS
        if self.training_eligible and forbidden:
            raise ValueError("eval-only world cannot be training eligible")
        for name in ("identity_hash", "seed_digest"):
            value = getattr(self, name)
            if value and not _SHA256.fullmatch(value):
                raise ValueError("{} must be lowercase sha256".format(name))
        if self.seed_preimage and not self.seed_preimage.startswith("hex:"):
            raise ValueError("seed_preimage must be an exact hex-encoded byte preimage")

    @property
    def sampling_identity(self):
        if self.identity_hash:
            return self.identity_hash
        payload = json.dumps(
            {
                "task_id": self.task_id,
                "world_id": self.world_id,
                "role": self.role,
                "seed": self.seed,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
