"""Public checkpoint schema."""

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Dict


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CheckpointHeader:
    schema_version: int
    runtime_identity: Dict[str, Any]
    runtime_identity_hash: str
    lora_identity_hash: str
    update_index: int

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint schema")
        for name in ("runtime_identity_hash", "lora_identity_hash"):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError("{} must be lowercase SHA256".format(name))
        if self.update_index < 0:
            raise ValueError("update_index must be non-negative")

    def to_json(self):
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text):
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate checkpoint header key: {}".format(key))
                value[key] = item
            return value
        return cls(**json.loads(text, object_pairs_hook=unique))

    @property
    def schema_hash(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()
