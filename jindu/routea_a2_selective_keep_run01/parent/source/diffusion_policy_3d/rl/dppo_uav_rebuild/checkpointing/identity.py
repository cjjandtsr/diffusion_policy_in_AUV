"""Checkpoint identity comparison before tensor deserialization."""


class IdentityMismatch(RuntimeError):
    pass


def verify_checkpoint_identity(header, runtime_identity, lora_identity):
    if header.runtime_identity_hash != runtime_identity.content_hash:
        raise IdentityMismatch("runtime identity mismatch")
    if header.lora_identity_hash != lora_identity.content_hash:
        raise IdentityMismatch("LoRA identity mismatch")

