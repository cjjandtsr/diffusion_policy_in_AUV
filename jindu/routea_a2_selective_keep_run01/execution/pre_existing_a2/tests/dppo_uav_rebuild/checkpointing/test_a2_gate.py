import pytest
import torch
from scripts.run_routea_a2 import assert_equal_state


def test_continue_requires_pass_and_bound_checkpoint(tmp_path):
    import json
    from scripts.run_routea_a2 import continuation_checkpoint, digest
    checkpoint=tmp_path/'checkpoint_update_000129'; checkpoint.mkdir()
    (checkpoint/'state.pt').write_bytes(b'original')
    config=tmp_path/'C.yaml'; config.write_text('config')
    receipt=tmp_path/'gate.json'
    payload=dict(status='PASS',checkpoints={'C':dict(path=str(checkpoint),sha256=digest(checkpoint/'state.pt'),config_sha256=digest(config))})
    receipt.write_text(json.dumps(payload))
    assert continuation_checkpoint(receipt,'C',config)==checkpoint
    (checkpoint/'state.pt').write_bytes(b'changed')
    with pytest.raises(ValueError): continuation_checkpoint(receipt,'C',config)
    payload['status']='FAIL'; receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError): continuation_checkpoint(receipt,'C',config)


def test_continuity_gate_rejects_small_adam_or_rng_difference():
    a={'adam':torch.tensor([.2]),'rng':torch.tensor([1,2],dtype=torch.uint8)}
    assert_equal_state(a,{'adam':a['adam'].clone(),'rng':a['rng'].clone()})
    with pytest.raises(ValueError):
        assert_equal_state(a,{'adam':torch.tensor([.200001]),'rng':a['rng']})
    with pytest.raises(ValueError):
        assert_equal_state(a,{'adam':a['adam'],'rng':torch.tensor([1,3],dtype=torch.uint8)})


def test_keep_rng_and_calibration_survive_real_checkpoint(tmp_path):
    from types import SimpleNamespace
    from tests.dppo_uav_rebuild.checkpointing.test_checkpoint_roundtrip import _components
    from tests.dppo_uav_rebuild.engine.test_selective_keep import bank
    from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.selective_keep import SelectiveKeep
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.save import save_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.restore import restore_checkpoint
    from diffusion_policy_3d.rl.dppo_uav_rebuild.checkpointing.schema import CheckpointHeader, SCHEMA_VERSION
    lora,critic,ao,co,counters=_components()
    keep=SelectiveKeep(bank(),coefficient=.123,seed=4)
    torch.rand(7,generator=keep.generator)
    kwargs=dict(lora_bundle=lora,critic=critic,actor_optimizer=ao,critic_optimizer=co,
                counters=counters,schedules={'selective_keep':keep})
    path=tmp_path/'checkpoint'
    save_checkpoint(path,header=CheckpointHeader(SCHEMA_VERSION,{},'a'*64,'b'*64,129),**kwargs)
    expected=torch.rand(9,generator=keep.generator)
    keep.coefficient=99
    restore_checkpoint(path,runtime_identity=SimpleNamespace(content_hash='a'*64),
                       lora_identity=SimpleNamespace(content_hash='b'*64),**kwargs)
    assert keep.coefficient==.123
    assert torch.equal(expected,torch.rand(9,generator=keep.generator))
