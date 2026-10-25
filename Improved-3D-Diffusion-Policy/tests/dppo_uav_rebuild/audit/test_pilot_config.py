from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = ROOT / "diffusion_policy_3d" / "rl" / "dppo_uav_rebuild" / "config"


def test_pilot10_changes_only_budget_seed_label_and_output():
    smoke = OmegaConf.to_container(
        OmegaConf.load(CONFIG_ROOT / "train_e5_dppo_v1.yaml"), resolve=True
    )
    pilot = OmegaConf.to_container(
        OmegaConf.load(CONFIG_ROOT / "train_e5_dppo_pilot10_v1.yaml"), resolve=True
    )

    assert pilot["run"]["updates"] == 10
    assert pilot["run"]["rollout_transitions"] == 128
    assert pilot["run"]["update_epochs"] == 1
    assert pilot["output_dir"] == (
        "data/outputs/dppo_uav_rebuild/e5_dppo_pilot10_v1_seed20260827"
    )
    assert pilot["checkpoint"] == smoke["checkpoint"]
    assert pilot["vendor_root"] == smoke["vendor_root"]
    assert pilot["admission_manifest"] == smoke["admission_manifest"]
    assert pilot["worlds_train"] == smoke["worlds_train"]
    assert pilot["lora"] == smoke["lora"]
    assert pilot["critic"] == smoke["critic"]
    assert pilot["optimizer"] == smoke["optimizer"]
    assert pilot["dppo"] == smoke["dppo"]
    assert pilot["gates"] == smoke["gates"]
    assert pilot["environment"] == smoke["environment"]
    assert pilot["reward"] == smoke["reward"]
