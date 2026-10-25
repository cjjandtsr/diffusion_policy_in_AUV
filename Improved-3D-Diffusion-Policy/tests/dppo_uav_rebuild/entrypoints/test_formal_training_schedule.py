from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints.pipeline import (
    _actor_lr_for_update,
    _prepare_resume_output,
    _run_optimizer_attempt,
    _should_save_checkpoint,
    _update_outer_failure_streaks,
    _validate_run_config,
)
from diffusion_policy_3d.rl.dppo_uav_rebuild.entrypoints import pipeline
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.update_transaction import (
    UpdateRejected,
)


def _config(**run_overrides):
    run = {
        "updates": 256,
        "rollout_transitions": 1024,
        "minibatch_size": 128,
        "update_epochs": 5,
        "critic_warmup_updates": 1,
        "checkpoint_every_updates": 8,
    }
    run.update(run_overrides)
    return SimpleNamespace(
        run=SimpleNamespace(**run),
        optimizer=SimpleNamespace(actor_lr=2e-5, actor_lr_end=1e-5),
    )


def test_formal_schedule_accepts_1024_rollout_and_128_minibatches():
    _validate_run_config(_config())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rollout_transitions": 1000}, "divisible"),
        ({"minibatch_size": 64}, "128"),
        ({"critic_warmup_updates": 256}, "smaller than updates"),
        ({"checkpoint_every_updates": 0}, "positive"),
    ],
)
def test_formal_schedule_rejects_uninterpretable_run_shapes(overrides, message):
    with pytest.raises(ValueError, match=message):
        _validate_run_config(_config(**overrides))


def test_actor_lr_starts_after_warmup_and_reaches_requested_endpoint():
    config = _config()

    assert _actor_lr_for_update(config, 1) == pytest.approx(2e-5)
    assert _actor_lr_for_update(config, 2) == pytest.approx(2e-5)
    assert _actor_lr_for_update(config, 129) == pytest.approx(1.5e-5)
    assert _actor_lr_for_update(config, 256) == pytest.approx(1e-5)


def test_fresh_384_run_uses_the_whole_budget_for_actor_lr_decay():
    config = _config(updates=384, critic_warmup_updates=0)
    assert _actor_lr_for_update(config, 1) == pytest.approx(2e-5)
    assert _actor_lr_for_update(config, 256) == pytest.approx(
        2e-5 - 255 / 383 * 1e-5)
    assert _actor_lr_for_update(config, 384) == pytest.approx(1e-5)


def test_split_run_places_its_writer_lock_in_results(tmp_path, monkeypatch):
    config = _config(seed=202609203)
    config.output_dir = str(tmp_path / "weights")
    config.result_dir = str(tmp_path / "program" / "run")

    def stop_before_model_loading(_config, _root):
        assert (Path(config.result_dir) / "run.lock").is_file()
        assert not (Path(config.output_dir) / "run.lock").exists()
        raise RuntimeError("test stop before model loading")

    monkeypatch.setattr(pipeline, "build_system", stop_before_model_loading)
    with pytest.raises(RuntimeError, match="test stop before model loading"):
        pipeline.run(config, tmp_path)
    assert not (Path(config.result_dir) / "run.lock").exists()


def test_split_resume_keeps_weights_and_writes_move_report_in_results(tmp_path):
    weights = tmp_path / "weights"
    results = tmp_path / "program" / "run"
    weights.mkdir()
    results.mkdir(parents=True)
    checkpoint = weights / "checkpoint_update_000016"
    checkpoint.mkdir()
    (checkpoint / "state.pt").write_bytes(b"preserved weight")
    quarantine = _prepare_resume_output(weights, 9, report_dir=results)
    assert (quarantine / checkpoint.name / "state.pt").read_bytes() == b"preserved weight"
    assert not list(weights.rglob("resume_quarantine.json"))
    assert len(list(results.glob("resume_quarantine_after_*_weights.json"))) == 1


def test_checkpoint_schedule_keeps_cadence_and_always_saves_final_update():
    config = _config(updates=258)

    assert not _should_save_checkpoint(config, 7)
    assert _should_save_checkpoint(config, 8)
    assert not _should_save_checkpoint(config, 257)
    assert _should_save_checkpoint(config, 258)


def test_operational_stop_at_32_keeps_configured_256_update_identity():
    """A pause changes the executed span, never the frozen training budget."""

    config = _config()
    stop = getattr(pipeline, "_execution_last_update", None)
    assert callable(stop)
    assert stop(config, start_update=1, stop_after_update=32) == 32
    assert stop(config, start_update=33, stop_after_update=None) == 256
    assert config.run.updates == 256
    with pytest.raises(ValueError, match="checkpoint"):
        stop(config, start_update=1, stop_after_update=31)
    with pytest.raises(ValueError, match="before"):
        stop(config, start_update=33, stop_after_update=32)


def test_training_cli_passes_stop_without_editing_frozen_config(monkeypatch):
    config = _config()
    observed = {}
    monkeypatch.setattr(pipeline.OmegaConf, "load", lambda _path: config)
    monkeypatch.setattr(
        pipeline, "run",
        lambda actual, _root, **kwargs: observed.update(
            config=actual, **kwargs
        ),
    )

    assert pipeline.main(["--config", "unused.yaml", "--stop-after-update", "32"]) == 0
    assert observed == {
        "config": config,
        "resume_from": None,
        "stop_after_update": 32,
    }
    assert config.run.updates == 256


def test_shipped_formal_config_has_one_full_critic_warmup_outer_update():
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(
        root
        / "diffusion_policy_3d"
        / "rl"
        / "dppo_uav_rebuild"
        / "config"
        / "train_e5_dppo_formal_seed1_v1.yaml"
    )

    _validate_run_config(config)
    minibatches_per_outer = (
        config.run.rollout_transitions // config.run.minibatch_size
    )

    assert config.run.updates * config.run.rollout_transitions == 262144
    assert minibatches_per_outer == 8
    assert (
        config.run.critic_warmup_updates
        * config.run.update_epochs
        * minibatches_per_outer
        == 40
    )


def test_resume_quarantines_only_artifacts_newer_than_restored_checkpoint(tmp_path):
    (tmp_path / "metrics_update_000007.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metrics_update_000009.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint_update_000008").mkdir()
    (tmp_path / "checkpoint_update_000010").mkdir()
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    quarantine = _prepare_resume_output(tmp_path, start_update=9)

    assert quarantine is not None
    assert (tmp_path / "metrics_update_000007.json").exists()
    assert (tmp_path / "checkpoint_update_000008").exists()
    assert (tmp_path / "notes.txt").exists()
    assert not (tmp_path / "metrics_update_000009.json").exists()
    assert not (tmp_path / "checkpoint_update_000010").exists()
    assert (quarantine / "metrics_update_000009.json").exists()
    assert (quarantine / "checkpoint_update_000010").exists()
    assert (quarantine / "resume_quarantine.json").exists()


def test_recoverable_minibatch_rejection_is_recorded_without_stopping_outer_update():
    class Trainer:
        def update(self, batch, *, update_actor):
            raise UpdateRejected(
                "transient KL",
                reason="approx_kl_gate",
                recoverable=True,
            )

    evidence = _run_optimizer_attempt(Trainer(), object(), update_actor=True)

    assert evidence["status"] == "rejected"
    assert evidence["reason"] == "approx_kl_gate"


def test_nonrecoverable_minibatch_failure_still_stops_immediately():
    class Trainer:
        def update(self, batch, *, update_actor):
            raise UpdateRejected("NaN", reason="nonfinite_loss")

    with pytest.raises(UpdateRejected, match="NaN"):
        _run_optimizer_attempt(Trainer(), object(), update_actor=True)


def test_critic_warmup_does_not_advance_after_a_rejected_minibatch():
    class Trainer:
        def update(self, batch, *, update_actor):
            raise UpdateRejected(
                "zero critic gradient",
                reason="zero_critic_gradient",
                recoverable=True,
            )

    with pytest.raises(UpdateRejected, match="zero critic gradient"):
        _run_optimizer_attempt(
            Trainer(),
            object(),
            update_actor=False,
            allow_recoverable_rejection=False,
        )


def test_outer_failure_requires_five_consecutive_joint_updates():
    counters = SimpleNamespace(
        consecutive_all_rejected_outer_updates=0,
        consecutive_zero_actor_delta_outer_updates=0,
    )

    for _ in range(4):
        assert (
            _update_outer_failure_streaks(
                counters,
                update_actor=True,
                actor_delta=0.0,
                committed=True,
            )
            is None
        )
    assert (
        _update_outer_failure_streaks(
            counters,
            update_actor=True,
            actor_delta=0.0,
            committed=True,
        )
        == "five_consecutive_zero_actor_delta_outer_updates"
    )
