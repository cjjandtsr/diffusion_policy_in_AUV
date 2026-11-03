"""F5 mixture proportions, exact restart and frozen input identity."""
from collections import Counter
import copy
import hashlib
import json

import pytest

from diffusion_policy_3d.rl.dppo_uav_rebuild.environments import task_sampler as module
from diffusion_policy_3d.rl.dppo_uav_rebuild.environments.task_spec import EpisodeTaskSpec, TRAINABLE_ROLES


def fixture_bank(tmp_path):
    tasks, rows, worlds = [], [], {}
    for i, role in enumerate(TRAINABLE_ROLES):
        world = 'train_' + role
        path = tmp_path / (world + '.json')
        path.write_text('{}', encoding='utf-8')
        n = 25 if i % 2 else 4
        worlds[world] = dict(role=role, obstacle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             n_obstacles=n)
        tasks.append(EpisodeTaskSpec(world, world, str(path), role, i + 1, True))
        for tier, kind in [('L0', 'random'), ('L1', 'random'), ('L2', 'random'),
                           ('L3', 'near_obstacle_start'), ('L4', 'random')]:
            rows.append(dict(task_id=world+tier, world=world, role=role, tier=tier, kind=kind,
                             start=[-2., 0., 2.], goal=[2., 0., 2.], start_yaw=0.,
                             n_world_obstacles=n))
    bank = dict(schema='routea_training_task_bank_v1', worlds=worlds, tasks=rows)
    path = tmp_path / 'bank.json'
    path.write_text(json.dumps(bank), encoding='utf-8')
    return tasks, path, hashlib.sha256(path.read_bytes()).hexdigest()


def sampler(tmp_path, seed=91):
    assert hasattr(module, 'TaskBankMixtureSampler'), 'F5 sampler is not implemented'
    return module.TaskBankMixtureSampler(*fixture_bank(tmp_path), seed=seed,
                                        mixture=[.15,.10,.15,.30,.20,.10])


def test_f5_proportions_and_original_sequence(tmp_path):
    s = sampler(tmp_path)
    original = module.RoleBalancedTaskSampler(s.tasks, seed=91)
    counts, density, bank_counts = Counter(), Counter(), Counter()
    for _ in range(100000):
        task = s.sample()
        meta = s.last_metadata
        source = meta['source']
        counts[source] += 1
        if source == 'original':
            assert task == original.sample()
            assert s.last_reset_options is None
        else:
            bank_counts[source] += 1
            density[source] += meta['world_class'] == 'dense'
            assert s.last_reset_options['fixed_start_goal']['start_pos'] == [-2.,0.,2.]
    for name, fraction in zip(s.SOURCES, [.15,.10,.15,.30,.20,.10]):
        assert abs(counts[name]/100000 - fraction) < .005
    for source, n in bank_counts.items():
        assert abs(density[source]/n - (.5 if source in ('L0','L1') else .8)) < .02


def test_f5_exact_resume_and_changed_bank_rejected(tmp_path):
    s = sampler(tmp_path)
    for _ in range(19):
        s.sample()
    state = s.state_dict()
    expected = []
    for _ in range(100):
        expected.append((s.sample().task_id, copy.deepcopy(s.last_metadata),
                         copy.deepcopy(s.last_reset_options)))
    restored = sampler(tmp_path)
    restored.load_state_dict(state)
    actual = []
    for _ in range(100):
        actual.append((restored.sample().task_id, copy.deepcopy(restored.last_metadata),
                       copy.deepcopy(restored.last_reset_options)))
    assert actual == expected
    state['bank_sha256'] = '0'*64
    with pytest.raises(ValueError, match='identity'):
        restored.load_state_dict(state)


def test_f5_uses_nonempty_worlds_and_rejects_unknown_world(tmp_path):
    s = sampler(tmp_path)
    tasks, path, _ = fixture_bank(tmp_path)
    bank = json.loads(path.read_text())
    # L4 legitimately has fewer worlds; neither source nor class probability changes.
    bank['tasks'] = [r for r in bank['tasks'] if not (
        r['tier'] == 'L4' and r['world'] in (tasks[2].world_id,tasks[3].world_id))]
    path.write_text(json.dumps(bank))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    s = module.TaskBankMixtureSampler(tasks,path,digest,91,[.15,.10,.15,.30,.20,.10])
    assert len(s.worlds_by_source['L4']['dense']) == 1
    assert len(s.worlds_by_source['L4']['sparse']) == 1
    bank['tasks'][0]['world'] = 'w4'
    path.write_text(json.dumps(bank))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='world'):
        module.TaskBankMixtureSampler(tasks,path,digest,91,[.15,.10,.15,.30,.20,.10])

