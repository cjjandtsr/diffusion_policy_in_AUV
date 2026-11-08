"""Resolve the approved O-prime freeze into plans for the shared E2 evaluator."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_routea_f3_resume import read, write, digest, canonical


def bind_bank(name, bank, index, report):
    assert canonical({k:v for k,v in bank.items() if k!='content_hash'}) == bank['content_hash']
    components = {'added'} if name == 'added' else ({'test12'} if name == 'test' else {'retained_run02','added'})
    approved = [e for e in index['conditions'] if e['component'] in components]
    identity = lambda e: (e['key'],e['task_id'],e['repeat'],e['initial_hash'])
    assert len(bank['episodes']) == len(approved)
    assert {identity(e) for e in bank['episodes']} == {identity(e) for e in approved}
    expected_hash = (report['incremental_hash'] if name == 'added' else
                     index['test_bank_hash'] if name == 'test' else index['extension_v2_hash'])
    assert bank['content_hash'] == expected_hash
    if name == 'added':
        assert {e['source_bank_hash'] for e in approved} == {bank['content_hash']}


def prepare(args):
    setup = read(args.setup)
    freeze = Path(args.freeze)
    report = read(freeze/'verification.json')
    index = read(freeze/'O_prime_index.json')
    assert canonical({k:v for k,v in index.items() if k!='content_hash'}) == index['content_hash']
    assert report['O_prime_index_hash'] == index['content_hash']
    assert report['O_prime_episodes'] == 995 and report['task_categories'] == dict(zero=59,low=70,high=70)
    review = read(args.review)
    assert review['reviewer'] == 'Claude' and review['classification_and_isolation_approved'] is True
    assert review['O_prime_index_hash'] == index['content_hash'], 'approval is for another freeze'
    existing = args.mode == 'existing'
    paths = dict(added=freeze/'incremental/bank.json') if existing else dict(
        test=Path(setup['test_bank']), ext=freeze/'bank/bank.json')
    banks = {}
    for name, path in paths.items():
        bank = read(path)
        bind_bank(name, bank, index, report)
        banks[name] = dict(paths=[str(path.resolve())], split=bank['split'],
            worlds=setup['worlds'], episodes=len(bank['episodes']),
            content_hash=bank['content_hash'], file_sha256=digest(path))
    if existing:
        models = copy.deepcopy(setup['models'])
    else:
        completion = read(Path(setup['stage2_run'])/'training_complete.json')
        assert completion['status'] == 'passed' and completion['updates'] == 128
        models = [completion['model']]
        assert models[0]['policy'] == 'P15' and models[0]['update'] == 256
    plan = dict(schema='routea_native_e2_plan_v1', label='FINAL', repo=setup['repo'],
        source_root=setup['source_root'], source_files=setup['source_files'],
        scripts=setup['scripts'], banks=banks, models=models,
        expected_episodes=sum(b['episodes'] for b in banks.values())*len(models),
        O_prime_index_hash=index['content_hash'], review_sha256=digest(args.review),
        old_conditions_reused=existing, policy_aliases={'bc':'P0'})
    write(args.output, plan)
    print('FINAL_EVAL_PLAN_PASS', args.mode, plan['expected_episodes'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setup', required=True)
    p.add_argument('--freeze', required=True)
    p.add_argument('--review', required=True, help='Claude approval for the actual frozen index hash')
    p.add_argument('--mode', choices=['existing','P15'], required=True)
    p.add_argument('--output', required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
