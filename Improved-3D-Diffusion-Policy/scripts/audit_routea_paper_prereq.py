"""Read-only run03 checks and H10 evidence tables; never approve worlds or load a policy.

Geometry normalization follows Claude's testext_world_check.py in cjjdelunwen,
paper-v2-restructure, blob ae001a713342385e8349a595678fefa4d68e3904.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def geometry_key(value):
    fields = ('x', 'y', 'z', 'radius', 'height', 'size_x', 'size_y', 'size_z', 'yaw')
    return frozenset(tuple([o['shape']] + [round(float(o.get(k, 0.)), 2) for k in fields])
                     for o in value['obstacles'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, out = Path(args.run).resolve(), Path(args.output).resolve()
    if out == run or run in out.parents:
        raise ValueError('output must be outside immutable return directory')
    evidence = run / 'evidence'
    archive = run / (run.name + '.tar.gz')
    expected = (run / (archive.name + '.sha256')).read_text(encoding='utf-8').split()[0]
    if digest(archive) != expected:
        raise ValueError('archive SHA differs')
    tar_files, evidence_members = 0, set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            if not member.isfile():
                continue
            relative = Path(member.name).relative_to(run.name)
            target = (run / relative).resolve()
            target.relative_to(run)
            archived = hashlib.sha256(stream.extractfile(member).read()).hexdigest()
            if archived != digest(target):
                raise ValueError('unpacked file differs: ' + str(relative))
            tar_files += 1
            if relative.parts[0] == 'evidence':
                if target in evidence_members:
                    raise ValueError('duplicate archive entry')
                evidence_members.add(target)
    actual = {p.resolve() for p in evidence.rglob('*') if p.is_file()}
    if evidence_members != actual:
        raise ValueError('archive evidence set differs from unpacked evidence')
    sums = read(evidence / 'SHA256.json')
    for name, expected_sha in sums.items():
        target = (evidence / name).resolve()
        target.relative_to(evidence)
        if digest(target) != expected_sha:
            raise ValueError('evidence checksum differs: ' + name)
    if {(evidence / name).resolve() for name in sums} | {evidence / 'SHA256.json'} != actual:
        raise ValueError('evidence manifest coverage differs')
    if int((run / 'inventory.rc').read_text().strip()) != 0:
        raise ValueError('inventory rc nonzero')
    inventory = read(evidence / 'inventory.json')
    checkpoint_rows = []
    source_roots = list(inventory['native_sources'])
    for name, item in inventory['checkpoints'].items():
        folder = evidence / 'checkpoints' / name
        header, declared, committed = (read(folder / n) for n in ('header.json', 'SHA256.json', 'COMMITTED.json'))
        for filename in ('header.json', 'SHA256.json', 'COMMITTED.json'):
            if digest(folder / filename) != item['files'][filename]['sha256']:
                raise ValueError('checkpoint metadata differs: ' + name)
        if (declared != committed['files'] or declared['header.json'] != digest(folder / 'header.json')
                or declared['state.pt'] != item['files']['state.pt']['sha256']
                or committed['status'] != 'committed'
                or canonical_hash(header['runtime_identity']) != header['runtime_identity_hash']):
            raise ValueError('checkpoint commitment or identity differs: ' + name)
        suffix = 'native_a1' if name == 'P3' else 'native_a2' if name == 'P6_u192' else None
        source = next(p for p in source_roots if p.endswith(suffix)) if suffix else inventory['repo']
        checkpoint_rows.append(dict(policy=name, update=header['update_index'], checkpoint=item['path'],
            source_root=source, state_sha256=declared['state.pt'], runtime_identity_hash=header['runtime_identity_hash'],
            header_sha256=declared['header.json'], source_set_hash=header['runtime_identity']['source_set_hash'],
            status='metadata_verified_load_unverified'))
    histories = []
    for item in inventory['copied_evidence']:
        path = evidence / item['copy']
        if digest(path) != item['sha256']:
            raise ValueError('history differs')
        fixture = 'test_tmp' in item['source'] or '/fixtures/' in item['source']
        histories.append(dict(item, fixture=fixture, text=path.read_text(encoding='utf-8')))
    worlds = []
    for item in inventory['world_assets']:
        path = evidence / item['copy']
        if digest(path) != item['sha256']:
            raise ValueError('world asset differs')
        value = read(path)
        worlds.append(dict(item, world_id=value['world_id'], geometry=geometry_key(value)))
    bank = next(read(evidence / h['copy']) for h in histories
                if h['source'].endswith('/routea_test12_independent_run02/bank/bank.json'))
    used_test12 = set(bank['worlds'])
    reference = [w for w in worlds if w['split'] in ('train', 'val')]
    findings = []
    for world in (w for w in worlds if w['split'] == 'test'):
        shape = world['geometry']
        exact = [dict(split=r['split'], world_id=r['world_id']) for r in reference if shape == r['geometry']]
        hits = [dict(source=h['source'], copy=h['copy'], sha256=h['sha256'])
                for h in histories if not h['fixture'] and world['world_id'] in h['text']]
        findings.append(dict(world_id=world['world_id'], source=world['source'], copy=world['copy'],
            sha256=world['sha256'], in_test12=world['world_id'] in used_test12,
            geometry_equal_train_or_val=exact, history_text_mentions=hits,
            review_candidate=world['world_id'] not in used_test12 and not exact,
            history_scope='textual world-id mentions in exported non-fixture files only; not proof of never used'))
    out.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (out / name).open('x', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    write('H3_checkpoint_metadata.json', dict(models=checkpoint_rows, source_roots=source_roots,
        note='source_root candidates follow historical provenance; actual strict native loading is still required.'))
    write('H10_world_evidence.json', dict(worlds=findings, approval=None,
        geometry_rule='Claude canon: obstacle set, shape + coordinates/dimensions/yaw rounded to 2 decimals; names ignored',
        historical_records=[{k: v for k, v in h.items() if k != 'text'} for h in histories],
        note='No approved_worlds.json generated; user/Claude reviews historical coverage and decides.'))
    write('verification.json', dict(status='passed_export_integrity_only', archive_sha256=digest(archive),
        tar_files_compared=tar_files, evidence_files=len(actual), evidence_checksums=len(sums),
        inventory_sha256=digest(evidence / 'inventory.json'), sha_manifest_sha256=digest(evidence / 'SHA256.json'),
        worlds_by_split=dict(Counter(w['split'] for w in worlds)), historical_records=len(histories),
        fixture_records=sum(h['fixture'] for h in histories),
        review_candidates=sum(w['review_candidate'] for w in findings), script_sha256=digest(__file__)))
    print('PREREQ_INTEGRITY_PASS', len(actual), 'REVIEW_CANDIDATES', sum(w['review_candidate'] for w in findings))


if __name__ == '__main__':
    main()
