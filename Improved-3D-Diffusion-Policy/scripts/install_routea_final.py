"""Install the exact final delivery; preserve overwritten launcher/input helpers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True)
    args=p.parse_args()
    bundle=Path(args.bundle).resolve()
    manifest=json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    root=Path(manifest['wsl_root']).resolve()
    evidence=Path('/home/cjj/AUV/program/results/routea_final_install_run01')
    assert not evidence.exists(), 'preserve previous installation evidence'
    assert root.is_dir()
    for relative,expected in manifest['files'].items():
        src=(bundle/'payload'/relative).resolve()
        src.relative_to(bundle/'payload')
        assert digest(src)==expected
    for relative,allowed in manifest['preconditions'].items():
        assert digest(root/relative) in allowed, 'existing input/helper differs: '+relative
    for relative in manifest['files']:
        target=(root/relative).resolve()
        target.relative_to(root)
        if relative.startswith('native_versions/routea_final_a2_v1/'):
            assert not target.exists(), 'new native tree must not overwrite an existing run source'
    evidence.mkdir(parents=True)
    before={}
    for relative,expected in manifest['files'].items():
        target=root/relative
        if target.exists():
            before[relative]=digest(target)
            backup=evidence/'before'/relative
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,backup)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(bundle/'payload'/relative,target)
        assert digest(target)==expected
    (evidence/'installation.json').write_text(json.dumps(dict(status='DELIVERY_INSTALLED',
        files=manifest['files'],before=before,manifest_sha256=digest(bundle/'manifest.json')),
        indent=2,sort_keys=True)+'\n',encoding='utf-8')
    shutil.copy2(bundle/'manifest.json',evidence/'manifest.json')
    print('FINAL_DELIVERY_INSTALLED',len(manifest['files']))


if __name__=='__main__':
    main()
