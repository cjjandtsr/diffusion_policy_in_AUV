"""Read-only F6 deployment check; exit 0 for PASS, 1 for FAIL.

Usage: python -I f6_base_closure_check.py F5_prepared.json F6_manifest.json BASE [ROOT]
Run outside the frozen training source tree.
"""
import hashlib
import json
from pathlib import Path
import sys


def main():
    prepared, manifest, base = map(Path, sys.argv[1:4])
    root = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    f5_rows = json.loads(prepared.read_text(encoding='utf-8'))['files']
    overlay_rows = json.loads(manifest.read_text(encoding='utf-8'))['source_files']
    f5 = {r['relative_path']: r['sha256'] for r in f5_rows}
    overlay = {r['relative_path']: r['sha256'] for r in overlay_rows}
    failures = []
    for label, actual, expected in (
            ('F5 rows', len(f5_rows), 143), ('F5 unique paths', len(f5), 143),
            ('overlay rows', len(overlay_rows), 28), ('overlay unique paths', len(overlay), 28),
            ('overlap', len(f5.keys() & overlay.keys()), 12)):
        if actual != expected:
            failures.append('%s: expected %s, actual %s' % (label, expected, actual))

    def check(label, directory, expected):
        matched = 0
        for name, wanted in expected.items():
            path = directory / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual == wanted:
                matched += 1
            else:
                failures.append('%s %s: expected %s, actual %s' % (label, name, wanted, actual))
        print('%s: %d/%d match' % (label, matched, len(expected)))

    check('BASE vs F5 frozen closure', base, f5)
    if root is not None:
        check('ROOT vs F5 with overlay', root, {n: overlay.get(n, h) for n, h in f5.items()})
        check('ROOT overlay', root, overlay)
    for failure in failures:
        print('FAIL:', failure)
    print('PASS' if not failures else 'FAIL (%d differences)' % len(failures))
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
