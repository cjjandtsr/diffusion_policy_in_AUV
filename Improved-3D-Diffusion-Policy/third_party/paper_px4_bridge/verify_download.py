"""Verify upstream git blobs; normalize connector-added final newline once."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent / 'px4_avoidance_c386852'
manifest = json.loads((root / 'DOWNLOAD_MANIFEST.json').read_text(encoding='utf-8'))


def blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


count = 0
for entry in manifest['files']:
    if not entry['downloaded']:
        continue
    path = root / entry['path']
    data = path.read_bytes()
    if blob_sha(data) != entry['git_blob_sha']:
        assert blob_sha(data[:-1]) == entry['git_blob_sha'], str(path)
        path.write_bytes(data[:-1])
    assert blob_sha(path.read_bytes()) == entry['git_blob_sha'], str(path)
    count += 1
print(f'{count} downloaded official files match upstream git blob SHA1')
