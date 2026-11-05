"""Apply the exact compatibility patches to their pinned installed sources."""

import hashlib
import json
from pathlib import Path
import subprocess

CONTEXT = Path(__file__).resolve().parent
PACKAGES = Path('/usr/local/lib/python3.12/dist-packages')


def verify(manifest, phase):
    for name, hashes in manifest.items():
        actual = hashlib.sha256((PACKAGES / name).read_bytes()).hexdigest()
        if actual != hashes[phase + '_sha256']:
            raise ValueError(f'{phase} source differs: {name}')


def main():
    manifest = json.loads((CONTEXT / 'source-hashes.json').read_text())
    verify(manifest, 'original')
    for patch in sorted(CONTEXT.glob('*.patch')):
        with patch.open('rb') as source:
            subprocess.run(['patch', '-p1', '--batch', '--forward',
                            '--no-backup-if-mismatch', '--fuzz=0'],
                           stdin=source, cwd='/', check=True)
    verify(manifest, 'patched')


if __name__ == '__main__':
    main()
