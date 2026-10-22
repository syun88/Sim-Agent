"""Refresh a repaired external package's model inventory and checksums.

No robot assets are copied into this repository. With --prune, remove only OBJ
collision files absent from the repaired inventory, after checking the package's
original recoverable backup exists. Run cross-format validation separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--prune', action='store_true')
    args = parser.parse_args()
    folder = args.folder.resolve()
    geometry = json.loads((folder / 'source/collision_repair_geometry.json').read_text())
    model_path = folder / 'source/model.json'
    model = json.loads(model_path.read_text())
    assert {link['name'] for link in model['links']} == set(geometry)
    names = {name for entries in geometry.values() for name in entries}
    directories = ['genesis', 'urdf', 'mujoco', 'gazebo_jetty/models/scorpion']
    for directory in directories:
        for name in names:
            if not (folder / directory / name).is_file():
                raise ValueError(f'Missing repaired collision file: {directory}/{name}')
    inventory_path = folder/'source/collision_inventory.json'
    previous = json.loads(inventory_path.read_text()) if inventory_path.is_file() else {}
    removed = previous.get('removed_unused_collision_files', [])
    if args.prune:
        backup = folder.parent / 'old_no_use/for_sim_collision_before_20261021.zip'
        with zipfile.ZipFile(backup) as archive:
            if archive.testzip() is not None:
                raise ValueError('Original backup failed its CRC check')
            if 'genesis/scorpion.urdf' not in archive.namelist():
                raise ValueError('Original package backup does not contain the source model')
        for directory in directories:
            for path in (folder / directory / 'meshes/collision').glob('*.obj'):
                if path.relative_to(folder / directory).as_posix() not in names:
                    removed.append(str(path.relative_to(folder)))
                    path.unlink()
    for link in model['links']:
        link['collision'] = geometry[link['name']]
    model_path.write_text(json.dumps(model, indent=2) + '\n')
    report = {'collision_pieces': len(names), 'removed_unused_collision_files': removed,
              'scope': 'Inventory and file integrity only; no simulator runtime claim'}
    inventory_path.write_text(json.dumps(report, indent=2) + '\n')
    # This file deliberately excludes itself; a digest cannot include its own bytes.
    checksum = folder / 'SHA256SUMS'
    paths = sorted(path for path in folder.rglob('*') if path.is_file()
                   and path != checksum and '__pycache__' not in path.parts
                   and not path.name.startswith('.'))
    checksum.write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  '
                               f'{path.relative_to(folder).as_posix()}\n' for path in paths))
    print(json.dumps({'collision_pieces': len(names), 'removed_unused_files': len(removed),
                      'checksummed_files': len(paths)}))


if __name__ == '__main__':
    main()
