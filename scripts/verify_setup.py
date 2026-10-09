"""Verify CPU runtime and all pinned H1 assets without modifying the checkout."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    upstream = root / 'vendor/unitree_rl_gym'
    reference = json.loads((root / 'results/baseline_h1/headless20/summary.json').read_text())
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != reference['upstream_commit']:
        raise RuntimeError('Upstream commit mismatch')
    hashes = dict(reference['model_asset_sha256'])
    hashes.update({
        'deploy/pre_train/h1/motion.pt': reference['checkpoint_sha256'],
        'deploy/deploy_mujoco/configs/h1.yaml': reference['config_sha256'],
    })
    for name, expected in hashes.items():
        path = upstream / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Asset hash mismatch: {name}; check Git LFS and pinned checkout')
    if not (upstream / 'LICENSE').is_file():
        raise RuntimeError('Missing upstream LICENSE')
    versions = {}
    for line in (root / 'requirements.txt').read_text().splitlines():
        if '==' not in line or line.startswith('#'):
            continue
        name, expected = line.split('==')
        versions[name] = importlib.metadata.version(name)
        if versions[name] != expected:
            raise RuntimeError(f'{name}: installed {versions[name]}, expected {expected}')
    import torch
    import mujoco
    import glfw
    import yaml
    import matplotlib
    import imageio
    import imageio_ffmpeg
    if torch.version.cuda is not None:
        raise RuntimeError('Expected CPU-only PyTorch')
    result = {'python': sys.version, 'platform': platform.platform(),
              'upstream_commit': commit, 'verified_asset_count': len(hashes),
              'versions': versions, 'torch_cuda_build': torch.version.cuda,
              'gui_tested': False}
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
