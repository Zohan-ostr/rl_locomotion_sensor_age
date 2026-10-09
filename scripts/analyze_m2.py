"""Summarize real paired development episodes against the preserved pilot."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.output
    selection = json.loads((root/'selection.json').read_text())
    chosen = selection['selected']
    records = []
    for folder in sorted((root/'episodes'/chosen).glob('seed_*/*')):
        candidate = json.loads((folder/'summary.json').read_text())
        seed, cond = candidate['seed'], candidate['condition']['id']
        reference = Path(f'results/pilot_h1/episodes/seed_{seed}/{cond}')
        cases = {'M2': candidate}
        for method, suffix in [('B1', 'H0_0'), ('M', 'H20_20')]:
            old = json.loads((reference/f'{method}_{suffix}/summary.json').read_text())
            for key in ('schedule_sha256', 'initial_state_sha256', 'checkpoint_sha256', 'scene_sha256',
                        'config_sha256', 'model_asset_sha256', 'upstream_commit', 'command', 'joint_order',
                        'fall_rule', 'pilot_config_sha256', 'torch', 'mujoco', 'numpy'):
                assert old[key] == candidate[key], (seed, cond, key)
            cases[method] = old
        for method, summary in cases.items():
            records.append({'seed': seed, 'condition': cond, 'method': method,
                            'fallen': summary['fallen'], 'alive_s': summary['time_alive_s'],
                            'velocity_rmse': summary['velocity_rmse_after_warmup_before_fall_m_s'],
                            'success': summary['walking_success'],
                            'joint_error_rad': summary['joint_prediction_rmse_rad'],
                            'gravity_error_rad': summary['gravity_prediction_mean_angle_rad']})
    assert len(records) == 180
    groups = {}
    for cond in ['ALL', 'J20', 'J40', 'J80', 'I20', 'I40', 'I80']:
        groups[cond] = {}
        for method in ('B1', 'M', 'M2'):
            values = [r for r in records if r['method'] == method and (cond == 'ALL' or r['condition'] == cond)]
            groups[cond][method] = {'n': len(values), 'falls': sum(r['fallen'] for r in values),
                                   'mean_alive_s': float(np.mean([r['alive_s'] for r in values])),
                                   'velocity_rmse': float(np.mean([r['velocity_rmse'] for r in values])),
                                   'walking_successes': sum(r['success'] for r in values)}
    differences = {}
    for reference in ('B1', 'M'):
        per_seed = []
        for seed in range(10):
            subset = [r for r in records if r['seed'] == seed]
            per_seed.append((sum(r['fallen'] for r in subset if r['method'] == reference)-
                             sum(r['fallen'] for r in subset if r['method'] == 'M2'))/6)
        rng = np.random.default_rng(20261009)
        bootstrap = np.mean(rng.choice(per_seed, size=(5000, 10), replace=True), axis=1)
        differences[reference+'-M2'] = {'mean': float(np.mean(per_seed)),
                                        'descriptive_95_interval': np.quantile(bootstrap, [.025, .975]).tolist()}
    summary = {'selected': chosen, 'groups': groups, 'paired_fall_differences': differences,
               'independent_test': False, 'closed_test_seeds_used': False,
               'hypothesis_supported': None, 'comparisons_from_preserved_pilot': True}
    (root/'comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    (root/'paired_records.json').write_text(json.dumps(records, indent=2)+'\n')
    lines = ['| Условие | B1 падения | M падения | M2 падения | B1/M/M2 RMSE, м/с |',
             '|---|---:|---:|---:|---|']
    for condition, values in groups.items():
        lines.append('| '+condition+' | '+' | '.join(f"{values[m]['falls']}/{values[m]['n']}" for m in ('B1', 'M', 'M2'))+
                     ' | '+' / '.join(f"{values[m]['velocity_rmse']:.3f}" for m in ('B1', 'M', 'M2'))+' |')
    (root/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
