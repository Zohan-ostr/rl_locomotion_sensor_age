"""Predeclared follow-up after acceleration/Fourier development outcomes."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from run_sensor_episode import configure_threads, run_episode
from check_sensor_age import audit_episode
from run_h1_baseline import sha


CANDIDATES = {
    'cycle_g1': {'horizon': 120, 'config': {'family': 'cycle', 'gain': 1.}},
    'cycle_g05': {'horizon': 120, 'config': {'family': 'cycle', 'gain': .5}},
    'confidence_j025': {'horizon': 20, 'config': {'family': 'conservative', 'joint_gain': .25}},
    'confidence_j0': {'horizon': 20, 'config': {'family': 'conservative', 'joint_gain': 0.}},
}


def main():
    root = Path('results/m2_followup_h1')
    root.mkdir(parents=True, exist_ok=False)
    conditions = ['J20', 'J40', 'J80', 'I20', 'I40', 'I80']
    (root/'plan.json').write_text(json.dumps({'status': 'exploratory_followup_after_negative_candidates',
        'candidates': CANDIDATES, 'conditions': conditions, 'seeds': list(range(10)),
        'closed_test_seeds_used': False, 'ranking': ['falls', '-mean_alive_s', 'velocity_rmse'],
        'source_sha256': sha(Path(__file__))}, indent=2)+'\n')
    configure_threads(1)
    source = Path('results/pilot_h1')
    table, audits = {}, []
    started = time.perf_counter()
    for name, settings in CANDIDATES.items():
        outcomes = []
        h = settings['horizon']
        for seed in range(10):
            for condition in conditions:
                folder = root/'episodes'/name/f'seed_{seed}'/condition
                summary = run_episode(folder, source/f'inputs/seed_{seed}/{condition}.json',
                                      source/f'inputs/seed_{seed}/initial.json', 'M2', (h, h),
                                      adapter_config=settings['config'])
                audits.append(audit_episode(folder))
                outcomes.append(summary)
            print(json.dumps({'candidate': name, 'seed_done': seed,
                              'elapsed_s': round(time.perf_counter()-started, 2),
                              'falls_so_far': sum(s['fallen'] for s in outcomes)}), flush=True)
        table[name] = {'n': len(outcomes), 'falls': sum(s['fallen'] for s in outcomes),
                      'mean_alive_s': float(np.mean([s['time_alive_s'] for s in outcomes])),
                      'velocity_rmse': float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in outcomes])),
                      'walking_successes': sum(s['walking_success'] for s in outcomes)}
    selected = min(table, key=lambda k: (table[k]['falls'], -table[k]['mean_alive_s'], table[k]['velocity_rmse']))
    (root/'selection.json').write_text(json.dumps({'candidates': table, 'selected': selected, 'independent_test': False}, indent=2)+'\n')
    (root/'audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    print(json.dumps({'candidates': table, 'selected': selected}, indent=2), flush=True)


if __name__ == '__main__':
    main()
