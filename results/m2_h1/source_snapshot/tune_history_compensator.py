"""Development on pilot seeds only. Keeps every candidate and real episode."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from run_sensor_episode import configure_threads, run_episode
from run_h1_baseline import sha
from check_sensor_age import audit_episode


CANDIDATES = {'h40_w3': (40, 3), 'h80_w3': (80, 3), 'h80_w5': (80, 5), 'h120_w5': (120, 5)}
CONDITIONS = ['J20', 'J40', 'J80', 'I20', 'I40', 'I80']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/m2_h1'))
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    plan = {'status': 'exploratory_development_not_closed_test', 'seeds': list(range(10)),
            'conditions': CONDITIONS, 'candidates': CANDIDATES, 'ranking': ['falls', '-mean_alive_s', 'velocity_rmse'],
            'known_old_pilot_results': True, 'closed_test_seeds_used': False,
            'note': 'Same pilot seeds used to propose and select candidates; all outcomes retained.',
            'source_sha256': {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('history_compensator.py')]}}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    configure_threads(1)
    source = Path('results/pilot_h1')
    validations = []
    for method in ('B1', 'M', 'M2'):
        folder = root/'validation'/method
        run_episode(folder, source/'validation/clean.json', source/'validation/initial.json', method, (80, 80))
        validations.append(audit_episode(folder))
    with np.load(root/'validation/B1/signals.npz') as base:
        for method in ('M', 'M2'):
            with np.load(root/'validation'/method/'signals.npz') as other:
                for key in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
                    np.testing.assert_array_equal(base[key], other[key])
    (root/'validation/checks.json').write_text(json.dumps(validations, indent=2)+'\n')
    all_results, audits, table = [], [], {}
    started = time.perf_counter()
    for name, (horizon, window) in CANDIDATES.items():
        outcomes = []
        for seed in range(10):
            for condition in CONDITIONS:
                folder = root/'episodes'/name/f'seed_{seed}'/condition
                summary = run_episode(folder, source/f'inputs/seed_{seed}/{condition}.json',
                                      source/f'inputs/seed_{seed}/initial.json', 'M2', (horizon, horizon),
                                      adapter_config={'window': window})
                audits.append(audit_episode(folder))
                summary['candidate'] = name
                outcomes.append(summary)
                all_results.append(summary)
            print(json.dumps({'candidate': name, 'seed_done': seed, 'completed': len(all_results),
                              'elapsed_s': round(time.perf_counter()-started, 2),
                              'falls_so_far': sum(s['fallen'] for s in outcomes)}), flush=True)
        table[name] = {'n': len(outcomes), 'falls': sum(s['fallen'] for s in outcomes),
                       'mean_alive_s': float(np.mean([s['time_alive_s'] for s in outcomes])),
                       'velocity_rmse': float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in outcomes])),
                       'walking_successes': sum(s['walking_success'] for s in outcomes)}
    selected = min(table, key=lambda k: (table[k]['falls'], -table[k]['mean_alive_s'], table[k]['velocity_rmse']))
    (root/'audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    (root/'selection.json').write_text(json.dumps({'candidates': table, 'selected': selected,
                                                 'independent_test': False}, indent=2)+'\n')
    print(json.dumps({'candidates': table, 'selected': selected}, indent=2), flush=True)


if __name__ == '__main__':
    main()
