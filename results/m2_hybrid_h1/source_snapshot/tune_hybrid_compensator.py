import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from run_sensor_episode import configure_threads, run_episode
from check_sensor_age import audit_episode


def main():
    root = Path('results/m2_hybrid_h1')
    root.mkdir(parents=True, exist_ok=False)
    sources = []
    for path in ('results/m2_h1', 'results/m2_periodic_h1', 'results/m2_followup_h1'):
        for candidate in json.loads((Path(path)/'selection.json').read_text())['candidates']:
            sources.append({'name': candidate, 'path': str(Path(path)/'episodes'/candidate), 'legacy': False})
    sources.append({'name': 'legacy_M20', 'path': 'results/pilot_h1/episodes', 'legacy': True})
    selected, table = {}, {}
    def folder(source, seed, condition):
        path = Path(source['path'])/f'seed_{seed}'/condition
        return path/'M_H20_20' if source['legacy'] else path
    for stream, prefix in [('joints', 'J'), ('imu', 'I')]:
        ranking = []
        for source in sources:
            values = [json.loads((folder(source, seed, f'{prefix}{d}')/'summary.json').read_text())
                      for seed in range(10) for d in (20, 40, 80)]
            ranking.append({**source, 'falls': sum(v['fallen'] for v in values),
                            'mean_alive_s': float(np.mean([v['time_alive_s'] for v in values])),
                            'velocity_rmse': float(np.mean([v['velocity_rmse_after_warmup_before_fall_m_s'] for v in values]))})
        selected[stream] = min(ranking, key=lambda v: (v['falls'], -v['mean_alive_s'], v['velocity_rmse']))
        table[stream] = ranking
    config, horizons = {'family': 'hybrid'}, []
    for stream, prefix in [('joints', 'J'), ('imu', 'I')]:
        source = selected[stream]
        sample = json.loads((folder(source, 0, prefix+'20')/'summary.json').read_text())
        settings = dict(sample.get('adapter_config') or {})
        family = settings.pop('family', 'legacy' if source['legacy'] else 'acceleration')
        config[stream] = {'family': family, 'config': settings}
        horizons.append(sample['horizon_'+stream+'_ms'])
    plan = {'selected_components': selected, 'component_candidates': table, 'adapter_config': config,
            'horizons_ms': horizons, 'test_seeds_used': False,
            'selection_rule': 'per stream: falls then mean alive then velocity RMSE; original M may be retained',
            'scope': 'pilot E2 seeds 0–9; exact trace equivalence to component sources checked'}
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    configure_threads(1)
    results, audits = [], []
    base = Path('results/pilot_h1/inputs')
    for seed in range(10):
        for condition in ('J20', 'J40', 'J80', 'I20', 'I40', 'I80'):
            output = root/'episodes'/'hybrid'/f'seed_{seed}'/condition
            summary = run_episode(output, base/f'seed_{seed}/{condition}.json', base/f'seed_{seed}/initial.json',
                                  'M2', tuple(horizons), adapter_config=config)
            audits.append(audit_episode(output))
            component = selected['joints' if condition.startswith('J') else 'imu']
            with np.load(output/'signals.npz') as a, np.load(folder(component, seed, condition)/'signals.npz') as b:
                for key in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
                    np.testing.assert_array_equal(a[key], b[key])
            results.append(summary)
        print(json.dumps({'seed_done': seed, 'falls_so_far': sum(v['fallen'] for v in results)}), flush=True)
    metrics = {'n': len(results), 'falls': sum(v['fallen'] for v in results),
               'mean_alive_s': float(np.mean([v['time_alive_s'] for v in results])),
               'velocity_rmse': float(np.mean([v['velocity_rmse_after_warmup_before_fall_m_s'] for v in results])),
               'walking_successes': sum(v['walking_success'] for v in results)}
    (root/'selection.json').write_text(json.dumps({'candidates': {'hybrid': metrics}, 'selected': 'hybrid', 'independent_test': False}, indent=2)+'\n')
    (root/'audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    print(json.dumps({'plan': plan, 'result': metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
