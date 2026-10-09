"""Freeze the selected development profile, then evaluate without retuning."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from run_sensor_episode import configure_threads, run_episode
from run_h1_baseline import sha
from check_sensor_age import audit_episode
from sensor_age import make_schedule


ROOTS = ['results/m2_h1', 'results/m2_periodic_h1', 'results/m2_followup_h1', 'results/m2_hybrid_h1']
CONDITIONS = [{'id': f'{prefix}{d}', f'delay_{stream}_ms': d}
              for prefix, stream in [('J', 'joints'), ('I', 'imu')] for d in (20, 40, 80)]


def main():
    output = Path('results/m2_validation_h1')
    output.mkdir(parents=True, exist_ok=False)
    candidates = []
    for root in ROOTS:
        table = json.loads((Path(root)/'selection.json').read_text())['candidates']
        candidates += [{'root': root, 'candidate': name, **summary} for name, summary in table.items()]
    best = min(candidates, key=lambda c: (c['falls'], -c['mean_alive_s'], c['velocity_rmse']))
    chosen_root = Path(best['root'])
    manifest = json.loads(next((chosen_root/'episodes'/best['candidate']).glob('seed_*/*/summary.json')).read_text())
    config = manifest['adapter_config'].copy()
    config.setdefault('family', 'acceleration')
    horizons = (manifest['horizon_joints_ms'], manifest['horizon_imu_ms'])
    profile = {'status': 'development_candidate_frozen_before_additional_validation',
               'selection': best, 'adapter_config': config,
               'horizon_joints_ms': horizons[0], 'horizon_imu_ms': horizons[1],
               'closed_test_seeds_used': False, 'not_validated_yet': True}
    (output/'frozen_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    plan = {'profile': profile, 'validation_seeds': list(range(10, 20)),
            'conditions': CONDITIONS, 'methods': ['B1', 'M', 'M2'],
            'no_retuning_on_validation': True, 'closed_test_seeds_100_139_used': False,
            'extra_pilot_checks': 'clean + E3 drops 1/2/4/6 each stream + jitter, seeds 0–9, selected M2 only'}
    (output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    model = mujoco.MjModel.from_xml_path('vendor/unitree_rl_gym/resources/robots/h1/scene.xml')
    for seed in range(10, 20):
        folder = output/f'inputs/seed_{seed}'
        folder.mkdir(parents=True)
        delta = np.random.default_rng(np.random.SeedSequence([seed, 20261009])).uniform(-.003, .003, 10)
        qpos = model.qpos0.copy()
        qpos[7:] += delta
        (folder/'initial.json').write_text(json.dumps({'seed': seed, 'qpos': qpos.tolist(),
                 'qvel': np.zeros(model.nv).tolist(), 'joint_delta_rad': delta.tolist()}, indent=2)+'\n')
        for condition in CONDITIONS:
            (folder/(condition['id']+'.json')).write_text(json.dumps(make_schedule(seed, condition), indent=2)+'\n')
    configure_threads(1)
    audits, outcomes = [], []
    started = time.perf_counter()
    for seed in range(10, 20):
        for condition in CONDITIONS:
            folder = output/f'inputs/seed_{seed}'
            reference = None
            for method in ('B1', 'M', 'M2'):
                result_folder = output/'episodes'/f'seed_{seed}'/condition['id']/method
                summary = run_episode(result_folder, folder/(condition['id']+'.json'), folder/'initial.json',
                                      method, horizons if method == 'M2' else (20, 20),
                                      adapter_config=config if method == 'M2' else None)
                audits.append(audit_episode(result_folder))
                if reference:
                    for key in ('initial_state_sha256', 'schedule_sha256', 'checkpoint_sha256', 'config_sha256', 'model_asset_sha256', 'command', 'fall_rule'):
                        assert summary[key] == reference[key]
                else:
                    reference = summary
                outcomes.append(summary)
        print(json.dumps({'validation_seed_done': seed, 'episodes_done': len(outcomes),
                          'elapsed_s': round(time.perf_counter()-started, 2)}), flush=True)
    groups = {}
    for condition in ['ALL']+[c['id'] for c in CONDITIONS]:
        groups[condition] = {}
        for method in ('B1', 'M', 'M2'):
            values = [s for s in outcomes if s['method'] == method and (condition == 'ALL' or s['condition']['id'] == condition)]
            groups[condition][method] = {'n': len(values), 'falls': sum(s['fallen'] for s in values),
                'mean_alive_s': float(np.mean([s['time_alive_s'] for s in values])),
                'velocity_rmse': float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in values])),
                'walking_successes': sum(s['walking_success'] for s in values)}
    (output/'validation_summary.json').write_text(json.dumps({'groups': groups, 'no_retuning': True,
                  'closed_test_seeds_used': False, 'hypothesis_supported': None}, indent=2)+'\n')
    source = Path('results/pilot_h1')
    extra = []
    for seed in range(10):
        for condition in ['clean', 'jitter']+[f'drop_{s}_{n}' for s in ('joints', 'imu') for n in (1, 2, 4, 6)]:
            folder = output/'pilot_checks'/f'seed_{seed}'/condition
            summary = run_episode(folder, source/f'inputs/seed_{seed}/{condition}.json',
                                  source/f'inputs/seed_{seed}/initial.json', 'M2', horizons, adapter_config=config)
            audits.append(audit_episode(folder))
            extra.append(summary)
            if condition == 'clean':
                with np.load(folder/'signals.npz') as a, np.load(source/f'episodes/seed_{seed}/clean/B1_H40_40/signals.npz') as b:
                    for key in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
                        np.testing.assert_array_equal(a[key], b[key])
        print(json.dumps({'extra_seed_done': seed, 'extra_episodes_done': len(extra)}), flush=True)
    (output/'extra_summary.json').write_text(json.dumps([{'seed': s['seed'], 'condition': s['condition'],
              'fallen': s['fallen'], 'alive_s': s['time_alive_s'], 'rmse': s['velocity_rmse_after_warmup_before_fall_m_s']} for s in extra], indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audits, indent=2)+'\n')
    profile['not_validated_yet'] = False
    (output/'validated_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps({'selected': best, 'validation': groups, 'extra_falls': sum(s['fallen'] for s in extra)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
