"""Build pilot tables and paired exploratory statistics from real manifests."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from document_sections import write_section
from run_pilot import case_folder, E2

import numpy as np


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    margin = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., float(center-margin)), min(1., float(center+margin))]


def read_summary(root, case):
    result = json.loads((case_folder(root, case)/'summary.json').read_text())
    result['_folder'] = str(case_folder(root, case))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/pilot_h1'))
    args = p.parse_args()
    root = args.output
    plan = json.loads((root/'plan.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    extensions = json.loads((root/'extensions_plan.json').read_text())
    raw = [read_summary(root, c) for c in plan['primary_cases']+extensions['cases']]
    assert len(raw) == plan['planned_episode_count']
    assert set(s['seed'] for s in raw) == set(range(10))
    selected = []
    for s in raw:
        condition = s['condition']['id']
        if s['method'] == 'M' and condition in [c['id'] for c in E2]:
            stream = 'joints' if condition.startswith('J') else 'imu'
            if s['horizon_'+stream+'_ms'] != selection['selected'][stream+'_ms']:
                continue
        selected.append(s)
    # Check pairing, restored recurrent memory, warning counters and fresh warmup.
    pairing = {}
    warmup_references = {}
    zero_age_cases = {}
    baseline_manifest = json.loads(Path('results/baseline_h1/headless20/summary.json').read_text())
    for s in raw:
        for key in ('upstream_commit', 'checkpoint_sha256', 'scene_sha256', 'config_sha256', 'model_asset_sha256', 'joint_order'):
            assert s[key] == baseline_manifest[key]
        assert s['upstream_config'] == baseline_manifest['config']
        assert s['input_shape'] == [1, 41] and s['output_shape'] == [1, 10]
        assert s['cuda_build'] is None and s['torch_threads'] == 1 and s['interop_threads'] == 1
        assert s['shape_probe_memory_restored'] and all(v == 0 for v in s['initial_policy_buffer_max_abs'].values())
        assert not any(s['mujoco_warnings'])
        key = (s['seed'], s['condition']['id'])
        hashes = (s['initial_state_sha256'], s['schedule_sha256'])
        assert key not in pairing or pairing[key] == hashes
        pairing[key] = hashes
        signals = np.load(Path(s['_folder'])/'signals.npz')
        warmup = signals['decision_steps'] <= round(plan['config']['runtime']['warmup_s']/0.002)
        snapshot = (signals['obs'][warmup], signals['actions'][warmup], signals['q_true'][:101])
        if s['seed'] not in warmup_references:
            warmup_references[s['seed']] = [x.copy() for x in snapshot]
        else:
            for a, b in zip(snapshot, warmup_references[s['seed']]):
                np.testing.assert_array_equal(a, b)
        if s['condition']['id'] == 'clean':
            snapshot = [signals[k] for k in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true')]
            if s['seed'] not in zero_age_cases:
                zero_age_cases[s['seed']] = [x.copy() for x in snapshot]
            else:
                for a, b in zip(snapshot, zero_age_cases[s['seed']]):
                    np.testing.assert_array_equal(a, b)
        signals.close()
    grouped = {}
    for s in selected:
        grouped.setdefault((s['condition']['id'], s['method']), []).append(s)
    groups = []
    for (condition, method), runs in sorted(grouped.items()):
        n = len(runs)
        falls = sum(s['fallen'] for s in runs)
        groups.append({'condition': condition, 'method': method, 'n': n, 'falls': falls,
                       'fall_rate': falls/n, 'wilson95': wilson(falls, n),
                       'mean_alive_s': float(np.mean([s['time_alive_s'] for s in runs])),
                       'walking_successes': sum(s['walking_success'] for s in runs),
                       'mean_velocity_rmse_m_s': float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in runs])),
                       'mean_forward_speed_m_s': float(np.mean([s['mean_vx_yaw_after_warmup_before_fall_m_s'] for s in runs])),
                       'mean_joint_position_rmse_rad': float(np.mean([s['joint_prediction_rmse_rad'] for s in runs])),
                       'mean_gravity_error_rad': float(np.mean([s['gravity_prediction_mean_angle_rad'] for s in runs])),
                       'max_age_joints_ms': 1000*max(s['ages_s']['joints']['max'] for s in runs),
                       'max_age_imu_ms': 1000*max(s['ages_s']['imu']['max'] for s in runs)})
    pairs = np.zeros((10, 6))
    for i, seed in enumerate(range(10)):
        for j, c in enumerate(E2):
            b = [s for s in selected if s['seed'] == seed and s['condition']['id'] == c['id'] and s['method'] == 'B1']
            m = [s for s in selected if s['seed'] == seed and s['condition']['id'] == c['id'] and s['method'] == 'M']
            assert len(b) == len(m) == 1
            pairs[i, j] = int(b[0]['fallen'])-int(m[0]['fallen'])
    seed_differences = pairs.mean(axis=1)
    rng = np.random.default_rng(plan['config']['statistics']['bootstrap_rng_seed'])
    bootstrap = seed_differences[rng.integers(0, 10, size=(plan['config']['statistics']['bootstrap_resamples'], 10))].mean(axis=1)
    primary = [s for s in selected if s['condition']['id'] in [c['id'] for c in E2] and s['method'] in ('B1', 'M')]
    aggregate = {m: {'n': 60, 'falls': sum(s['fallen'] for s in primary if s['method'] == m),
                     'walking_successes': sum(s['walking_success'] for s in primary if s['method'] == m),
                     'mean_alive_s': float(np.mean([s['time_alive_s'] for s in primary if s['method'] == m]))} for m in ('B1', 'M')}
    timing = {}
    for method in ('B0', 'B1', 'M', 'B2', 'AJ', 'AI'):
        method_runs = [s for s in selected if s['method'] == method]
        values = {k: [] for k in ('adapter_ms', 'inference_ms', 'control_cycle_ms')}
        for s in method_runs:
            with np.load(Path(s['_folder'])/'signals.npz') as x:
                for k in values:
                    values[k].extend(x[k].tolist())
        timing[method] = {'episodes': len(method_runs), 'decision_count': len(values['adapter_ms']),
                          'median_simulation_to_wall_ratio': float(np.median([s['simulation_to_wall_ratio'] for s in method_runs])),
                          'median_episode_wall_s_including_setup_and_output': float(np.median([s['episode_wall_seconds_including_setup_and_output'] for s in method_runs])),
                          'measurements_ms': {k: {'median': float(np.median(v)), 'p95': float(np.percentile(v, 95))} for k, v in values.items()}}
    result = {'status': 'exploratory_pilot_only', 'hypothesis_supported': None, 'closed_test_executed': False,
              'raw_episode_count': len(raw), 'selected_episode_count': len(selected),
              'validation_episode_count': 3, 'seeds': list(range(10)),
              'horizon_selection': selection, 'groups': groups, 'primary_E2': aggregate,
              'paired_E2_fall_rate_difference_B1_minus_M': float(seed_differences.mean()),
              'paired_seed_cluster_bootstrap95_exploratory': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
              'bootstrap_resamples': len(bootstrap), 'bootstrap_rng_seed': plan['config']['statistics']['bootstrap_rng_seed'],
              'inference_limitation': 'Same seed 0–9 used for horizon selection. Intervals describe pilot and cannot confirm hypothesis.',
              'pairing_and_warmup_checks': {'paired_condition_seed_groups': len(pairing),
                                           'model_weights_control_config_and_joint_order_identical_to_stage_one': True,
                                           'warmup_identical_for_every_method_and_condition': True,
                                           'E0_observations_actions_states_bitwise_identical_all_10_seeds': True},
              'timing': timing, 'total_episode_wall_s_including_setup_and_output': sum(s['episode_wall_seconds_including_setup_and_output'] for s in raw),
              'recorded_failures': [str(f) for f in (root/'episodes').rglob('failure.json')],
              'excluded_episodes': [], 'body_contact_episodes': sum(s['floor_contact_steps']['pelvis'] > 0 for s in raw)}
    (root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    group_headers = ['condition', 'method', 'n', 'falls', 'fall_rate', 'mean_alive_s', 'walking_successes',
                     'mean_velocity_rmse_m_s', 'mean_forward_speed_m_s', 'mean_joint_position_rmse_rad',
                     'mean_gravity_error_rad', 'max_age_joints_ms', 'max_age_imu_ms']
    with (root/'groups.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=group_headers, extrasaction='ignore')
        writer.writeheader(); writer.writerows(groups)
    episode_headers = ['seed', 'condition', 'method', 'horizon_joints_ms', 'horizon_imu_ms', 'fallen',
                       'time_alive_s', 'censored', 'walking_success', 'velocity_rmse_after_warmup_before_fall_m_s',
                       'joint_prediction_rmse_rad', 'gravity_prediction_mean_angle_rad', 'wall_seconds_loop',
                       'episode_wall_seconds_including_setup_and_output', 'folder']
    with (root/'episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=episode_headers, extrasaction='ignore')
        writer.writeheader()
        for s in raw:
            writer.writerow({**s, 'condition': s['condition']['id'], 'folder': s['_folder']})
    lines = ['# Численные таблицы пилота', '', 'Это пилот seed 0–9 с подбором горизонтов. Закрытая серия не выполнялась.', '',
             '| Условие | Метод | Падения / N | Среднее T_alive, с | Успешная ходьба / N | Среднее RMSE скорости, м/с |',
             '|---|---|---:|---:|---:|---:|']
    for g in groups:
        lines.append(f'| {g["condition"]} | {g["method"]} | {g["falls"]}/{g["n"]} | {g["mean_alive_s"]:.3f} | '
                     f'{g["walking_successes"]}/{g["n"]} | {g["mean_velocity_rmse_m_s"]:.3f} |')
    write_section(root/'REPORT.md', 'tables', 'Численные таблицы пилота', '\n'.join(lines))
    print(json.dumps({k: v for k, v in result.items() if k not in ('groups', 'timing')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
