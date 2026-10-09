"""Prepare paired inputs, run only seed 0–9, tune horizons, then pilot extensions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h1_baseline import sha
from run_sensor_episode import configure_threads, run_episode
from sensor_age import make_schedule
from check_sensor_age import audit_episode

import mujoco
import numpy as np
import yaml


E2 = [{'id': f'{s}{d}', f'delay_{stream}_ms': d}
      for s, stream in [('J', 'joints'), ('I', 'imu')] for d in (20, 40, 80)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Different existing input: {path}')
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(root, config):
    seeds = config['initialization']['seeds']
    if seeds != list(range(10)):
        raise ValueError('This pilot is restricted to tuning seed 0–9')
    model = mujoco.MjModel.from_xml_path('vendor/unitree_rl_gym/resources/robots/h1/scene.xml')
    extension_conditions = ([{'id': f'equal{d}', 'delay_joints_ms': d, 'delay_imu_ms': d} for d in (20, 40, 80, 120)]
                            + [{'id': f'drop_{stream}_{n}', 'drop_stream': stream, 'missing_packets': n}
                               for stream in ('joints', 'imu') for n in (1, 2, 4, 6)]
                            + [{'id': 'jitter', 'jitter': True}])
    conditions = [{'id': 'clean'}]+E2+extension_conditions
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 20261009]))
        delta = rng.uniform(-config['initialization']['joint_perturbation_rad'],
                            config['initialization']['joint_perturbation_rad'], model.nq-7)
        qpos = model.qpos0.copy()
        qpos[7:] += delta
        write_json(root/f'inputs/seed_{seed}/initial.json', {'seed': seed, 'qpos': qpos.tolist(),
                   'qvel': np.zeros(model.nv).tolist(), 'joint_delta_rad': delta.tolist()})
        for condition in conditions:
            schedule = make_schedule(seed, condition, duration=config['runtime']['duration_s'],
                                     warmup=config['runtime']['warmup_s'])
            write_json(root/f'inputs/seed_{seed}/{condition["id"]}.json', schedule)
    write_json(root/'validation/initial.json', {'seed': 0, 'qpos': model.qpos0.tolist(), 'qvel': np.zeros(model.nv).tolist(),
                                             'joint_delta_rad': np.zeros(model.nq-7).tolist()})
    write_json(root/'validation/clean.json', make_schedule(0, {'id': 'clean'}))
    primary = []
    for seed in seeds:
        for method in ('B0', 'B1', 'M'):
            primary.append({'seed': seed, 'condition': 'clean', 'method': method, 'horizons_ms': [40, 40]})
        for condition in E2:
            primary.append({'seed': seed, 'condition': condition['id'], 'method': 'B1', 'horizons_ms': [0, 0]})
            for horizon in config['adapter']['horizon_candidates_ms']:
                primary.append({'seed': seed, 'condition': condition['id'], 'method': 'M', 'horizons_ms': [horizon, horizon]})
    plan = {'status': 'pilot_only', 'test_seeds_used': False, 'config': config,
            'primary_cases': primary, 'extension_conditions': extension_conditions,
            'diagnostic_scope': 'B2/AJ/AI, six E2 conditions, seed 0 only',
            'selection_note': 'In J-only/I-only conditions the other stream has exact zero-age identity. '
                              'Three common-horizon runs therefore cover all nine independent horizon pairs without duplicate runs.',
            'planned_episode_count': len(primary)+len(extension_conditions)*len(seeds)*2+18,
            'validation_episode_count': 3,
            'not_run': ['closed seeds 100–139', 'mixed E2 delay pairs', 'noise', 'second command speed']}
    write_json(root/'plan.json', plan)
    print(f'Prepared {len(primary)} primary and {plan["planned_episode_count"]-len(primary)} extension/diagnostic episodes', flush=True)
    return plan


def case_folder(root, case):
    hj, hi = case['horizons_ms']
    return root/f'episodes/seed_{case["seed"]}/{case["condition"]}/{case["method"]}_H{hj}_{hi}'


def execute(root, cases, resume=False):
    summaries = []
    start = time.perf_counter()
    audit_records = []
    for i, case in enumerate(cases):
        folder = case_folder(root, case)
        initial = root/f'inputs/seed_{case["seed"]}/initial.json'
        schedule = root/f'inputs/seed_{case["seed"]}/{case["condition"]}.json'
        if resume and (folder/'summary.json').exists():
            summary = json.loads((folder/'summary.json').read_text())
            if summary['schedule_sha256'] != sha(schedule) or summary['initial_state_sha256'] != sha(initial):
                raise RuntimeError('Resume input mismatch')
            for name, digest in summary['runner_sha256'].items():
                if sha(Path(__file__).with_name(name)) != digest:
                    raise RuntimeError('Cannot resume outcomes generated by different implementation')
        else:
            t0 = time.perf_counter()
            try:
                summary = run_episode(folder, schedule, initial, case['method'], tuple(case['horizons_ms']))
            except Exception as exc:
                folder.mkdir(parents=True, exist_ok=True)
                (folder/'failure.json').write_text(json.dumps({'case': case, 'error': repr(exc)}, indent=2)+'\n')
                raise
            summary['episode_wall_seconds_including_setup_and_output'] = time.perf_counter()-t0
            (folder/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        audit_records.append(audit_episode(folder))
        summaries.append(summary)
        if (i+1) % 5 == 0 or i+1 == len(cases):
            print(json.dumps({'completed': i+1, 'total': len(cases), 'elapsed_s': round(time.perf_counter()-start, 2),
                              'last_case': case, 'last_fallen': summary['fallen'], 'last_alive_s': summary['time_alive_s']}, ensure_ascii=False), flush=True)
    return summaries, audit_records


def select_horizons(root, plan):
    summaries = [json.loads((case_folder(root, c)/'summary.json').read_text()) for c in plan['primary_cases']]
    selection = {'status': 'pilot_recommendation_not_test_result', 'test_seeds_used': False,
                 'candidates': {}, 'selected': {}, 'ranking': plan['config']['adapter']['ranking']}
    for stream, prefix in [('joints', 'J'), ('imu', 'I')]:
        candidates = []
        for h in plan['config']['adapter']['horizon_candidates_ms']:
            subset = [s for s in summaries if s['method'] == 'M' and s['condition']['id'].startswith(prefix)
                      and s['horizon_joints_ms'] == h and s['horizon_imu_ms'] == h]
            assert len(subset) == 30
            record = {'horizon_ms': h, 'n': len(subset), 'falls': sum(s['fallen'] for s in subset),
                      'mean_alive_s': float(np.mean([s['time_alive_s'] for s in subset])),
                      'mean_velocity_rmse_m_s': float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in subset])),
                      'walking_successes': sum(s['walking_success'] for s in subset)}
            candidates.append(record)
        best = min(candidates, key=lambda x: (x['falls'], -x['mean_alive_s'], x['mean_velocity_rmse_m_s'], x['horizon_ms']))
        selection['candidates'][stream] = candidates
        selection['selected'][stream+'_ms'] = best['horizon_ms']
    write_json(root/'selection.json', selection)
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def validate(root):
    reports = []
    for method in ('B0', 'B1', 'M'):
        folder = root/f'validation/{method}'
        run_episode(folder, root/'validation/clean.json', root/'validation/initial.json', method)
        reports.append(audit_episode(folder))
    a = np.load(root/'validation/B0/signals.npz')
    comparisons = {}
    for method in ('B1', 'M'):
        b = np.load(root/f'validation/{method}/signals.npz')
        for key in ('obs', 'actions', 'q_true', 'dq_true', 'g_true', 'omega_true'):
            np.testing.assert_array_equal(a[key], b[key])
        comparisons[method] = 'bitwise identical observations, actions and physical sensor samples to B0'
    fresh = np.genfromtxt(root/'validation/B0/trajectory.csv', delimiter=',', names=True)
    baseline = np.genfromtxt('results/baseline_h1/headless20/trajectory.csv', delimiter=',', names=True)
    physical = ['time_s', 'x_m', 'y_m', 'height_m', 'roll_rad', 'pitch_rad', 'yaw_rad', 'vx_world_m_s', 'vy_world_m_s', 'vz_world_m_s']
    errors = {k: float(np.max(np.abs(fresh[k]-baseline[k]))) for k in physical}
    # New logger uses exact step*dt instead of accumulated d.time. Compare within roundoff.
    assert max(errors.values()) < 1e-9
    result = {'episode_audits': reports, 'zero_age_integration': comparisons, 'stage_one_baseline_max_errors': errors}
    (root/'validation/checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('results/pilot_h1'))
    p.add_argument('--stage', choices=('prepare', 'validate', 'primary', 'extensions'), required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    config = yaml.safe_load(Path('configs/pilot.yaml').read_text())
    configure_threads(config['runtime']['threads'])
    if args.stage == 'prepare':
        prepare(args.output, config)
        return
    plan = json.loads((args.output/'plan.json').read_text())
    if plan['config'] != config:
        raise RuntimeError('Configuration changed after preparing plan')
    if args.stage == 'validate':
        validate(args.output)
        return
    if not (args.output/'validation/checks.json').exists():
        raise RuntimeError('Run validation before tuning episodes')
    if args.stage == 'primary':
        summaries, audit = execute(args.output, plan['primary_cases'], args.resume)
        (args.output/'primary_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        select_horizons(args.output, plan)
    else:
        selection = json.loads((args.output/'selection.json').read_text())
        horizons = [selection['selected']['joints_ms'], selection['selected']['imu_ms']]
        cases = [{'seed': seed, 'condition': c['id'], 'method': m, 'horizons_ms': horizons if m == 'M' else [0, 0]}
                 for seed in range(10) for c in plan['extension_conditions'] for m in ('B1', 'M')]
        cases += [{'seed': 0, 'condition': c['id'], 'method': m, 'horizons_ms': horizons}
                  for c in E2 for m in ('B2', 'AJ', 'AI')]
        write_json(args.output/'extensions_plan.json', {'selection': selection, 'cases': cases})
        summaries, audit = execute(args.output, cases, args.resume)
        (args.output/'extensions_audit.json').write_text(json.dumps(audit, indent=2)+'\n')


if __name__ == '__main__':
    main()
