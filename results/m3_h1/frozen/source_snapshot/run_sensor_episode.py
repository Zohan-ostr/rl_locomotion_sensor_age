"""H1 CPU episode with causal packet transport, using the stage-one controller.

Controller equations derived from Unitree Robotics, BSD-3-Clause;
see results/baseline_h1/UNITREE_LICENSE.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

# Explicit project imports even under python -I; external PYTHONPATH stays excluded.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_h1_baseline import gravity, angles, sha
from sensor_age import STREAMS, METHODS, Packet, PacketQueue, adapt
from history_compensator import HistoryCompensator
from periodic_compensator import PeriodicCompensator
from cycle_compensator import CycleCompensator
from conservative_compensator import ConservativeCompensator
from hybrid_compensator import HybridCompensator

import mujoco
import numpy as np
import torch
import yaml


def configure_threads(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)


def percentile_stats(values):
    a = np.asarray(values)
    return {'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)), 'max': float(np.max(a))}


def run_episode(output, schedule_path, initial_path, method='B1', horizons_ms=(40, 40),
                upstream=Path('vendor/unitree_rl_gym'), pilot_config=Path('configs/pilot.yaml'), video=False,
                adapter_config=None, oracle_components=(), record_diagnostics=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite episode: {output}')
    output.mkdir(parents=True, exist_ok=True)
    root = upstream.resolve()
    cfg_path = root/'deploy/deploy_mujoco/configs/h1.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    pilot = yaml.safe_load(pilot_config.read_text())
    schedule = json.loads(Path(schedule_path).read_text())
    initial = json.loads(Path(initial_path).read_text())
    dt, decimation = cfg['simulation_dt'], cfg['control_decimation']
    if schedule['dt'] != dt or schedule['decimation'] != decimation:
        raise ValueError('Schedule and physical/controller time steps differ')
    if schedule['duration_s'] != pilot['runtime']['duration_s']:
        raise ValueError('Schedule duration differs from pilot configuration')
    if schedule['seed'] != initial['seed']:
        raise ValueError('Initial-state/schedule seed mismatch')
    if method not in METHODS or min(horizons_ms) < 0:
        raise ValueError('Invalid method or prediction horizon')
    if method == 'B0' and any(entry['drop'] or entry['sample_step'] != entry['delivery_step']
                             for entries in schedule['streams'].values() for entry in entries):
        raise ValueError('B0 fresh is permitted only for E0')
    policy_path = Path(cfg['policy_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)))
    xml = Path(cfg['xml_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)))
    policy = torch.jit.load(str(policy_path), map_location='cpu').eval()
    buffers = {k: v.clone() for k, v in policy.named_buffers()}
    with torch.inference_mode():
        probe = policy(torch.zeros(1, cfg['num_obs']))
        for k, v in policy.named_buffers():
            v.copy_(buffers[k])
            assert torch.equal(v, buffers[k])
    n = cfg['num_actions']
    assert list(probe.shape) == [1, n]
    assert all(v.device.type == 'cpu' for v in policy.parameters())
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    assert (model.nq, model.nv, model.nu) == (7+n, 6+n, n)
    assert np.array_equal(model.jnt_qposadr[1:], np.arange(7, 7+n))
    assert np.array_equal(model.jnt_dofadr[1:], np.arange(6, 6+n))
    assert np.array_equal(model.actuator_trnid[:, 0], np.arange(1, n+1))
    data.qpos[:] = initial['qpos']
    data.qvel[:] = initial['qvel']
    mujoco.mj_forward(model, data)
    limits = model.jnt_range[1:].copy()
    limits[~model.jnt_limited[1:].astype(bool)] = [-np.inf, np.inf]
    kp, kd, default = [np.array(cfg[k], dtype=np.float32) for k in ('kps', 'kds', 'default_angles')]
    cmd = np.array(cfg['cmd_init'], dtype=np.float32)
    cmd_scale = np.array(cfg['cmd_scale'], dtype=np.float32)
    action = np.zeros(n, dtype=np.float32)
    target = default.copy()
    obs = np.zeros(cfg['num_obs'], dtype=np.float32)
    horizons = dict(zip(STREAMS, np.array(horizons_ms)/1000))
    adapter_parameters = dict(adapter_config or {})
    adapter_family = adapter_parameters.pop('family', 'acceleration')
    if adapter_family not in ('acceleration', 'periodic', 'cycle', 'conservative', 'hybrid', 'linear'):
        raise ValueError('Unknown adapter family')
    adapter_class = {'periodic': PeriodicCompensator, 'acceleration': HistoryCompensator,
                     'cycle': CycleCompensator, 'conservative': ConservativeCompensator,
                     'hybrid': HybridCompensator}.get(adapter_family)
    if method == 'M3':
        from linear_compensator import LinearCompensator
        history_adapter = LinearCompensator(limits, horizons, adapter_parameters)
    else:
        history_adapter = adapter_class(limits, horizons, adapter_parameters) if method == 'M2' else None
    if set(oracle_components)-{'q', 'dq', 'omega', 'g'}:
        raise ValueError('Unknown oracle component')
    if method == 'M3' and oracle_components:
        raise ValueError('Oracle truth is forbidden in operational M3')
    diagnostic_observer = None
    if record_diagnostics:
        profile = json.loads(Path('configs/m2_profile.json').read_text())
        diagnostic_observer = HybridCompensator(limits, {'joints': .12, 'imu': .02},
                                               {k: v for k, v in profile['adapter_config'].items() if k != 'family'})
    diagnostic_arrays = {}
    history_update_ms = 0.
    queues = {s: PacketQueue(s) for s in STREAMS}
    total_steps = round(schedule['duration_s']/dt)
    for stream in STREAMS:
        entries = schedule['streams'][stream]
        if len(entries) != total_steps//decimation+1:
            raise ValueError('Incomplete sample schedule')
        if [e['sample_step'] for e in entries] != list(range(0, total_steps+1, decimation)):
            raise ValueError('Nonuniform sample schedule')

    def measured():
        return {'joints': {'q': data.qpos[7:].copy(), 'dq': data.qvel[6:].copy()},
                'imu': {'omega': data.qvel[3:6].copy(), 'g': gravity(data.qpos[3:7])}}

    sample_arrays = {k: [] for k in ('sample_steps', 'q_true', 'dq_true', 'g_true', 'omega_true')}
    decision_arrays = {k: [] for k in ('decision_steps', 'obs', 'actions', 'q_delivered', 'dq_delivered',
                                     'g_delivered', 'omega_delivered', 'q_predicted', 'dq_predicted',
                                     'g_predicted', 'omega_predicted', 'selected_sample_steps',
                                     'selected_delivery_steps', 'ages_s', 'adapter_ms', 'inference_ms',
                                     'control_cycle_ms')}
    event_rows = []
    def sample_and_deliver(step, truth):
        nonlocal history_update_ms
        sample_arrays['sample_steps'].append(step)
        for key in ('q', 'dq'):
            sample_arrays[key+'_true'].append(truth['joints'][key])
        for key in ('g', 'omega'):
            sample_arrays[key+'_true'].append(truth['imu'][key])
        delivered, ages, selected = {}, {}, {}
        for stream in STREAMS:
            entry = schedule['streams'][stream][step//decimation]
            if not entry['drop']:
                packet = Packet(stream, step, entry['delivery_step'], truth[stream], dt)
                queues[stream].enqueue(packet, step)
            latest, events = queues[stream].deliver(step)
            for packet, accepted in events:
                event_rows.append([step, stream, packet.sample_step, packet.delivery_step, int(accepted)])
            selected[stream] = latest
            ages[stream] = (step-latest.sample_step)*dt
            delivered[stream] = latest.payload
        if history_adapter:
            update_started = time.perf_counter()
            history_adapter.update(selected, step)
            history_update_ms = (time.perf_counter()-update_started)*1000
        if diagnostic_observer:
            diagnostic_observer.update(selected, step)
        return delivered, ages, selected

    # Initial packet is a real t=0 measurement; never fabricate zero-valued fallback.
    sample_and_deliver(0, measured())
    fall_time = None
    consecutive = 0
    fall = pilot['fall']
    height_threshold = fall['nominal_height_m']*fall['height_fraction']
    required_steps = round(fall['persistence_s']/dt)
    max_roll_pitch = np.zeros(2)
    height_min = float(data.qpos[2])
    path_length = 0.0
    previous_xy = data.qpos[:2].copy()
    contact_names = ('left_ankle_link', 'right_ankle_link', 'pelvis')
    contacts = dict.fromkeys(contact_names, 0)
    onsets = dict.fromkeys(contact_names, 0)
    last_contacts = dict.fromkeys(contact_names, False)
    rows, q_errors, g_errors, action_changes = [], [], [], []
    valid_velocities = []
    renderer = writer = camera = None
    if video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(output/'walk.mp4', fps=25)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5, 135, -15
    start = time.perf_counter()
    cpu_start = time.process_time()
    cycle_start = start
    with torch.inference_mode():
        for step in range(1, total_steps+1):
            # Fresh local PD, unchanged across all sensor methods.
            data.ctrl[:] = (target-data.qpos[7:])*kp-data.qvel[6:]*kd
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError(f'Nonfinite physics at {step}')
            rpy = angles(data.qpos[3:7])
            height_min = min(height_min, float(data.qpos[2]))
            max_roll_pitch = np.maximum(max_roll_pitch, np.abs(rpy[:2]))
            path_length += float(np.linalg.norm(data.qpos[:2]-previous_xy))
            previous_xy = data.qpos[:2].copy()
            bad = data.qpos[2] < height_threshold or np.max(np.abs(rpy[:2])) > np.deg2rad(fall['roll_pitch_deg'])
            consecutive = consecutive+1 if bad else 0
            if consecutive >= required_steps and fall_time is None:
                fall_time = step*dt
            touching = dict.fromkeys(contact_names, False)
            for c in data.contact:
                a, b = int(c.geom1), int(c.geom2)
                if c.dist <= 0 and 0 in (model.geom_bodyid[a], model.geom_bodyid[b]):
                    body = model.geom_bodyid[b] if model.geom_bodyid[a] == 0 else model.geom_bodyid[a]
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
                    if name in touching:
                        touching[name] = True
            for name, value in touching.items():
                contacts[name] += int(value)
                onsets[name] += int(value and not last_contacts[name])
            last_contacts = touching
            if step % decimation == 0:
                truth = measured()
                delivered, ages, selected = sample_and_deliver(step, truth)
                t0 = time.perf_counter()
                phase = (step*dt % 0.8)/0.8
                if method == 'M3':
                    history_adapter.context(step, action, cmd, phase)
                predicted = (history_adapter.predict(step) if history_adapter else
                             truth if method == 'B0' else adapt(delivered, ages, method, horizons, limits))
                for key in oracle_components:
                    stream = 'joints' if key in ('q', 'dq') else 'imu'
                    predicted[stream][key] = truth[stream][key].copy()
                adapter_ms = (time.perf_counter()-t0)*1000+history_update_ms
                # Predict in physical units, then apply exact upstream normalization/order.
                obs[:3] = predicted['imu']['omega']*cfg['ang_vel_scale']
                obs[3:6] = predicted['imu']['g']
                obs[6:9] = cmd*cmd_scale
                obs[9:9+n] = (predicted['joints']['q']-default)*cfg['dof_pos_scale']
                obs[9+n:9+2*n] = predicted['joints']['dq']*cfg['dof_vel_scale']
                obs[9+2*n:9+3*n] = action
                obs[-2:] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
                if record_diagnostics:
                    saved = {k: v.clone() for k, v in policy.named_buffers()}
                    variants = {'B1': delivered, 'M': adapt(delivered, ages, 'M',
                                {'joints': .02, 'imu': .02}, limits),
                                'M2': diagnostic_observer.predict(step), 'fresh': truth}
                    for key in ('q', 'dq', 'omega', 'g'):
                        payload = {s: {k: v.copy() for k, v in delivered[s].items()} for s in STREAMS}
                        payload['joints' if key in ('q', 'dq') else 'imu'][key] = truth['joints' if key in ('q', 'dq') else 'imu'][key].copy()
                        variants['oracle_'+key] = payload
                    for name, value in saved.items():
                        diagnostic_arrays.setdefault('lstm_'+name, []).append(value.numpy().copy())
                    for name, payload in variants.items():
                        for key, value in policy.named_buffers():
                            value.copy_(saved[key])
                        candidate_obs = obs.copy()
                        candidate_obs[:3] = payload['imu']['omega']*cfg['ang_vel_scale']
                        candidate_obs[3:6] = payload['imu']['g']
                        candidate_obs[9:9+n] = (payload['joints']['q']-default)*cfg['dof_pos_scale']
                        candidate_obs[9+n:9+2*n] = payload['joints']['dq']*cfg['dof_vel_scale']
                        candidate_action = policy(torch.from_numpy(candidate_obs).unsqueeze(0)).numpy().copy().squeeze()
                        diagnostic_arrays.setdefault('action_'+name, []).append(candidate_action)
                        diagnostic_arrays.setdefault('obs_'+name, []).append(candidate_obs)
                    for key, value in policy.named_buffers():
                        value.copy_(saved[key])
                        assert torch.equal(value, saved[key])
                t0 = time.perf_counter()
                result = policy(torch.from_numpy(obs).unsqueeze(0))
                new_action = result.numpy().squeeze().copy()
                inference_ms = (time.perf_counter()-t0)*1000
                if result.shape != (1, n) or not np.isfinite(new_action).all() or not np.isfinite(obs).all():
                    raise FloatingPointError('Invalid policy I/O')
                action_changes.append(float(np.linalg.norm(new_action-action)))
                action = new_action
                target = action*cfg['action_scale']+default
                q_error = float(np.sqrt(np.mean((predicted['joints']['q']-truth['joints']['q'])**2)))
                # atan2 is accurate near zero and equals angular distance for unit vectors.
                pg, tg = predicted['imu']['g'], truth['imu']['g']
                g_error = float(np.arctan2(np.linalg.norm(np.cross(pg, tg)), np.dot(pg, tg)))
                yaw = rpy[2]
                vx, vy, vz = data.qvel[:3]
                body_vx = np.cos(yaw)*vx+np.sin(yaw)*vy
                body_vy = -np.sin(yaw)*vx+np.cos(yaw)*vy
                if step*dt >= pilot['runtime']['warmup_s'] and fall_time is None:
                    valid_velocities.append([body_vx, body_vy])
                    q_errors.append(q_error)
                    g_errors.append(g_error)
                now = time.perf_counter()
                cycle_ms = (now-cycle_start)*1000
                cycle_start = now
                decision_arrays['decision_steps'].append(step)
                decision_arrays['obs'].append(obs.copy())
                decision_arrays['actions'].append(action.copy())
                for key in ('q', 'dq', 'g', 'omega'):
                    stream = 'joints' if key in ('q', 'dq') else 'imu'
                    decision_arrays[key+'_delivered'].append(delivered[stream][key].copy())
                    decision_arrays[key+'_predicted'].append(predicted[stream][key].copy())
                decision_arrays['selected_sample_steps'].append([selected[s].sample_step for s in STREAMS])
                decision_arrays['selected_delivery_steps'].append([selected[s].delivery_step for s in STREAMS])
                decision_arrays['ages_s'].append([ages[s] for s in STREAMS])
                for key, value in [('adapter_ms', adapter_ms), ('inference_ms', inference_ms), ('control_cycle_ms', cycle_ms)]:
                    decision_arrays[key].append(value)
                rows.append([step*dt, *data.qpos[:3], *rpy, vx, vy, vz, body_vx, body_vy,
                             *cmd, int(fall_time is not None), ages['joints'], ages['imu'],
                             selected['joints'].sample_time, selected['imu'].sample_time,
                             selected['joints'].delivery_time, selected['imu'].delivery_time,
                             q_error, g_error, adapter_ms, inference_ms, cycle_ms,
                             *[int(touching[s]) for s in contact_names]])
            if renderer is not None and step % 20 == 0:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
            if fall_time is not None and fall['stop_after_confirmed_fall']:
                break
    elapsed, cpu_elapsed = time.perf_counter()-start, time.process_time()-cpu_start
    if writer is not None:
        writer.close()
        renderer.close()
    names = ['time_s', 'x_m', 'y_m', 'height_m', 'roll_rad', 'pitch_rad', 'yaw_rad',
             'vx_world_m_s', 'vy_world_m_s', 'vz_world_m_s', 'vx_yaw_m_s', 'vy_yaw_m_s',
             'cmd_vx_m_s', 'cmd_vy_m_s', 'cmd_wz_rad_s', 'fallen', 'age_joints_s', 'age_imu_s',
             'sample_time_joints_s', 'sample_time_imu_s', 'delivery_time_joints_s', 'delivery_time_imu_s',
             'joint_position_rmse_rad', 'gravity_error_rad', 'adapter_ms', 'inference_ms', 'control_cycle_ms',
             'left_foot_contact', 'right_foot_contact', 'pelvis_contact']
    with (output/'trajectory.csv').open('w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(names)
        csvwriter.writerows(rows)
    with (output/'delivery_events.csv').open('w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['decision_step', 'stream', 'sample_step', 'delivery_step', 'accepted'])
        csvwriter.writerows(event_rows)
    np.savez_compressed(output/'signals.npz', **{k: np.asarray(v) for k, v in {**sample_arrays, **decision_arrays}.items()})
    if diagnostic_arrays:
        np.savez_compressed(output/'diagnostics.npz', **{k: np.asarray(v) for k, v in diagnostic_arrays.items()})
    valid = np.array(valid_velocities).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((valid-cmd[:2])**2, axis=1)))) if len(valid) else None
    time_alive = fall_time if fall_time is not None else schedule['duration_s']
    ages_array = np.asarray(decision_arrays['ages_s'])
    assets = {str(f.relative_to(root)): sha(f) for f in sorted((root/'resources/robots/h1').rglob('*')) if f.is_file()}
    summary = {
        'seed': initial['seed'], 'condition': schedule['condition'], 'method': method,
        'oracle_components': list(oracle_components), 'record_diagnostics': record_diagnostics,
        'horizon_joints_ms': horizons_ms[0], 'horizon_imu_ms': horizons_ms[1],
        'adapter_config': {'family': adapter_family, **history_adapter.config} if history_adapter else None,
        'schedule_path': str(Path(schedule_path).resolve()), 'schedule_sha256': sha(Path(schedule_path)),
        'initial_state_path': str(Path(initial_path).resolve()), 'initial_state_sha256': sha(Path(initial_path)),
        'initial_qpos': initial['qpos'], 'initial_qvel': initial['qvel'],
        'upstream_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': sha(policy_path), 'scene_sha256': sha(xml), 'config_sha256': sha(cfg_path),
        'model_asset_sha256': assets, 'upstream_config': cfg, 'pilot_config': pilot,
        'pilot_config_sha256': sha(pilot_config),
        'runner_sha256': {f.name: sha(f) for f in [Path(__file__), Path(__file__).with_name('sensor_age.py'), Path(__file__).with_name('run_h1_baseline.py'), Path(__file__).with_name('history_compensator.py'), Path(__file__).with_name('periodic_compensator.py'), Path(__file__).with_name('cycle_compensator.py'), Path(__file__).with_name('conservative_compensator.py'), Path(__file__).with_name('hybrid_compensator.py')]},
        'python': platform.python_version(), 'platform': platform.platform(), 'torch': torch.__version__,
        'mujoco': mujoco.__version__, 'numpy': np.__version__, 'cuda_build': torch.version.cuda,
        'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
        'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'joint_order': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt)],
        'shape_probe_memory_restored': True, 'initial_policy_buffer_max_abs': {k: float(v.abs().max()) for k, v in buffers.items()},
        'input_shape': [1, cfg['num_obs']], 'output_shape': list(probe.shape),
        'simulation_steps': step, 'simulation_seconds': step*dt, 'mujoco_time_s': float(data.time),
        'max_episode_seconds': schedule['duration_s'], 'policy_calls': len(rows), 'command': cmd.tolist(),
        'fallen': fall_time is not None, 'fall_time_s': fall_time, 'time_alive_s': time_alive,
        'censored': fall_time is None, 'fall_rule': fall, 'height_threshold_m': height_threshold,
        'height_min_m': height_min, 'max_abs_roll_pitch_deg': np.rad2deg(max_roll_pitch).tolist(),
        'displacement_xy_m': (data.qpos[:2]-np.asarray(initial['qpos'][:2])).tolist(), 'path_length_xy_m': path_length,
        'mean_vx_yaw_after_warmup_before_fall_m_s': float(valid[:, 0].mean()) if len(valid) else None,
        'velocity_rmse_after_warmup_before_fall_m_s': rmse, 'velocity_valid_samples': len(valid),
        'walking_success': fall_time is None and rmse is not None and rmse <= pilot['statistics']['walking_rmse_threshold_m_s'],
        'joint_prediction_rmse_rad': float(np.sqrt(np.mean(np.square(q_errors)))) if q_errors else None,
        'gravity_prediction_mean_angle_rad': float(np.mean(g_errors)) if g_errors else None,
        'action_increment_mean_l2': float(np.mean(action_changes)),
        'ages_s': {s: percentile_stats(ages_array[:, i]) for i, s in enumerate(STREAMS)},
        'transport': {s: {'arrived': queues[s].arrived, 'accepted': queues[s].accepted, 'rejected_old': queues[s].rejected,
                          'dropped_sample_count_executed': sum(e['drop'] for e in schedule['streams'][s] if e['sample_step'] <= step)} for s in STREAMS},
        'floor_contact_steps': contacts, 'floor_contact_onsets': onsets,
        'wall_seconds_loop': elapsed, 'cpu_seconds_loop': cpu_elapsed, 'simulation_to_wall_ratio': (step*dt)/elapsed,
        'timing_ms': {k: percentile_stats(decision_arrays[k]) for k in ('adapter_ms', 'inference_ms', 'control_cycle_ms')},
        'mujoco_warnings': [int(w.number) for w in data.warning], 'video': video,
        'signal_stream_order': list(STREAMS), 'test_seeds_used': 100 <= initial['seed'] < 140,
    }
    if method == 'M3':
        model_path = Path(history_adapter.config['model_path'])
        summary['adapter_model_sha256'] = sha(model_path)
        summary['runner_sha256']['linear_compensator.py'] = sha(Path(__file__).with_name('linear_compensator.py'))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--method', choices=METHODS, default='B1')
    p.add_argument('--horizon-joints-ms', type=int)
    p.add_argument('--horizon-imu-ms', type=int)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--upstream', type=Path, default=Path('vendor/unitree_rl_gym'))
    p.add_argument('--pilot-config', type=Path, default=Path('configs/pilot.yaml'))
    p.add_argument('--video', action='store_true')
    p.add_argument('--adapter-config', type=Path, help='JSON parameters for M2')
    p.add_argument('--oracle', action='append', default=[], choices=['q', 'dq', 'omega', 'g'])
    p.add_argument('--record-diagnostics', action='store_true')
    args = p.parse_args()
    parameters = json.loads(args.adapter_config.read_text()) if args.adapter_config else None
    default_horizons = (40, 40)
    if args.method == 'M2' and parameters is None:
        profile = json.loads(Path('configs/m2_profile.json').read_text())
        parameters = profile['adapter_config']
        default_horizons = (profile['horizon_joints_ms'], profile['horizon_imu_ms'])
    if args.method == 'M3' and parameters is None:
        profile = json.loads(Path('configs/m3_profile.json').read_text())
        parameters = profile['adapter_config']
        default_horizons = (profile['horizon_joints_ms'], profile['horizon_imu_ms'])
    horizons = tuple(default if explicit is None else explicit
                     for explicit, default in zip((args.horizon_joints_ms, args.horizon_imu_ms), default_horizons))
    configure_threads(args.threads)
    summary = run_episode(args.output, args.schedule, args.initial, args.method,
                          horizons, args.upstream, args.pilot_config, args.video, parameters,
                          args.oracle, args.record_diagnostics)
    print(json.dumps({k: summary[k] for k in ('seed', 'condition', 'method', 'fallen', 'time_alive_s',
                      'velocity_rmse_after_warmup_before_fall_m_s', 'timing_ms')}, indent=2))


if __name__ == '__main__':
    main()
