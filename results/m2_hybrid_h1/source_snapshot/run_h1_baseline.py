"""CPU-only headless adaptation of Unitree's deploy_mujoco.py.

Derived controller equations: Unitree Robotics, BSD-3-Clause;
see results/baseline_h1/UNITREE_LICENSE. No training or sensor faults.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
import torch
import yaml


def gravity(quat):
    w, x, y, z = quat
    return np.array([2 * (-z*x + w*y), -2 * (z*y + w*x),
                     1 - 2 * (w*w + z*z)])


def angles(quat):
    w, x, y, z = quat
    return np.array([np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, default=Path('vendor/unitree_rl_gym'))
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = args.upstream.resolve()
    config_path = root / 'deploy/deploy_mujoco/configs/h1.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    policy_path = Path(cfg['policy_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)))
    xml = Path(cfg['xml_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)))
    if policy_path.read_bytes().startswith(b'version https://git-lfs'):
        raise RuntimeError('Checkpoint is a Git LFS pointer')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    policy = torch.jit.load(str(policy_path), map_location='cpu').eval()
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.opt.timestep = cfg['simulation_dt']
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    n, dim = cfg['num_actions'], cfg['num_obs']
    assert (model.nq, model.nv, model.nu) == (7+n, 6+n, n)
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(1, model.njnt)]
    assert np.array_equal(model.jnt_qposadr[1:], np.arange(7, 7+n))
    assert np.array_equal(model.jnt_dofadr[1:], np.arange(6, 6+n))
    assert np.array_equal(model.actuator_trnid[:, 0], np.arange(1, n+1))
    initial_buffers = {name: value.clone() for name, value in policy.named_buffers()}
    with torch.inference_mode():
        probe = policy(torch.zeros((1, dim), dtype=torch.float32))
        # The shipped policy is recurrent: a shape probe must not change its memory.
        for name, value in policy.named_buffers():
            value.copy_(initial_buffers[name])
            assert torch.equal(value, initial_buffers[name])
    assert tuple(probe.shape) == (1, n) and torch.isfinite(probe).all()
    assert all(v.device.type == 'cpu' for v in policy.parameters())
    default = np.array(cfg['default_angles'], dtype=np.float32)
    kp, kd = (np.array(cfg[k], dtype=np.float32) for k in ('kps', 'kds'))
    cmd = np.array(cfg['cmd_init'], dtype=np.float32)
    cmd_scale = np.array(cfg['cmd_scale'], dtype=np.float32)
    action = np.zeros(n, dtype=np.float32)
    target = default.copy()
    obs = np.zeros(dim, dtype=np.float32)
    initial = data.qpos.copy()
    initial_height = float(initial[2])
    total_steps = round(args.seconds/model.opt.timestep)
    assert total_steps > 0 and abs(total_steps*model.opt.timestep-args.seconds) < 1e-8
    renderer = writer = camera = None
    if args.video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(args.output / 'walk.mp4', fps=25)
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 3.5, 135, -15
    rows, inference_ms, physics_ms, cycle_ms = [], [], [], []
    contact_counts = {name: 0 for name in ('left_ankle_link', 'right_ankle_link', 'pelvis')}
    contact_transitions = {name: 0 for name in contact_counts}
    contact_previous = {name: False for name in contact_counts}
    fall_time = None
    consecutive = 0
    max_bad_steps = round(0.2/model.opt.timestep)
    max_roll_pitch = np.zeros(2)
    height_min = initial_height
    path_length = 0.0
    previous_xy = initial[:2].copy()
    start = time.perf_counter()
    cpu_start = time.process_time()
    cycle_start = start
    with torch.inference_mode():
        for step in range(1, total_steps+1):
            data.ctrl[:] = (target-data.qpos[7:])*kp - data.qvel[6:]*kd
            t0 = time.perf_counter()
            mujoco.mj_step(model, data)
            physics_ms.append((time.perf_counter()-t0)*1000)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError(f'Nonfinite state at step {step}')
            rpy = angles(data.qpos[3:7])
            height_min = min(height_min, float(data.qpos[2]))
            max_roll_pitch = np.maximum(max_roll_pitch, np.abs(rpy[:2]))
            path_length += float(np.linalg.norm(data.qpos[:2]-previous_xy))
            previous_xy = data.qpos[:2].copy()
            bad = data.qpos[2] < 0.6*initial_height or np.max(np.abs(rpy[:2])) > np.pi/3
            consecutive = consecutive+1 if bad else 0
            if consecutive >= max_bad_steps and fall_time is None:
                fall_time = float(data.time)
            touching = {name: False for name in contact_counts}
            for contact in data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                # Floor belongs to world body 0. Count only active contacts.
                if contact.dist <= 0 and 0 in (model.geom_bodyid[a], model.geom_bodyid[b]):
                    body_id = model.geom_bodyid[b] if model.geom_bodyid[a] == 0 else model.geom_bodyid[a]
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if name in touching:
                        touching[name] = True
            for name, value in touching.items():
                contact_counts[name] += int(value)
                contact_transitions[name] += int(value and not contact_previous[name])
            contact_previous = touching
            if step % cfg['control_decimation'] == 0:
                # Exact upstream ordering: physics first, then policy at t=0.02.
                obs[:3] = data.qvel[3:6]*cfg['ang_vel_scale']
                obs[3:6] = gravity(data.qpos[3:7])
                obs[6:9] = cmd*cmd_scale
                obs[9:9+n] = (data.qpos[7:]-default)*cfg['dof_pos_scale']
                obs[9+n:9+2*n] = data.qvel[6:]*cfg['dof_vel_scale']
                obs[9+2*n:9+3*n] = action
                phase = (step*model.opt.timestep % 0.8)/0.8
                obs[-2:] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
                t0 = time.perf_counter()
                result = policy(torch.from_numpy(obs).unsqueeze(0))
                action = result.numpy().squeeze()
                inference_ms.append((time.perf_counter()-t0)*1000)
                assert result.shape == (1, n) and np.isfinite(action).all()
                target = action*cfg['action_scale'] + default
                now = time.perf_counter()
                cycle_ms.append((now-cycle_start)*1000)
                cycle_start = now
                # Free-joint linear qvel is world-frame; rotate to yaw frame for tracking.
                yaw = rpy[2]
                vx, vy, vz = data.qvel[:3]
                body_vx = np.cos(yaw)*vx+np.sin(yaw)*vy
                body_vy = -np.sin(yaw)*vx+np.cos(yaw)*vy
                rows.append([data.time, *data.qpos[:3], *rpy, vx, vy, vz,
                             body_vx, body_vy, *cmd, int(fall_time is not None),
                             inference_ms[-1], cycle_ms[-1],
                             *[int(touching[k]) for k in contact_counts]])
            if renderer is not None and step % 20 == 0:
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    elapsed = time.perf_counter()-start
    cpu_elapsed = time.process_time()-cpu_start
    if writer is not None:
        writer.close()
        renderer.close()
    headers = ['time_s', 'x_m', 'y_m', 'height_m', 'roll_rad', 'pitch_rad', 'yaw_rad',
               'vx_world_m_s', 'vy_world_m_s', 'vz_world_m_s', 'vx_yaw_m_s', 'vy_yaw_m_s',
               'cmd_vx_m_s', 'cmd_vy_m_s', 'cmd_wz_rad_s', 'fallen', 'inference_ms',
               'control_cycle_ms', 'left_foot_contact', 'right_foot_contact', 'pelvis_contact']
    with (args.output/'trajectory.csv').open('w') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(headers)
        writer_csv.writerows(rows)
    array = np.array(rows)
    steady = array[array[:, 0] >= 2.0]
    def timing(values):
        return {'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95))}
    assets = {str(f.relative_to(root)): sha(f) for f in sorted((root/'resources/robots/h1').rglob('*')) if f.is_file()}
    summary = {
        'upstream_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': sha(policy_path), 'scene_sha256': sha(xml),
        'config_sha256': sha(config_path), 'model_asset_sha256': assets,
        'config': cfg, 'python': platform.python_version(), 'platform': platform.platform(),
        'torch': torch.__version__, 'mujoco': mujoco.__version__, 'numpy': np.__version__,
        'torch_cuda_build': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(), 'input_shape': [1, dim], 'output_shape': list(probe.shape),
        'initial_policy_buffers': {k: {'shape': list(v.shape), 'max_abs': float(v.abs().max())}
                                   for k, v in initial_buffers.items()},
        'shape_probe_memory_restored': True,
        'joint_order': joints, 'initial_qpos': initial.tolist(), 'initialization': 'unchanged XML qpos0',
        'simulation_steps': total_steps, 'simulation_seconds': float(data.time), 'policy_calls': len(rows),
        'command': cmd.tolist(), 'fall_time_s': fall_time,
        'fall_rule': {'body': 'pelvis', 'reference_height_m': initial_height, 'height_fraction': 0.6,
                      'roll_pitch_deg': 60, 'persistence_s': 0.2, 'status': 'baseline diagnostic, not frozen pilot rule'},
        'height_min_m': height_min, 'height_after_2s_mean_m': float(steady[:, 3].mean()),
        'max_abs_roll_pitch_deg': np.rad2deg(max_roll_pitch).tolist(),
        'displacement_xy_m': (data.qpos[:2]-initial[:2]).tolist(), 'path_length_xy_m': path_length,
        'mean_vx_yaw_after_2s_m_s': float(steady[:, 10].mean()),
        'velocity_xy_rmse_after_2s_m_s': float(np.sqrt(np.mean(np.sum((steady[:, 10:12]-cmd[:2])**2, axis=1)))),
        'floor_contact_steps': contact_counts, 'floor_contact_onsets': contact_transitions,
        'wall_seconds_loop': elapsed, 'process_cpu_seconds_loop': cpu_elapsed,
        'simulation_to_wall_ratio': float(data.time)/elapsed,
        'physics_step': timing(physics_ms), 'policy_inference': timing(inference_ms),
        'control_cycle_including_metrics': timing(cycle_ms), 'video': args.video,
        'mujoco_warnings': [int(w.number) for w in data.warning],
    }
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'model_asset_sha256'}, indent=2))


if __name__ == '__main__':
    main()
