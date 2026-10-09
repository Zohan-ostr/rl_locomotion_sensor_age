"""CPU H1 demonstration; controller derived from Unitree (BSD-3-Clause).

See results/baseline_h1/UNITREE_LICENSE. Sensor faults affect policy only.
"""
import argparse
import csv
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
import torch
import yaml
from run_h1_baseline import angles, gravity, sha
from sensor_age import STREAMS, Packet, PacketQueue, adapt
from hybrid_compensator import HybridCompensator


class SensorGate:
    def __init__(self, dt):
        self.dt = dt
        self.queues = {s: PacketQueue(s) for s in STREAMS}
        self.dropped = dict.fromkeys(STREAMS, 0)

    def sample(self, step, truth, blocked):
        delivered, ages = {}, {}
        for s in STREAMS:
            # Always establish a real initial measurement, including when a key
            # was pressed before the simulation began.
            if step == 0 or not blocked[s]:
                self.queues[s].enqueue(Packet(s, step, step, truth[s], self.dt), step)
            else:
                self.dropped[s] += 1
            packet, _ = self.queues[s].deliver(step)
            delivered[s] = packet.payload
            ages[s] = (step-packet.sample_step)*self.dt
        return delivered, ages


class DemoControls:
    def __init__(self, method):
        self.method = method
        self.walking = True
        self.commands = []

    def apply(self, key):
        if key == 'w':
            self.walking = not self.walking
        elif key == 'c':
            self.method = 'M' if self.method == 'B1' else 'B1'
        elif key == 'r':
            self.walking = False
            return True
        return False


def reset_robot(model, data, policy, buffers, default):
    """Teleport upright at the current XY; clear velocities and policy memory."""
    xy, clock = data.qpos[:2].copy(), float(data.time)
    mujoco.mj_resetData(model, data)
    data.qpos[:2] = xy
    data.qpos[7:] = default
    data.time = clock
    for name, value in policy.named_buffers():
        value.copy_(buffers[name])
    mujoco.mj_forward(model, data)


class Window:
    def __init__(self, model):
        import glfw
        self.glfw = glfw
        if not glfw.init():
            raise RuntimeError('GLFW: нет доступа к графическому сеансу DISPLAY.')
        self.window = glfw.create_window(640, 480, 'H1 | hold I: IMU, J: joints', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Не удалось создать окно OpenGL. Используйте локальный графический сеанс.')
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.options = mujoco.MjvOption()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 3.5
        self.camera.azimuth, self.camera.elevation = 135, -15
        self.held = dict.fromkeys(STREAMS, False)
        self.controls = None
        self.cursor = None
        glfw.set_key_callback(self.window, self.key)
        glfw.set_window_focus_callback(self.window, self.focus)
        glfw.set_cursor_pos_callback(self.window, self.mouse)
        # No wheel/pan/zoom callbacks: orbit is the only camera interaction.

    def key(self, window, key, scancode, action, mods):
        g = self.glfw
        stream = {g.KEY_I: 'imu', g.KEY_J: 'joints'}.get(key)
        if stream:
            self.held[stream] = action != g.RELEASE
        if action == g.PRESS and getattr(self, 'controls', None) is not None:
            command = {g.KEY_W: 'w', g.KEY_C: 'c', g.KEY_R: 'r'}.get(key)
            if command:
                self.controls.commands.append(command)
        if key == g.KEY_ESCAPE and action == g.PRESS:
            g.set_window_should_close(window, True)

    def focus(self, window, focused):
        if not focused:
            self.held = dict.fromkeys(STREAMS, False)
            self.cursor = None

    def mouse(self, window, x, y):
        previous, self.cursor = self.cursor, (x, y)
        if previous and self.glfw.get_mouse_button(window, self.glfw.MOUSE_BUTTON_LEFT) == self.glfw.PRESS:
            self.camera.azimuth -= (x-previous[0])*0.3
            self.camera.elevation = np.clip(self.camera.elevation-(y-previous[1])*0.3, -85, 85)

    def poll(self):
        self.glfw.poll_events()
        return not self.glfw.window_should_close(self.window)

    def draw(self, model, data, ages, blocked, method, fallen, finished=False):
        self.camera.lookat[:] = data.qpos[:3]
        self.camera.distance = 3.5
        mujoco.mjv_updateScene(model, data, self.options, None, self.camera,
                             mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        width, height = self.glfw.get_framebuffer_size(self.window)
        if width <= 0 or height <= 0:
            return
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        sandbox = self.controls is not None
        labels = 'Simulation\nMethod (C)\nIMU (hold I)\nJoints (hold J)\nRobot (W / R)\nCamera\nExit'
        values = (f'{data.time:.2f}' + (' s | unlimited' if sandbox else ' / 20 s') + (' | FINISHED' if finished else '') +
                  f'\n{method}\n' + ('BLOCKED' if blocked['imu'] else 'LIVE') +
                  f" | age {ages['imu']*1000:.0f} ms\n" +
                  ('BLOCKED' if blocked['joints'] else 'LIVE') +
                  f" | age {ages['joints']*1000:.0f} ms\n" +
                  ('FALL DETECTED | R: reset' if fallen else
                   ('WALK' if not sandbox or self.controls.walking else 'STAND')) +
                  '\nDrag left mouse: orbit\nEsc / close window')
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, labels, values, self.context)
        self.glfw.swap_buffers(self.window)

    def close(self):
        self.context.free()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


def main():
    p = argparse.ArgumentParser(description='Интерактивная H1: удерживайте I / J для потери пакетов.')
    p.add_argument('--method', choices=['B1', 'M'], default='B1')
    p.add_argument('--output', type=Path)
    p.add_argument('--headless', action='store_true', help='Проверка без окна и без real-time sleep')
    p.add_argument('--auto-close', action='store_true', help='Закрыть окно после 20 секунд')
    p.add_argument('--sandbox', action='store_true', help='Отдельный свободный режим W/C/R до Esc')
    p.add_argument('--compensator', choices=['legacy', 'm2'], default='legacy', help='Какой M включается клавишей C')
    p.add_argument('--seconds', type=float, help='Ограничение для автоматической проверки свободного режима')
    p.add_argument('--event', action='append', default=[], metavar='TIME:KEY', help='Проверочный W/C/R в заданное время')
    p.add_argument('--block', action='append', default=[], metavar='STREAM:START:END',
                   help='Воспроизводимый пропуск, например imu:5:5.12 (секунды симуляции)')
    args = p.parse_args()
    if args.seconds is not None and (not args.sandbox or not np.isfinite(args.seconds) or args.seconds <= 0):
        p.error('--seconds: положительная длительность только для --sandbox')
    if args.headless and args.sandbox and args.seconds is None:
        p.error('Для headless sandbox укажите --seconds')
    duration = args.seconds if args.sandbox else 20
    scheduled_events = []
    for spec in args.event:
        when, key = spec.split(':')
        when = float(when)
        if not args.sandbox or key not in ('w', 'c', 'r') or not np.isfinite(when) or when < 0:
            p.error('--event: TIME:w/c/r только в sandbox')
        scheduled_events.append((when, key))
    scheduled_events.sort(key=lambda item: item[0])
    blocks = []
    for spec in args.block:
        stream, start, end = spec.split(':')
        start, end = float(start), float(end)
        if stream not in STREAMS or not np.isfinite([start, end]).all() or not 0 <= start < end or (duration is not None and end > duration):
            p.error('Блокировка: imu или joints, 0 <= START < END, в пределах длительности')
        blocks.append((stream, start, end))
    output = args.output or Path('results/interactive_h1')/time.strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(__file__).resolve().parents[1]
    upstream = root/'vendor/unitree_rl_gym'
    cfg_path = upstream/'deploy/deploy_mujoco/configs/h1.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    pilot = yaml.safe_load((root/'configs/pilot.yaml').read_text())
    policy_path = Path(cfg['policy_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(upstream)))
    xml = Path(cfg['xml_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(upstream)))
    model = mujoco.MjModel.from_xml_path(str(xml))
    dt, decimation = cfg['simulation_dt'], cfg['control_decimation']
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = data.qpos.copy()
    n = cfg['num_actions']
    assert (model.nq, model.nv, model.nu) == (7+n, 6+n, n)
    assert np.array_equal(model.actuator_trnid[:, 0], np.arange(1, n+1))
    policy = torch.jit.load(str(policy_path), map_location='cpu').eval()
    saved = {k: v.clone() for k, v in policy.named_buffers()}
    with torch.inference_mode():
        probe = policy(torch.zeros(1, cfg['num_obs']))
        for k, v in policy.named_buffers():
            v.copy_(saved[k])
            assert torch.equal(v, saved[k])
    assert probe.shape == (1, n)
    kp, kd, default = [np.array(cfg[k], dtype=np.float32) for k in ('kps', 'kds', 'default_angles')]
    cmd = np.array(cfg['cmd_init'], dtype=np.float32)
    walk_cmd = cmd.copy()
    controls = DemoControls(args.method)
    limits = model.jnt_range[1:].copy()
    limits[~model.jnt_limited[1:].astype(bool)] = [-np.inf, np.inf]
    profile_path = root/'configs/m2_profile.json'
    profile = json.loads(profile_path.read_text()) if args.compensator == 'm2' else None
    def new_adapter():
        if profile is None:
            return None
        parameters = dict(profile['adapter_config'])
        if parameters.pop('family') != 'hybrid':
            raise ValueError('Interactive M2 expects the validated hybrid profile')
        horizons = {'joints': profile['horizon_joints_ms']/1000, 'imu': profile['horizon_imu_ms']/1000}
        return HybridCompensator(limits, horizons, parameters)
    history_adapter = new_adapter()
    def measured():
        return {'joints': {'q': data.qpos[7:].copy(), 'dq': data.qvel[6:].copy()},
                'imu': {'g': gravity(data.qpos[3:7]), 'omega': data.qvel[3:6].copy()}}
    gate = SensorGate(dt)
    blocked = dict.fromkeys(STREAMS, False)
    _, ages = gate.sample(0, measured(), blocked)
    if history_adapter:
        history_adapter.update({s: gate.queues[s].latest for s in STREAMS}, 0)
    action, target = np.zeros(n, dtype=np.float32), default.copy()
    obs = np.zeros(cfg['num_obs'], dtype=np.float32)
    rows, signals = [], {k: [] for k in ('obs', 'actions', 'qpos', 'qvel', 'ages_s')}
    events, previous_blocked = [], blocked.copy()
    total_calls, chunk_index, resets, reset_step = 0, 0, 0, 0
    max_ages = np.zeros(2)
    dropped_total = dict.fromkeys(STREAMS, 0)
    fall_history = []
    names = ['time_s', 'x_m', 'y_m', 'height_m', 'roll_rad', 'pitch_rad', 'yaw_rad',
             'vx_m_s', 'vy_m_s', 'vz_m_s', 'blocked_joints', 'blocked_imu',
             'age_joints_s', 'age_imu_s', 'sample_time_joints_s', 'sample_time_imu_s', 'fallen']
    if args.sandbox:
        names += ['cmd_vx_m_s', 'compensation', 'reset_count']
    def flush_chunk():
        nonlocal chunk_index
        if not rows:
            return
        folder = output/'chunks'
        folder.mkdir(exist_ok=True)
        with (folder/f'{chunk_index:06d}.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(names)
            writer.writerows(rows)
        np.savez_compressed(folder/f'{chunk_index:06d}.npz', **{k: np.asarray(v) for k, v in signals.items()})
        rows.clear()
        for values in signals.values():
            values.clear()
        chunk_index += 1
    def record(event):
        if args.sandbox:
            with (output/'events.jsonl').open('a') as f:
                f.write(json.dumps(event)+'\n')
        else:
            events.append(event)
    window = None
    fall_time, bad_steps, step = None, 0, 0
    status = 'completed'
    started = time.perf_counter()
    try:
        if not args.headless:
            window = Window(model)
            if args.sandbox:
                window.controls = controls
        print('I: IMU; J: суставы; W: идти/стоять; C: компенсация; R: поднять (свободный режим); Esc: выход.', flush=True)
        deadline = time.perf_counter()
        with torch.inference_mode():
            for step in itertools.count(1):
                if duration is not None and step > round(duration/dt):
                    step -= 1
                    break
                if window and not window.poll():
                    status = 'closed_by_user'
                    step -= 1
                    break
                if args.sandbox and (step-1) % decimation == 0:
                    while scheduled_events and scheduled_events[0][0] <= (step-1)*dt+1e-10:
                        controls.commands.append(scheduled_events.pop(0)[1])
                    for key in controls.commands:
                        reset = controls.apply(key)
                        if reset:
                            reset_robot(model, data, policy, saved, default)
                            for s in STREAMS:
                                dropped_total[s] += gate.dropped[s]
                            gate = SensorGate(dt)
                            gate.sample(step-1, measured(), dict.fromkeys(STREAMS, False))
                            history_adapter = new_adapter()
                            if history_adapter:
                                history_adapter.update({s: gate.queues[s].latest for s in STREAMS}, step-1)
                            ages = dict.fromkeys(STREAMS, 0.)
                            action[:] = 0
                            target = default.copy()
                            fall_time, bad_steps = None, 0
                            reset_step = step-1
                            resets += 1
                        cmd[:] = walk_cmd if controls.walking else 0
                        event = {'time_s': (step-1)*dt, 'key': key, 'walking': controls.walking,
                                 'method': controls.method, 'reset_count': resets}
                        record(event)
                        print(json.dumps(event), flush=True)
                    controls.commands.clear()
                data.ctrl[:] = (target-data.qpos[7:])*kp-data.qvel[6:]*kd
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise FloatingPointError('Неконечное физическое состояние')
                rpy = angles(data.qpos[3:7])
                fall = pilot['fall']
                bad = (data.qpos[2] < fall['nominal_height_m']*fall['height_fraction'] or
                       np.max(np.abs(rpy[:2])) > np.deg2rad(fall['roll_pitch_deg']))
                bad_steps = bad_steps+1 if bad else 0
                if fall_time is None and bad_steps >= round(fall['persistence_s']/dt):
                    fall_time = step*dt
                    fall_history.append(fall_time)
                    print(f'Падение зарегистрировано: {fall_time:.3f} с; демонстрация продолжается.', flush=True)
                if step % decimation:
                    continue
                t = step*dt
                blocked = window.held.copy() if window else dict.fromkeys(STREAMS, False)
                for stream, start, end in blocks:
                    blocked[stream] |= start <= t < end
                for s in STREAMS:
                    if blocked[s] != previous_blocked[s]:
                        record({'time_s': t, 'stream': s, 'blocked': blocked[s]})
                        print(f'{t:.2f} с: {s} ' + ('BLOCKED' if blocked[s] else 'LIVE'), flush=True)
                previous_blocked = blocked.copy()
                delivered, ages = gate.sample(step, measured(), blocked)
                if history_adapter:
                    history_adapter.update({s: gate.queues[s].latest for s in STREAMS}, step)
                predicted = (history_adapter.predict(step) if history_adapter and controls.method == 'M' else
                             adapt(delivered, ages, controls.method, dict.fromkeys(STREAMS, .020), limits))
                obs[:3] = predicted['imu']['omega']*cfg['ang_vel_scale']
                obs[3:6] = predicted['imu']['g']
                obs[6:9] = cmd*np.array(cfg['cmd_scale'], dtype=np.float32)
                obs[9:9+n] = (predicted['joints']['q']-default)*cfg['dof_pos_scale']
                obs[9+n:9+2*n] = predicted['joints']['dq']*cfg['dof_vel_scale']
                obs[9+2*n:9+3*n] = action
                phase = ((step-reset_step)*dt % .8)/.8
                obs[-2:] = [np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)]
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().copy()
                if not np.isfinite(action).all() or not np.isfinite(obs).all():
                    raise FloatingPointError('Неконечный вход/выход политики')
                target = action*cfg['action_scale']+default
                rows.append([t, *data.qpos[:3], *rpy, *data.qvel[:3],
                             int(blocked['joints']), int(blocked['imu']), ages['joints'], ages['imu'],
                             gate.queues['joints'].latest.sample_time, gate.queues['imu'].latest.sample_time,
                             int(fall_time is not None)])
                if args.sandbox:
                    rows[-1] += [float(cmd[0]), int(controls.method == 'M'), resets]
                total_calls += 1
                max_ages = np.maximum(max_ages, [ages[s] for s in STREAMS])
                for k, v in [('obs', obs), ('actions', action), ('qpos', data.qpos),
                             ('qvel', data.qvel), ('ages_s', [ages[s] for s in STREAMS])]:
                    signals[k].append(np.array(v, copy=True))
                if window:
                    if step % (4*decimation) == 0:
                        label = 'M2 (history)' if history_adapter and controls.method == 'M' else controls.method
                        window.draw(model, data, ages, blocked, label, fall_time is not None)
                    deadline += dt*decimation
                    remaining = deadline-time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    # Keep the wall-clock deadline anchored to episode start:
                    # non-rendering ticks catch up after a costly render frame.
                    # Physics still advances by exactly dt, never a variable step.
                if args.sandbox and len(rows) >= 1000:
                    flush_chunk()
    except BaseException as exc:
        status = f'error: {type(exc).__name__}: {exc}'
        raise
    finally:
        elapsed = time.perf_counter()-started
        if args.sandbox:
            flush_chunk()
        else:
            with (output/'trajectory.csv').open('w') as f:
                writer = csv.writer(f)
                writer.writerow(names)
                writer.writerows(rows)
            np.savez_compressed(output/'signals.npz', **{k: np.asarray(v) for k, v in signals.items()})
            (output/'key_events.json').write_text(json.dumps(events, indent=2)+'\n')
        summary = {'status': status, 'simulation_seconds': float(data.time), 'wall_seconds': elapsed,
                   'compensator': args.compensator, 'm2_profile': profile,
                   'm2_profile_sha256': sha(profile_path) if profile else None,
                   'method': controls.method, 'initial_method': args.method, 'sandbox': args.sandbox,
                   'reset_count': resets, 'fall_history_s': fall_history,
                   'duration_limit_s': duration, 'horizons_ms': {'joints': 20, 'imu': 20}, 'headless': args.headless,
                   'scripted_blocks': blocks, 'command': cmd.tolist(), 'fall_time_s': fall_time,
                   'displacement_xy_m': (data.qpos[:2]-initial[:2]).tolist(),
                   'dropped_packets': {s: dropped_total[s]+gate.dropped[s] for s in STREAMS},
                   'max_ages_s': dict(zip(STREAMS, max_ages.tolist())) if total_calls else None,
                   'policy_calls': total_calls, 'mujoco_warnings': [int(w.number) for w in data.warning],
                   'argv': sys.argv, 'python': platform.python_version(), 'torch': torch.__version__,
                   'mujoco': mujoco.__version__, 'numpy': np.__version__, 'cuda_build': torch.version.cuda,
                   'cpu': Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].strip(' :\t'),
                   'cpu_affinity': sorted(os.sched_getaffinity(0)), 'torch_threads': torch.get_num_threads(),
                   'thread_environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
                   'upstream_commit': subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip(),
                   'checkpoint_sha256': sha(policy_path), 'config_sha256': sha(cfg_path),
                   'asset_sha256': {str(f.relative_to(upstream)): sha(f) for f in sorted((upstream/'resources/robots/h1').rglob('*')) if f.is_file()},
                   'source_sha256': {f.name: sha(f) for f in (Path(__file__), root/'scripts/sensor_age.py', root/'scripts/run_h1_baseline.py',
                       root/'scripts/hybrid_compensator.py', root/'scripts/history_compensator.py', root/'scripts/periodic_compensator.py',
                       root/'scripts/cycle_compensator.py', root/'scripts/conservative_compensator.py')},
                   'fresh_local_pd': True, 'signal_stream_order': list(STREAMS)}
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(f'Логи: {output.resolve()}', flush=True)
        if window:
            if status == 'completed' and not args.auto_close and not args.sandbox:
                print('20 секунд завершены. Итоговый кадр: ЛКМ вращает камеру, Esc закрывает окно.', flush=True)
                try:
                    while window.poll():
                        label = 'M2 (history)' if history_adapter and controls.method == 'M' else controls.method
                        window.draw(model, data, ages, blocked, label, fall_time is not None, True)
                        time.sleep(.02)
                finally:
                    window.close()
            else:
                window.close()


if __name__ == '__main__':
    main()
