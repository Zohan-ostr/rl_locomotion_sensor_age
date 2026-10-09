"""Independent checks of coordinate convention and saved baseline artifacts."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline', root/'scripts/run_h1_baseline.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = {}
errors = []
# MuJoCo quaternion rotation gives body-to-world; projected world gravity is R.T @ g.
for quat in ([1., 0, 0, 0], [np.cos(np.pi/4), 0, np.sin(np.pi/4), 0],
             [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0], [0.5, 0.5, 0.5, 0.5]):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.array(quat))
    errors.append(float(np.max(np.abs(module.gravity(quat)-mat.reshape(3, 3).T @ [0, 0, -1]))))
assert max(errors) < 1e-12
checks['gravity_matches_mujoco_rotation_max_error'] = max(errors)
assert np.allclose(module.gravity([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]), [1, 0, 0])
assert np.allclose(module.gravity([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]), [0, -1, 0])
checks['gravity_pitch_and_roll_sign'] = 'passed'
folder = root/'results/baseline_h1/headless20'
summary = json.loads((folder/'summary.json').read_text())
data = np.genfromtxt(folder/'trajectory.csv', delimiter=',', names=True)
assert len(data) == 1000 and summary['simulation_steps'] == 10000
assert np.allclose(np.diff(data['time_s']), 0.02, atol=1e-10, rtol=0)
assert abs(data['time_s'][-1]-20) < 1e-9
assert all(np.isfinite(data[k]).all() for k in data.dtype.names)
assert summary['torch_cuda_build'] is None
assert summary['shape_probe_memory_restored']
assert summary['initial_policy_buffers']['hidden_state']['max_abs'] == 0
assert summary['initial_policy_buffers']['cell_state']['max_abs'] == 0
checks['lstm_initial_memory_zero_and_restored_after_shape_probe'] = 'passed'
assert summary['input_shape'] == [1, 41] and summary['output_shape'] == [1, 10]
assert not any(summary['mujoco_warnings']) and summary['fall_time_s'] is None
assert summary['displacement_xy_m'][0] > 1
assert all(summary['floor_contact_onsets'][k] > 2 for k in ('left_ankle_link', 'right_ankle_link'))
assert summary['floor_contact_steps']['pelvis'] == 0
checks['trajectory_and_walking_checks'] = 'passed: finite values, 10000 steps, 1000 calls, 20 s, walking contacts, no fall/warnings'
video_folder = root/'results/baseline_h1/video20'
if (video_folder/'summary.json').exists():
    video = np.genfromtxt(video_folder/'trajectory.csv', delimiter=',', names=True)
    physical = [k for k in data.dtype.names if k not in ('inference_ms', 'control_cycle_ms')]
    difference = max(float(np.max(np.abs(data[k]-video[k]))) for k in physical)
    assert difference == 0
    checks['video_and_headless_physical_trajectory_max_difference'] = difference
print(json.dumps(checks, indent=2))
