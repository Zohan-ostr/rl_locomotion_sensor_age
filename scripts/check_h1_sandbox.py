import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glfw
import mujoco
import numpy as np
import torch
import yaml
from interactive_h1 import DemoControls, Window, reset_robot


class SandboxTests(unittest.TestCase):
    def test_switches_ignore_repeat_and_release(self):
        window = Window.__new__(Window)
        window.glfw, window.controls = glfw, DemoControls('B1')
        for key in (glfw.KEY_W, glfw.KEY_C, glfw.KEY_R):
            for event in (glfw.PRESS, glfw.REPEAT, glfw.RELEASE):
                window.key(None, key, 0, event, 0)
        self.assertEqual(window.controls.commands, ['w', 'c', 'r'])
        for command in window.controls.commands:
            reset = window.controls.apply(command)
        self.assertTrue(reset)
        self.assertFalse(window.controls.walking)
        self.assertEqual(window.controls.method, 'M')
        window.controls.apply('w')
        window.controls.apply('c')
        self.assertTrue(window.controls.walking)
        self.assertEqual(window.controls.method, 'B1')

    def test_reset_pose_velocity_clock_memory(self):
        root = Path('vendor/unitree_rl_gym').resolve()
        cfg = yaml.safe_load((root/'deploy/deploy_mujoco/configs/h1.yaml').read_text())
        model = mujoco.MjModel.from_xml_path(cfg['xml_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)))
        data = mujoco.MjData(model)
        policy = torch.jit.load(cfg['policy_path'].replace('{LEGGED_GYM_ROOT_DIR}', str(root)), map_location='cpu').eval()
        saved = {k: v.clone() for k, v in policy.named_buffers()}
        with torch.inference_mode():
            policy(torch.ones(1, 41))
            data.qpos[:3] = [12, 3, .1]
            data.qvel[:] = 4
            data.time = 123
            reset_robot(model, data, policy, saved, np.array(cfg['default_angles']))
        np.testing.assert_array_equal(data.qpos[:2], [12, 3])
        self.assertEqual(data.time, 123)
        self.assertEqual(data.qpos[2], model.qpos0[2])
        np.testing.assert_array_equal(data.qpos[3:7], [1, 0, 0, 0])
        np.testing.assert_array_equal(data.qvel, 0)
        for k, v in policy.named_buffers():
            self.assertTrue(torch.equal(v, saved[k]))

    def test_actual_controls_chunks_reset(self):
        root = Path('results/sandbox_h1_checks/final_controls')
        summary = json.loads((root/'summary.json').read_text())
        self.assertEqual(summary['policy_calls'], 1500)
        self.assertEqual(summary['reset_count'], 1)
        self.assertTrue(summary['fall_history_s'])
        self.assertLess(summary['fall_history_s'][0], 8)
        self.assertIsNone(summary['fall_time_s'])
        self.assertGreater(summary['simulation_seconds'], 29.99)
        files = sorted((root/'chunks').glob('*.csv'))
        self.assertEqual(len(files), 2)
        rows = np.concatenate([np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True)) for f in files])
        self.assertEqual(len(rows), 1500)
        self.assertTrue(np.all(np.diff(rows['time_s']) > 0))
        self.assertTrue(np.all(rows['cmd_vx_m_s'][(rows['time_s'] > 8) & (rows['time_s'] <= 12)] == 0))
        self.assertTrue(np.all(rows['compensation'][rows['time_s'] > 3] == 1))
        after = rows[np.isclose(rows['time_s'], 8.02)][0]
        self.assertEqual(after['fallen'], 0)
        self.assertEqual(after['reset_count'], 1)
        self.assertEqual(after['age_imu_s'], 0)
        with np.load(root/'chunks/000000.npz') as values:
            # Fresh previous-action slots and a restarted phase after R.
            obs = values['obs'][400]
            np.testing.assert_array_equal(obs[29:39], 0)
            np.testing.assert_allclose(obs[-2:], [np.sin(2*np.pi*.02/.8), np.cos(2*np.pi*.02/.8)], rtol=1e-6)
        baseline = json.loads(Path('results/baseline_h1/headless20/summary.json').read_text())
        bounded = json.loads(Path('results/sandbox_h1_checks/bounded/summary.json').read_text())
        np.testing.assert_array_equal(bounded['displacement_xy_m'], baseline['displacement_xy_m'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
