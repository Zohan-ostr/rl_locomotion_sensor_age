"""Check real demonstration logs and key/camera behavior without a display."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import glfw
import numpy as np
from interactive_h1 import SensorGate, Window


class DemoTests(unittest.TestCase):
    def test_hold_both_release_and_initial_sample(self):
        gate = SensorGate(.002)
        truth = {'joints': {'q': [1.], 'dq': [2.]}, 'imu': {'g': [0, 0, -1], 'omega': [0, 0, 0]}}
        both = {'joints': True, 'imu': True}
        gate.sample(0, truth, both)
        for step in (10, 20, 30):
            delivered, ages = gate.sample(step, truth, both)
            self.assertEqual(ages, dict.fromkeys(both, step*.002))
            np.testing.assert_array_equal(delivered['joints']['q'], [1.])
        _, ages = gate.sample(40, truth, {'joints': False, 'imu': True})
        self.assertEqual(ages, {'joints': 0, 'imu': .08})
        _, ages = gate.sample(50, truth, dict.fromkeys(both, False))
        self.assertEqual(ages, dict.fromkeys(both, 0))
        self.assertEqual(gate.dropped, {'joints': 3, 'imu': 4})

    def test_keyboard_press_repeat_release_focus(self):
        window = Window.__new__(Window)
        window.glfw = glfw
        window.held = dict.fromkeys(('joints', 'imu'), False)
        for key in (glfw.KEY_I, glfw.KEY_J):
            window.key(None, key, 0, glfw.PRESS, 0)
            window.key(None, key, 0, glfw.REPEAT, 0)
        self.assertTrue(all(window.held.values()))
        window.key(None, glfw.KEY_I, 0, glfw.RELEASE, 0)
        self.assertEqual(window.held, {'joints': True, 'imu': False})
        window.focus(None, False)
        self.assertFalse(any(window.held.values()))

    def test_orbit_does_not_pan_or_zoom(self):
        import mujoco
        window = Window.__new__(Window)
        class Mouse:
            MOUSE_BUTTON_LEFT, PRESS = 0, 1
            def get_mouse_button(self, *args):
                return self.PRESS
        window.glfw, window.cursor = Mouse(), (10, 20)
        window.camera = mujoco.MjvCamera()
        window.camera.distance = 3.5
        window.camera.lookat[:] = [1, 2, 3]
        window.mouse(None, 30, 40)
        self.assertEqual(window.camera.distance, 3.5)
        np.testing.assert_array_equal(window.camera.lookat, [1, 2, 3])
        self.assertNotEqual(window.camera.azimuth, 0)

    def test_actual_runs(self):
        root = Path('results/interactive_h1_checks')
        clean = json.loads((root/'clean/summary.json').read_text())
        self.assertEqual(clean['policy_calls'], 1000)
        self.assertIsNone(clean['fall_time_s'])
        baseline = json.loads(Path('results/baseline_h1/headless20/summary.json').read_text())
        np.testing.assert_array_equal(clean['displacement_xy_m'], baseline['displacement_xy_m'])
        # Compare every recorded physical component, not just final displacement.
        a = np.genfromtxt(root/'clean/trajectory.csv', delimiter=',', names=True)
        b = np.genfromtxt('results/baseline_h1/headless20/trajectory.csv', delimiter=',', names=True)
        for k in ('x_m', 'y_m', 'height_m', 'roll_rad', 'pitch_rad', 'yaw_rad'):
            np.testing.assert_array_equal(a[k], b[k])
        for case in ('blocks', 'compensation', 'window', 'window_fast', 'window_final', 'window_realtime'):
            summary = json.loads((root/case/'summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            if not case.startswith('window'):
                self.assertEqual(summary['dropped_packets'], {'joints': 6, 'imu': 6})
                self.assertEqual(summary['max_ages_s'], {'joints': .12, 'imu': .12})
            self.assertEqual(summary['policy_calls'], 1000)
            self.assertFalse(any(summary['mujoco_warnings']))
            data = np.genfromtxt(root/case/'trajectory.csv', delimiter=',', names=True)
            for s in ('joints', 'imu'):
                self.assertEqual(summary['dropped_packets'][s], int(data['blocked_'+s].sum()))
                np.testing.assert_array_equal(data['age_'+s+'_s'][data['blocked_'+s] == 0], 0)
                self.assertTrue(np.allclose(data['time_s']-data['sample_time_'+s+'_s'], data['age_'+s+'_s']))
        # GUI may contain real human key holds in addition to scripted blocks;
        # its exact schedule is recorded, so it is not a paired headless run.


if __name__ == '__main__':
    unittest.main(verbosity=2)
