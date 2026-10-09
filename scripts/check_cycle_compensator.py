import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sensor_age import Packet
from cycle_compensator import CycleCompensator


class CycleTests(unittest.TestCase):
    def filled(self):
        m = CycleCompensator([[-10, 10]], dict.fromkeys(('joints', 'imu'), .12))
        for step in range(0, 501, 10):
            t = step*.002
            angle = .1*np.sin(2*np.pi*t/.8)
            dq = .1*2*np.pi/.8*np.cos(2*np.pi*t/.8)
            m.update({'joints': Packet('joints', step, step, {'q': [angle], 'dq': [dq]}),
                      'imu': Packet('imu', step, step, {'g': [np.sin(angle), 0, -np.cos(angle)], 'omega': [0, dq, 0]})}, step)
        return m

    def test_periodic_replay_and_gravity_direction(self):
        m = self.filled()
        out = m.predict(540)
        angle = .1*np.sin(2*np.pi*1.08/.8)
        np.testing.assert_allclose(out['joints']['q'], [angle], atol=1e-12)
        np.testing.assert_allclose(out['imu']['g'], [np.sin(angle), 0, -np.cos(angle)], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(out['imu']['g']), 1.)

    def test_identity_cap_and_only_delivered_history(self):
        m = self.filled()
        for s in ('joints', 'imu'):
            for k, value in m.history[s][-1].payload.items():
                np.testing.assert_array_equal(m.predict(500)[s][k], value)
                np.testing.assert_array_equal(m.predict(560)[s][k], m.predict(10000)[s][k])
        future = Packet('imu', 600, 700, {'g': [0, 0, -1], 'omega': [0, 0, 0]})
        with self.assertRaises(ValueError):
            m.update({'joints': m.history['joints'][-1], 'imu': future}, 600)
        m.update({s: m.history[s][0] for s in ('joints', 'imu')}, 500)
        self.assertEqual(m.history['imu'][-1].sample_step, 500)


if __name__ == '__main__':
    unittest.main(verbosity=2)
