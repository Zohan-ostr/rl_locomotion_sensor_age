import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sensor_age import Packet
from history_compensator import HistoryCompensator


class HistoryTests(unittest.TestCase):
    def predictor(self, horizon=.08):
        return HistoryCompensator([[-10, 10]], dict.fromkeys(('joints', 'imu'), horizon))

    def packets(self, step, dq=1., q=.2, omega=(0, 0, 0), delay=0):
        return {'joints': Packet('joints', step, step+delay, {'q': [q], 'dq': [dq]}),
                'imu': Packet('imu', step, step+delay, {'g': [0, 0, -1], 'omega': omega})}

    def test_zero_age_exact_and_no_mutation(self):
        m = self.predictor()
        p = self.packets(0, q=20)
        m.update(p, 0)
        out = m.predict(0)
        for s in p:
            for key in p[s].payload:
                np.testing.assert_array_equal(out[s][key], p[s].payload[key])
        out['joints']['q'][0] = 1
        self.assertEqual(p['joints'].payload['q'][0], 20)

    def test_constant_velocity(self):
        m = self.predictor()
        for step in (0, 10, 20):
            m.update(self.packets(step, dq=2., q=.2+2*step*.002), step)
        out = m.predict(40)
        np.testing.assert_allclose(out['joints']['q'], [.36])
        np.testing.assert_allclose(out['joints']['dq'], [2])

    def test_constant_acceleration_irregular_timestamps(self):
        m = self.predictor()
        for step in (0, 10, 30):
            t = step*.002
            m.update(self.packets(step, dq=1+3*t, q=.2+t+1.5*t*t), step)
        out = m.predict(50)
        np.testing.assert_allclose(out['joints']['q'], [.315])
        np.testing.assert_allclose(out['joints']['dq'], [1.3])

    def test_delayed_and_old_packets_never_leak(self):
        m = self.predictor()
        with self.assertRaises(ValueError):
            m.update(self.packets(10, delay=20), 20)
        m.update(self.packets(20, delay=10), 30)
        m.update(self.packets(10, q=9), 30)
        self.assertEqual(m.history['joints'][-1].sample_step, 20)
        with self.assertRaises(ValueError):
            m.predict(29)

    def test_drop_age_cap_and_bounded_history(self):
        m = self.predictor()
        for step in range(0, 100, 10):
            m.update(self.packets(step), step)
        for step in (100, 110):
            m.update(self.packets(90), step)
        self.assertEqual(len(m.history['joints']), 5)
        self.assertLess(m.predict(100)['joints']['q'][0], m.predict(110)['joints']['q'][0])
        for s in ('joints', 'imu'):
            for key in m.predict(130)[s]:
                np.testing.assert_array_equal(m.predict(130)[s][key], m.predict(1000)[s][key])

    def test_gravity_sign_norm_and_angular_acceleration(self):
        m = self.predictor()
        m.update(self.packets(0, omega=(0, 1, 0)), 0)
        g = m.predict(10)['imu']['g']
        np.testing.assert_allclose(g, [np.sin(.02), 0, -np.cos(.02)], atol=1e-12)
        m.update(self.packets(10, omega=(0, 1.04, 0)), 10)
        out = m.predict(20)['imu']
        np.testing.assert_allclose(out['omega'], [0, 1.08, 0])
        np.testing.assert_allclose(out['g'], [np.sin(.0212), 0, -np.cos(.0212)], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(out['g']), 1.)

    def test_correction_and_joint_limits(self):
        m = self.predictor()
        m.update(self.packets(0, dq=1000., q=9.9), 0)
        out = m.predict(40)
        self.assertEqual(out['joints']['q'][0], 10.)
        self.assertEqual(out['joints']['dq'][0], 1000.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
