import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sensor_age import Packet
from periodic_compensator import PeriodicCompensator


class PeriodicTests(unittest.TestCase):
    def filled(self):
        m = PeriodicCompensator([[-10, 10]], {'joints': .12, 'imu': .12}, {'ridge': 0., 'harmonics': 1})
        w = 2*np.pi/.8
        for step in range(0, 501, 10):
            t = step*.002
            packets = {'joints': Packet('joints', step, step, {'q': [.1*np.sin(w*t)], 'dq': [.1*w*np.cos(w*t)]}),
                       'imu': Packet('imu', step, step, {'g': [0, 0, -1], 'omega': [0, 0, .1*np.sin(w*t)]})}
            m.update(packets, step)
        return m

    def test_periodic_joint_position_and_velocity(self):
        m = self.filled()
        out = m.predict(540)
        w = 2*np.pi/.8
        np.testing.assert_allclose(out['joints']['q'], [.1*np.sin(w*1.08)], atol=1e-12)
        np.testing.assert_allclose(out['joints']['dq'], [.1*w*np.cos(w*1.08)], atol=1e-12)
        np.testing.assert_allclose(out['imu']['omega'], [0, 0, .1*np.sin(w*1.08)], atol=1e-12)

    def test_exact_fresh_and_horizon_cap(self):
        m = self.filled()
        for s in ('joints', 'imu'):
            for k, value in m.history[s][-1].payload.items():
                np.testing.assert_array_equal(m.predict(500)[s][k], value)
            for k in m.predict(560)[s]:
                np.testing.assert_array_equal(m.predict(560)[s][k], m.predict(10000)[s][k])

    def test_causality_and_old_packet(self):
        m = self.filled()
        packet = Packet('joints', 600, 700, {'q': [9], 'dq': [9]})
        with self.assertRaises(ValueError):
            m.update({'joints': packet}, 600)
        packets = {s: m.history[s][0] for s in ('joints', 'imu')}
        m.update(packets, 500)
        self.assertEqual(m.history['joints'][-1].sample_step, 500)


if __name__ == '__main__':
    unittest.main(verbosity=2)
