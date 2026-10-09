import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sensor_age import Packet
from conservative_compensator import ConservativeCompensator


class ConfidenceTests(unittest.TestCase):
    def predictor(self):
        m = ConservativeCompensator([[-1, 1]], dict.fromkeys(('joints', 'imu'), .02))
        packets = {'joints': Packet('joints', 0, 0, {'q': [0.], 'dq': [1.]}),
                   'imu': Packet('imu', 0, 0, {'g': [0, 0, -1], 'omega': [0, 1, 0]})}
        m.update(packets, 0)
        return m, packets

    def test_identity_short_age_and_smooth_fade(self):
        m, packets = self.predictor()
        for s in packets:
            for k, v in packets[s].payload.items():
                np.testing.assert_array_equal(m.predict(0)[s][k], v)
        self.assertAlmostEqual(m.predict(10)['joints']['q'][0], .005)
        self.assertGreater(m.predict(20)['joints']['q'][0], m.predict(40)['joints']['q'][0])
        self.assertLess(abs(m.predict(1000)['joints']['q'][0]), 1e-20)
        np.testing.assert_array_equal(m.predict(10)['joints']['dq'], [1])
        np.testing.assert_allclose(m.predict(10)['imu']['g'], [np.sin(.02), 0, -np.cos(.02)])

    def test_delivered_only_and_no_reordering_regression(self):
        m, packets = self.predictor()
        newer = {s: Packet(s, 10, 10, p.payload) for s, p in packets.items()}
        m.update(newer, 10)
        m.update(packets, 20)
        self.assertEqual(m.latest['joints'].sample_step, 10)
        with self.assertRaises(ValueError):
            m.update(newer, 9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
