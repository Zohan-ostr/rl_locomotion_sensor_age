import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sensor_age import Packet
from hybrid_compensator import HybridCompensator


class HybridTests(unittest.TestCase):
    def test_distinct_models_horizons_and_fresh_identity(self):
        m = HybridCompensator([[-1, 1]], {'joints': .12, 'imu': .02},
            {'joints': {'family': 'periodic', 'config': {'harmonics': 1, 'ridge': 0., 'gain': .5}},
             'imu': {'family': 'legacy', 'config': {}}})
        w = 2*np.pi/.8
        for step in range(0, 501, 10):
            t = step*.002
            packets = {'joints': Packet('joints', step, step, {'q': [.1*np.sin(w*t)], 'dq': [.1*w*np.cos(w*t)]}),
                       'imu': Packet('imu', step, step, {'g': [0, 0, -1], 'omega': [0, 1, 0]})}
            m.update(packets, step)
        for s in packets:
            for k, v in packets[s].payload.items():
                np.testing.assert_array_equal(m.predict(500)[s][k], v)
        result = m.predict(540)
        expected_q = .1*np.sin(w)+.5*(.1*np.sin(w*1.08)-.1*np.sin(w))
        np.testing.assert_allclose(result['joints']['q'], [expected_q], atol=1e-12)
        np.testing.assert_allclose(result['imu']['g'], [np.sin(.02), 0, -np.cos(.02)], atol=1e-12)

    def test_missing_and_future_data_are_rejected(self):
        m = HybridCompensator([[-1, 1]], {'joints': .12, 'imu': .02},
            {'joints': {'family': 'legacy'}, 'imu': {'family': 'legacy'}})
        packets = {'joints': Packet('joints', 0, 10, {'q': [0], 'dq': [0]}),
                   'imu': Packet('imu', 0, 10, {'g': [0, 0, -1], 'omega': [0, 0, 0]})}
        with self.assertRaises(ValueError):
            m.update(packets, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
