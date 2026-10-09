"""Required causal/mathematical tests; optional audit of real episode logs."""
import argparse
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_age import Packet, PacketQueue, adapt, make_schedule, predict_imu, predict_joints

import mujoco
import numpy as np


def joints(q=0.0, dq=1.0):
    return {'q': np.array([q]), 'dq': np.array([dq])}


class SensorTests(unittest.TestCase):
    def test_packet_timestamps_and_snapshot(self):
        x = joints()
        p = Packet('joints', 0, 10, x)
        x['q'][0] = 99
        self.assertEqual(p.payload['q'][0], 0)
        with self.assertRaises(ValueError):
            p.payload['q'][0] = 2
        with self.assertRaises(ValueError):
            Packet('joints', 20, 10, x)

    def test_undelivered_packet_never_leaks(self):
        q = PacketQueue('joints')
        q.enqueue(Packet('joints', 0, 0, joints(1)), 0)
        q.deliver(0)
        q.enqueue(Packet('joints', 10, 40, joints(12345)), 10)
        self.assertEqual(q.deliver(39)[0].payload['q'][0], 1)
        self.assertEqual(q.deliver(40)[0].payload['q'][0], 12345)

    def test_future_sample_rejected(self):
        q = PacketQueue('joints')
        with self.assertRaises(ValueError):
            q.enqueue(Packet('joints', 20, 30, joints()), 10)

    def test_reordering_old_arrival_rejected(self):
        q = PacketQueue('joints')
        q.enqueue(Packet('joints', 0, 0, joints()), 0)
        q.deliver(0)
        q.enqueue(Packet('joints', 10, 40, joints(10)), 10)
        q.enqueue(Packet('joints', 20, 20, joints(20)), 20)
        q.deliver(20)
        latest, events = q.deliver(40)
        self.assertEqual(latest.sample_step, 20)
        self.assertFalse(events[0][1])
        self.assertEqual(q.rejected, 1)

    def test_simultaneous_arrivals_select_newest(self):
        q = PacketQueue('joints')
        q.enqueue(Packet('joints', 0, 0, joints()), 0)
        q.deliver(0)
        for step in (10, 20, 30):
            q.enqueue(Packet('joints', step, 40, joints(step)), step)
        self.assertEqual(q.deliver(40)[0].sample_step, 30)

    def test_loss_holds_value_and_age_grows_from_sample_time(self):
        q = PacketQueue('joints')
        q.enqueue(Packet('joints', 0, 0, joints(7)), 0)
        q.deliver(0)
        ages = []
        for now in (10, 20, 30):
            p, _ = q.deliver(now)
            ages.append((now-p.sample_step)*p.dt)
            self.assertEqual(p.payload['q'][0], 7)
        np.testing.assert_allclose(ages, [.02, .04, .06], atol=0, rtol=0)
        q.enqueue(Packet('joints', 40, 50, joints(8)), 40)
        p, _ = q.deliver(50)
        self.assertEqual((50-p.sample_step)*p.dt, .02)

    def test_clock_reversal_rejected(self):
        q = PacketQueue('joints')
        q.enqueue(Packet('joints', 0, 0, joints()), 0)
        q.deliver(0)
        q.deliver(10)
        with self.assertRaises(ValueError):
            q.deliver(9)

    def test_zero_age_is_exact_identity_even_outside_limits(self):
        delivered = {'joints': joints(2, 5), 'imu': {'g': np.array([.01, 0, -.999]), 'omega': np.array([1., 2, 3])}}
        limits = np.array([[-1., 1.]])
        for method in ('B1', 'M', 'B2', 'AJ', 'AI'):
            out = adapt(delivered, {'joints': 0., 'imu': 0.}, method, {'joints': .08, 'imu': .08}, limits)
            for s in delivered:
                for k in delivered[s]:
                    np.testing.assert_array_equal(out[s][k], delivered[s][k])

    def test_constant_joint_velocity_and_limit(self):
        x = joints(.3, 2.)
        out = predict_joints(x, .04, .08, np.array([[-1., 1.]]))
        np.testing.assert_allclose(out['q'], [.38], atol=1e-15)
        np.testing.assert_array_equal(out['dq'], [2.])
        np.testing.assert_allclose(predict_joints(joints(.95, 5), .04, .08, np.array([[-1., 1.]]))['q'], [1.])

    def test_gravity_rotation_sign_and_norm(self):
        out = predict_imu({'g': np.array([0., 0, -1]), 'omega': np.array([0., 1, 0])}, .2, .2)
        np.testing.assert_allclose(out['g'], [np.sin(.2), 0, -np.cos(.2)], atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(out['g']), 1.)
        omega = np.array([.3, -.7, .5])
        g = np.array([.2, .3, -.9]); g /= np.linalg.norm(g)
        angle = np.linalg.norm(omega)*.08
        quat = np.r_[np.cos(angle/2), omega/np.linalg.norm(omega)*np.sin(angle/2)]
        mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat)
        out = predict_imu({'g': g, 'omega': omega}, .08, .08)
        np.testing.assert_allclose(out['g'], mat.reshape(3, 3).T @ g, atol=1e-15)

    def test_horizon_cap_does_not_return_to_old_value(self):
        x = joints(.2, 1.)
        at_cap = predict_joints(x, .08, .08, np.array([[-1., 1.]]))
        later = predict_joints(x, 5., .08, np.array([[-1., 1.]]))
        np.testing.assert_array_equal(at_cap['q'], later['q'])
        self.assertNotEqual(later['q'][0], x['q'][0])
        i = {'g': np.array([0., 0, -1]), 'omega': np.array([1., 0, 0])}
        np.testing.assert_array_equal(predict_imu(i, .08, .08)['g'], predict_imu(i, 5., .08)['g'])

    def test_per_stream_age_and_payload_not_mutated(self):
        x = {'joints': joints(.2, 1.), 'imu': {'g': np.array([0., 0, -1]), 'omega': np.array([0., 1, 0])}}
        out = adapt(x, {'joints': .04, 'imu': 0.}, 'M', {'joints': .08, 'imu': .08}, np.array([[-1., 1.]]))
        self.assertAlmostEqual(out['joints']['q'][0], .24)
        np.testing.assert_array_equal(out['imu']['g'], x['imu']['g'])
        np.testing.assert_array_equal(x['joints']['q'], [.2])

    def test_schedule_reproducible_warmup_and_loss(self):
        c = {'id': 'drop', 'drop_stream': 'imu', 'missing_packets': 6}
        a, b = make_schedule(0, c), make_schedule(0, c)
        self.assertEqual(a, b)
        self.assertEqual(sum(e['drop'] for e in a['streams']['imu']), 6)
        self.assertEqual(sum(e['drop'] for e in a['streams']['joints']), 0)
        for e in a['streams']['imu']:
            if e['sample_time'] <= 2:
                self.assertFalse(e['drop'])
                self.assertEqual(e['sample_step'], e['delivery_step'])

    def test_jitter_reorders_and_respects_causality(self):
        a = make_schedule(0, {'id': 'E4', 'jitter': True})
        for stream in ('joints', 'imu'):
            e = a['streams'][stream]
            self.assertTrue(all(p['delivery_step'] >= p['sample_step'] for p in e))
            self.assertTrue(any(y['delivery_step'] < x['delivery_step'] for x, y in zip(e, e[1:])))


def audit_episode(folder):
    summary = json.loads((folder/'summary.json').read_text())
    schedule = json.loads(Path(summary['schedule_path']).read_text())
    x = np.load(folder/'signals.npz')
    steps = x['decision_steps']
    selected = x['selected_sample_steps']
    delivery = x['selected_delivery_steps']
    assert np.all(selected <= delivery) and np.all(delivery <= steps[:, None])
    assert np.all(np.diff(selected, axis=0) >= 0)
    np.testing.assert_array_equal(x['ages_s'], (steps[:, None]-selected)*schedule['dt'])
    assert summary['simulation_steps'] <= round(summary['max_episode_seconds']/schedule['dt'])
    assert abs(summary['mujoco_time_s']-summary['simulation_seconds']) < 1e-8
    assert summary['policy_calls'] == len(steps)
    assert all(np.isfinite(x[k]).all() for k in x.files)
    for i, stream in enumerate(('joints', 'imu')):
        for row, selected_step in enumerate(selected[:, i]):
            entry = schedule['streams'][stream][selected_step//schedule['decimation']]
            assert not entry['drop']
            assert entry['delivery_step'] == delivery[row, i]
        for name in (('q', 'dq') if stream == 'joints' else ('g', 'omega')):
            np.testing.assert_array_equal(x[name+'_delivered'], x[name+'_true'][selected[:, i]//schedule['decimation']])
    if summary['method'] in ('M', 'M2', 'AJ', 'AI'):
        for name, i in [('q', 0), ('dq', 0), ('g', 1), ('omega', 1)]:
            zero = x['ages_s'][:, i] == 0
            np.testing.assert_array_equal(x[name+'_predicted'][zero], x[name+'_delivered'][zero])
    return {'folder': str(folder), 'decisions': len(steps), 'causality_and_payload_checks': 'passed'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit', type=Path, nargs='*', default=[])
    args = p.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SensorTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    audits = [audit_episode(folder) for folder in args.audit]
    print(json.dumps({'unit_checks': result.testsRun, 'passed': True, 'episode_audits': audits}, indent=2))


if __name__ == '__main__':
    main()
