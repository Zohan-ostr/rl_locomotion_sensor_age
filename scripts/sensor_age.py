"""Causal sensor transport and bounded prediction in physical units.

No MuJoCo state or policy object is available to the prediction functions.
Time is represented by integer physics steps to avoid ambiguous float ties.
"""
from dataclasses import dataclass
import heapq

import numpy as np


STREAMS = ('joints', 'imu')
METHODS = ('B0', 'B1', 'M', 'B2', 'AJ', 'AI', 'M2', 'M3')


def frozen_payload(payload):
    result = {}
    for key, value in payload.items():
        array = np.array(value, dtype=np.float64, copy=True)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite sensor value')
        array.flags.writeable = False
        result[key] = array
    return result


@dataclass(frozen=True)
class Packet:
    stream: str
    sample_step: int
    delivery_step: int
    payload: dict
    dt: float = 0.002

    def __post_init__(self):
        if self.stream not in STREAMS or self.sample_step < 0 or self.delivery_step < self.sample_step:
            raise ValueError('Invalid packet timestamps or stream')
        object.__setattr__(self, 'payload', frozen_payload(self.payload))

    @property
    def sample_time(self):
        return self.sample_step*self.dt

    @property
    def delivery_time(self):
        return self.delivery_step*self.dt


class PacketQueue:
    def __init__(self, stream):
        if stream not in STREAMS:
            raise ValueError(stream)
        self.stream = stream
        self._pending = []
        self.latest = None
        self._serial = 0
        self._now = -1
        self.accepted = self.rejected = self.arrived = 0

    def enqueue(self, packet, now_step):
        if packet.stream != self.stream or packet.sample_step > now_step:
            raise ValueError('Wrong stream or future sample injected')
        if now_step < self._now:
            raise ValueError('Clock moved backwards')
        heapq.heappush(self._pending, (packet.delivery_step, packet.sample_step, self._serial, packet))
        self._serial += 1

    def deliver(self, now_step):
        if now_step < self._now:
            raise ValueError('Clock moved backwards')
        self._now = now_step
        events = []
        while self._pending and self._pending[0][0] <= now_step:
            packet = heapq.heappop(self._pending)[3]
            assert packet.sample_step <= packet.delivery_step <= now_step
            self.arrived += 1
            accepted = self.latest is None or packet.sample_step > self.latest.sample_step
            if accepted:
                self.latest = packet
                self.accepted += 1
            else:
                self.rejected += 1
            events.append((packet, accepted))
        if self.latest is None:
            raise RuntimeError('No delivered initial sensor packet')
        return self.latest, events


def predict_joints(payload, age, horizon, limits):
    q, dq = payload['q'], payload['dq']
    if age < 0 or horizon < 0:
        raise ValueError('Negative age or horizon')
    if age == 0:
        return {'q': q.copy(), 'dq': dq.copy()}
    tau = min(age, horizon)
    return {'q': np.clip(q+dq*tau, limits[:, 0], limits[:, 1]), 'dq': dq.copy()}


def predict_imu(payload, age, horizon):
    g, omega = payload['g'], payload['omega']
    if age < 0 or horizon < 0:
        raise ValueError('Negative age or horizon')
    if age == 0:
        return {'g': g.copy(), 'omega': omega.copy()}
    rotation = -omega*min(age, horizon)
    theta = float(np.linalg.norm(rotation))
    if theta == 0:
        predicted = g.copy()
    else:
        axis = rotation/theta
        predicted = (g*np.cos(theta) + np.cross(axis, g)*np.sin(theta)
                     + axis*np.dot(axis, g)*(1-np.cos(theta)))
    norm = np.linalg.norm(predicted)
    if norm == 0:
        raise ValueError('Zero gravity direction')
    return {'g': predicted/norm, 'omega': omega.copy()}


def adapt(delivered, ages, method, horizons, limits):
    """Only delivered payloads, ages, and fixed constants enter this boundary."""
    if method not in ('B1', 'M', 'B2', 'AJ', 'AI'):
        raise ValueError('B0 is handled by the physics-side sampler')
    effective = dict(ages)
    if method == 'B2':
        effective = dict.fromkeys(STREAMS, max(ages.values()))
    compensate_j = method in ('M', 'B2', 'AJ')
    compensate_i = method in ('M', 'B2', 'AI')
    joints = (predict_joints(delivered['joints'], effective['joints'], horizons['joints'], limits)
              if compensate_j else {k: v.copy() for k, v in delivered['joints'].items()})
    imu = (predict_imu(delivered['imu'], effective['imu'], horizons['imu'])
           if compensate_i else {k: v.copy() for k, v in delivered['imu'].items()})
    return {'joints': joints, 'imu': imu}


def make_schedule(seed, condition, duration=20.0, dt=0.002, decimation=10, warmup=2.0):
    """Generate delivery/drop times in advance, independent of episode state."""
    steps = round(duration/dt)
    warmup_step = round(warmup/dt)
    if abs(steps*dt-duration) > 1e-9 or steps % decimation:
        raise ValueError('Duration must be a whole number of control periods')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20261008]))
    onset_index = int(rng.integers(round(5/(dt*decimation)), round(10/(dt*decimation))+1))
    schedule = {'seed': seed, 'condition': condition, 'duration_s': duration, 'dt': dt,
                'decimation': decimation, 'warmup_s': warmup, 'burst_onset_s': onset_index*dt*decimation,
                'streams': {}}
    for stream in STREAMS:
        entries = []
        for index, step in enumerate(range(0, steps+1, decimation)):
            delay_ms = condition.get(f'delay_{stream}_ms', 0)
            if condition.get('jitter', False):
                delay_ms = int(rng.choice([0, 20, 40]))
            delay_steps = round(delay_ms/1000/dt)
            if abs(delay_steps*dt-delay_ms/1000) > 1e-9 or delay_steps < 0:
                raise ValueError('Delay must be nonnegative and quantized to physics dt')
            drop = (condition.get('drop_stream') == stream
                    and onset_index <= index < onset_index+condition.get('missing_packets', 0))
            if step <= warmup_step:
                delay_steps, drop = 0, False
            entries.append({'sample_step': step, 'delivery_step': step+delay_steps,
                            'sample_time': step*dt, 'delivery_time': (step+delay_steps)*dt,
                            'drop': bool(drop)})
        schedule['streams'][stream] = entries
    return schedule
