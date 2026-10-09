"""Anchored replay of observed within-cycle changes; delivered history only."""
from collections import deque
import numpy as np
from sensor_age import STREAMS


class CycleCompensator:
    def __init__(self, limits, horizons, config=None):
        self.config = dict(period=.8, gain=1., history=100, joint_correction_limit=.35,
                           velocity_correction_limit=6., omega_correction_limit=2.)
        if config:
            if set(config)-set(self.config):
                raise ValueError('Unknown cycle parameters')
            self.config.update(config)
        if self.config['period'] <= 0 or not isinstance(self.config['history'], int) or self.config['history'] < 2:
            raise ValueError('Invalid period/history')
        if any(not np.isfinite(v) or v < 0 for v in self.config.values()):
            raise ValueError('Invalid cycle parameters')
        self.limits, self.horizons = np.array(limits, copy=True), dict(horizons)
        if any(not np.isfinite(v) or v < 0 for v in self.horizons.values()):
            raise ValueError('Invalid horizons')
        self.history = {s: deque(maxlen=self.config['history']) for s in STREAMS}

    def update(self, packets, now_step):
        for s in STREAMS:
            p = packets[s]
            if p.stream != s or p.delivery_step > now_step:
                raise ValueError('Only delivered measurements')
            h = self.history[s]
            if not h or p.sample_step > h[-1].sample_step:
                h.append(p)

    def predict(self, now_step):
        out = {}
        c = self.config
        for s in STREAMS:
            h = self.history[s]
            if not h:
                raise RuntimeError('Missing initial measurement')
            p = h[-1]
            if now_step < p.delivery_step:
                raise ValueError('Prediction before delivery')
            out[s] = {k: v.copy() for k, v in p.payload.items()}
            age = (now_step-p.sample_step)*p.dt
            if age == 0:
                continue
            tau = min(age, self.horizons[s])
            start = p.sample_time-c['period']
            end = start+tau
            times = np.array([packet.sample_time for packet in h])
            if start < times[0] or end > times[-1]:
                continue
            for key, values in out[s].items():
                data = np.array([packet.payload[key] for packet in h])
                change = c['gain']*np.array([np.interp(end, times, data[:, j])-np.interp(start, times, data[:, j])
                                             for j in range(data.shape[1])])
                limit = {'q': c['joint_correction_limit'], 'dq': c['velocity_correction_limit'],
                         'omega': c['omega_correction_limit'], 'g': .35}[key]
                out[s][key] = values+np.clip(change, -limit, limit)
            if s == 'joints':
                out[s]['q'] = np.clip(out[s]['q'], self.limits[:, 0], self.limits[:, 1])
            else:
                norm = np.linalg.norm(out[s]['g'])
                out[s]['g'] = out[s]['g']/norm if norm > 1e-12 else p.payload['g'].copy()
        return out
