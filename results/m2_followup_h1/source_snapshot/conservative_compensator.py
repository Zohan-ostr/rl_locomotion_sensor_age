"""Bounded legacy prediction with an explicit confidence fade by sample age."""
import numpy as np
from sensor_age import STREAMS, predict_joints, predict_imu


class ConservativeCompensator:
    def __init__(self, limits, horizons, config=None):
        self.config = dict(joint_gain=.25, imu_gain=1., fade_after=.04, fade_scale=.04)
        if config:
            if set(config)-set(self.config):
                raise ValueError('Unknown confidence parameters')
            self.config.update(config)
        if any(not np.isfinite(v) or v < 0 for v in self.config.values()) or self.config['fade_scale'] <= 0:
            raise ValueError('Invalid confidence parameters')
        self.limits, self.horizons = np.array(limits, copy=True), dict(horizons)
        self.latest = {}

    def update(self, packets, now_step):
        for s in STREAMS:
            p = packets[s]
            if p.stream != s or p.delivery_step > now_step:
                raise ValueError('Only delivered packets')
            if s not in self.latest or p.sample_step > self.latest[s].sample_step:
                self.latest[s] = p

    def predict(self, now_step):
        result = {}
        c = self.config
        for s in STREAMS:
            p = self.latest[s]
            if now_step < p.delivery_step:
                raise ValueError('Prediction before delivery')
            age = (now_step-p.sample_step)*p.dt
            confidence = np.exp(-max(0., age-c['fade_after'])/c['fade_scale'])
            tau = min(age, self.horizons[s])*confidence*c['joint_gain' if s == 'joints' else 'imu_gain']
            # For fresh data the legacy functions copy exactly.
            result[s] = (predict_joints(p.payload, tau, tau, self.limits) if s == 'joints' else
                         predict_imu(p.payload, tau, tau))
        return result
