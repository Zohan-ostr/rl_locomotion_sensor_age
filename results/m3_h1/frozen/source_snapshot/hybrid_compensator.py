"""Independent causal predictors for joints and IMU, with fixed profiles."""
import numpy as np
from sensor_age import STREAMS, predict_joints, predict_imu
from history_compensator import HistoryCompensator
from periodic_compensator import PeriodicCompensator
from cycle_compensator import CycleCompensator
from conservative_compensator import ConservativeCompensator


class LegacyCompensator:
    def __init__(self, limits, horizons, config=None):
        if config:
            raise ValueError('Legacy predictor has no extra parameters')
        self.config, self.latest = {}, {}
        self.limits, self.horizons = np.array(limits, copy=True), dict(horizons)

    def update(self, packets, now_step):
        for s in STREAMS:
            p = packets[s]
            if p.stream != s or p.delivery_step > now_step:
                raise ValueError('Only delivered packets')
            if s not in self.latest or p.sample_step > self.latest[s].sample_step:
                self.latest[s] = p

    def predict(self, now_step):
        out = {}
        for s in STREAMS:
            p = self.latest[s]
            if now_step < p.delivery_step:
                raise ValueError('Prediction before delivery')
            age = (now_step-p.sample_step)*p.dt
            out[s] = (predict_joints(p.payload, age, self.horizons[s], self.limits) if s == 'joints' else
                      predict_imu(p.payload, age, self.horizons[s]))
        return out


class HybridCompensator:
    def __init__(self, limits, horizons, config):
        if set(config) != set(STREAMS):
            raise ValueError('Specify fixed predictors for both streams')
        classes = {'legacy': LegacyCompensator, 'acceleration': HistoryCompensator,
                   'periodic': PeriodicCompensator, 'cycle': CycleCompensator,
                   'conservative': ConservativeCompensator}
        self.models, self.config = {}, {}
        for s in STREAMS:
            settings = config[s]
            family = settings['family']
            self.models[s] = classes[family](limits, horizons, settings.get('config'))
            self.config[s] = {'family': family, 'config': self.models[s].config}

    def update(self, packets, now_step):
        for model in self.models.values():
            model.update(packets, now_step)

    def predict(self, now_step):
        return {s: self.models[s].predict(now_step)[s] for s in STREAMS}
