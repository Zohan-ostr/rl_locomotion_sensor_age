"""Causal local-acceleration predictor. No simulator/policy/commands access."""
from collections import deque
import numpy as np
from sensor_age import STREAMS, predict_imu


class HistoryCompensator:
    def __init__(self, limits, horizons, config=None):
        self.config = dict(window=5, acceleration_limit=100., angular_acceleration_limit=50.,
                           velocity_gain=1., acceleration_gain=1., joint_correction_limit=.35,
                           velocity_correction_limit=6., omega_correction_limit=2.)
        if config:
            if set(config)-set(self.config):
                raise ValueError('Unknown predictor configuration')
            self.config.update(config)
        c = self.config
        if not isinstance(c['window'], int) or c['window'] < 2:
            raise ValueError('window must be an integer >=2')
        if any(not np.isfinite(v) or v < 0 for k, v in c.items() if k != 'window'):
            raise ValueError('Invalid predictor coefficient')
        self.limits = np.array(limits, copy=True)
        self.horizons = dict(horizons)
        if any(not np.isfinite(self.horizons[s]) or self.horizons[s] < 0 for s in STREAMS):
            raise ValueError('Invalid horizons')
        self.history = {s: deque(maxlen=c['window']) for s in STREAMS}

    def update(self, packets, now_step):
        for s in STREAMS:
            packet = packets[s]
            if packet.stream != s or packet.sample_step > packet.delivery_step or packet.delivery_step > now_step:
                raise ValueError('Only delivered measurements are permitted')
            history = self.history[s]
            if history and packet.sample_step <= history[-1].sample_step:
                continue  # Old arrivals and duplicate holds do not enter the fit.
            history.append(packet)

    def acceleration(self, stream, key, limit):
        history = self.history[stream]
        if len(history) < 2:
            return np.zeros_like(history[-1].payload[key])
        times = np.array([p.sample_time for p in history])
        times -= times.mean()
        values = np.array([p.payload[key] for p in history])
        # Linear least-squares slope at actual sample timestamps, including gaps.
        slope = np.sum(times[:, None]*values, axis=0)/np.sum(times**2)
        return np.clip(slope, -limit, limit)*self.config['acceleration_gain']

    def predict(self, now_step):
        result = {}
        c = self.config
        for stream in STREAMS:
            if not self.history[stream]:
                raise RuntimeError('Missing initial measurement')
            packet = self.history[stream][-1]
            if now_step < packet.delivery_step:
                raise ValueError('Prediction before delivery')
            age = (now_step-packet.sample_step)*packet.dt
            p = packet.payload
            if age == 0:
                result[stream] = {k: v.copy() for k, v in p.items()}
                continue
            tau = min(age, self.horizons[stream])
            if stream == 'joints':
                a = self.acceleration(stream, 'dq', c['acceleration_limit'])
                correction = np.clip(p['dq']*tau+.5*a*tau*tau,
                                     -c['joint_correction_limit'], c['joint_correction_limit'])
                dv = np.clip(a*tau*c['velocity_gain'], -c['velocity_correction_limit'], c['velocity_correction_limit'])
                result[stream] = {'q': np.clip(p['q']+correction, self.limits[:, 0], self.limits[:, 1]),
                                  'dq': p['dq']+dv}
            else:
                alpha = self.acceleration(stream, 'omega', c['angular_acceleration_limit'])
                # Midpoint angular velocity for Rodrigues integration.
                midpoint = {'g': p['g'], 'omega': p['omega']+.5*alpha*tau}
                g = predict_imu(midpoint, tau, tau)['g']
                dw = np.clip(alpha*tau*c['velocity_gain'], -c['omega_correction_limit'], c['omega_correction_limit'])
                result[stream] = {'g': g, 'omega': p['omega']+dw}
        return result
