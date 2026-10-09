"""Exploratory anchored Fourier predictor from delivered measurements only."""
from collections import deque
import numpy as np
from sensor_age import STREAMS, predict_imu


class PeriodicCompensator:
    def __init__(self, limits, horizons, config=None):
        self.config = dict(period=.8, history=60, harmonics=3, ridge=.001, gain=1.,
                           joint_correction_limit=.35, velocity_correction_limit=6., omega_correction_limit=2.)
        if config:
            if set(config)-set(self.config):
                raise ValueError('Unknown Fourier predictor configuration')
            self.config.update(config)
        c = self.config
        if c['period'] <= 0 or not isinstance(c['history'], int) or c['history'] < 10 or not isinstance(c['harmonics'], int) or c['harmonics'] < 1:
            raise ValueError('Invalid periodic predictor')
        if any(not np.isfinite(v) or v < 0 for v in c.values()):
            raise ValueError('Invalid periodic coefficients')
        self.limits, self.horizons = np.array(limits, copy=True), dict(horizons)
        if any(not np.isfinite(v) or v < 0 for v in horizons.values()):
            raise ValueError('Invalid horizons')
        self.history = {s: deque(maxlen=c['history']) for s in STREAMS}
        self.fits = {}

    def basis(self, times, derivative=False):
        times = np.atleast_1d(times)
        cols = [np.zeros_like(times) if derivative else np.ones_like(times)]
        for k in range(1, self.config['harmonics']+1):
            w = 2*np.pi*k/self.config['period']
            cols += [w*np.cos(w*times), -w*np.sin(w*times)] if derivative else [np.sin(w*times), np.cos(w*times)]
        return np.stack(cols, axis=1)

    def update(self, packets, now_step):
        for s in STREAMS:
            p = packets[s]
            if p.stream != s or p.delivery_step > now_step:
                raise ValueError('Only delivered measurements')
            history = self.history[s]
            if history and p.sample_step <= history[-1].sample_step:
                continue
            history.append(p)
            # A full cycle is required; avoid an underdetermined partial fit.
            if history[-1].sample_time-history[0].sample_time < self.config['period']:
                self.fits.pop(s, None)
                continue
            times = np.array([packet.sample_time for packet in history])
            key = 'q' if s == 'joints' else 'omega'
            values = np.array([packet.payload[key] for packet in history])
            design = self.basis(times)
            penalty = self.config['ridge']*np.eye(design.shape[1])
            self.fits[s] = np.linalg.solve(design.T@design+penalty, design.T@values)

    def predict(self, now_step):
        out = {}
        c = self.config
        for s in STREAMS:
            if not self.history[s]:
                raise RuntimeError('Missing initial packet')
            packet = self.history[s][-1]
            if now_step < packet.delivery_step:
                raise ValueError('Prediction before delivery')
            payload = packet.payload
            age = (now_step-packet.sample_step)*packet.dt
            if age == 0 or s not in self.fits:
                out[s] = {k: v.copy() for k, v in payload.items()}
                continue
            tau = min(age, self.horizons[s])
            ts = packet.sample_time
            fit = self.fits[s]
            if s == 'joints':
                dq = c['gain']*((self.basis(ts+tau)-self.basis(ts))@fit)[0]
                dv = c['gain']*((self.basis(ts+tau, True)-self.basis(ts, True))@fit)[0]
                out[s] = {'q': np.clip(payload['q']+np.clip(dq, -c['joint_correction_limit'], c['joint_correction_limit']),
                                      self.limits[:, 0], self.limits[:, 1]),
                          'dq': payload['dq']+np.clip(dv, -c['velocity_correction_limit'], c['velocity_correction_limit'])}
            else:
                # Integrate the anchored angular-speed forecast in <=10 ms steps.
                g = payload['g'].copy()
                base = (self.basis(ts)@fit)[0]
                n = max(1, int(np.ceil(tau/.01)))
                for i in range(n):
                    omega = payload['omega']+c['gain']*((self.basis(ts+(i+.5)*tau/n)@fit)[0]-base)
                    g = predict_imu({'g': g, 'omega': omega}, tau/n, tau/n)['g']
                dw = c['gain']*((self.basis(ts+tau)@fit)[0]-base)
                out[s] = {'g': g, 'omega': payload['omega']+np.clip(dw, -c['omega_correction_limit'], c['omega_correction_limit'])}
        return out
