"""M3: causal ridge state predictor; no simulator/policy objects or true state."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from sensor_age import STREAMS, predict_imu


class FeatureHistory:
    def __init__(self, actions=False):
        self.actions_enabled=bool(actions)
        self.history={s:deque(maxlen=3) for s in STREAMS}
        self.action_history=deque(maxlen=4)
        self.now=-1;self.command=np.zeros(3);self.phase=0.

    def update(self, packets, now_step):
        for stream in STREAMS:
            p=packets[stream]
            if p.stream!=stream or p.delivery_step>now_step or p.sample_step>now_step:
                raise ValueError('Only causally delivered packets are permitted')
            history=self.history[stream]
            if not history or p.sample_step>history[-1].sample_step:history.append(p)

    def context(self, now_step, previous_action, command, phase):
        if now_step<self.now:raise ValueError('Clock reversal')
        action=np.asarray(previous_action,dtype=float);cmd=np.asarray(command,dtype=float)
        if action.shape!=(10,) or cmd.shape!=(3,) or not np.isfinite(np.r_[action,cmd,phase]).all():raise ValueError('Invalid known controller context')
        if now_step>self.now:self.action_history.append((now_step-10,action.copy()))
        self.now=now_step;self.command=cmd.copy();self.phase=float(phase)

    @staticmethod
    def vector(packet):
        return np.concatenate([packet.payload[k] for k in (('q','dq') if packet.stream=='joints' else ('omega','g'))])

    def features(self,stream,now_step,tau):
        if not all(self.history.values()):raise RuntimeError('Missing delivered initial measurements')
        if now_step!=self.now:raise ValueError('Set causal controller context before predicting')
        latest=self.history[stream][-1]
        if latest.delivery_step>now_step:raise ValueError('Future packet')
        history=list(self.history[stream]);values=[]
        for lag in range(3):
            p=history[max(0,len(history)-1-lag)]
            values.extend(self.vector(p));values.append((latest.sample_step-p.sample_step)*p.dt)
        for s in STREAMS:
            p=self.history[s][-1];values.extend(self.vector(p))
            values.append(min((now_step-p.sample_step)*p.dt,.08))
        values.extend(self.command)
        values.extend([np.sin(2*np.pi*k*self.phase) for k in (1,2,3)])
        values.extend([np.cos(2*np.pi*k*self.phase) for k in (1,2,3)])
        values.append(tau/.08)
        if self.actions_enabled:
            actions=list(self.action_history)
            for lag in range(4):
                if len(actions)>lag:
                    stamp,action=actions[-1-lag]
                    if stamp>=now_step:raise ValueError('Current/future action is unavailable')
                    values.extend(action)
                else:values.extend(np.zeros(10))
        result=np.asarray(values)
        if not np.isfinite(result).all():raise ValueError('Nonfinite delivered features')
        return result


class LinearCompensator(FeatureHistory):
    def __init__(self, limits, horizons, config):
        allowed={'model_path','gain'}
        if set(config)-allowed or 'model_path' not in config:raise ValueError('Specify frozen ridge model')
        self.config={'model_path':str(config['model_path']),'gain':float(config.get('gain',1.0))}
        if not 0<=self.config['gain']<=1:raise ValueError('Invalid prediction gain')
        self.model=np.load(Path(self.config['model_path']),allow_pickle=False)
        self.metadata=json.loads(str(self.model['metadata']))
        super().__init__(self.metadata['actions'])
        self.limits=np.array(limits,copy=True);self.horizons=dict(horizons)
        if any(v<0 or v>.08 for v in horizons.values()):raise ValueError('M3 model horizon is limited to 80 ms')

    def phi(self,stream,now_step,tau):
        raw=self.features(stream,now_step,tau)
        scaled=(raw-self.model[stream+'_mean'])/self.model[stream+'_std']
        return np.r_[scaled,1.]

    def joint_curve(self,packet,coefficients,tau):
        """q and dq are one cubic segment and its analytic derivative."""
        tau=min(tau,self.horizons['joints']);s=tau/.08
        a,b=coefficients
        gain=self.config['gain'];a=a*gain;b=b*gain
        # Bound coefficients over the entire segment, not independently at each query.
        # Thus saturation of correction magnitudes does not break dq=dq_curve/dt.
        maximum=self.horizons['joints']/.08
        q_bound=np.abs(a)*maximum**2+np.abs(b)*maximum**3
        dq_bound=(2*np.abs(a)*maximum+3*np.abs(b)*maximum**2)/.08
        scale=np.minimum(1,np.minimum(.35/np.maximum(q_bound,1e-12),6/np.maximum(dq_bound,1e-12)))
        a=a*scale;b=b*scale
        q=packet.payload['q']+packet.payload['dq']*tau+a*s*s+b*s*s*s
        dq=packet.payload['dq']+(2*a*s+3*b*s*s)/.08
        clipped=np.clip(q,self.limits[:,0],self.limits[:,1])
        dq=np.where(q==clipped,dq,0.)
        return {'q':clipped,'dq':dq}

    def predict(self,now_step):
        result={}
        for stream in STREAMS:
            packet=self.history[stream][-1]
            age=(now_step-packet.sample_step)*packet.dt
            if age<0 or packet.delivery_step>now_step:raise ValueError('Prediction before delivery')
            if age==0:
                result[stream]={k:v.copy() for k,v in packet.payload.items()};continue
            tau=min(age,self.horizons[stream]);phi=self.phi(stream,now_step,tau)
            if stream=='joints':
                w=self.model['joints_weights'];f=len(phi)
                result[stream]=self.joint_curve(packet,np.array([phi@w[:f],phi@w[f:]]),tau)
            else:
                correction=phi@self.model['imu_weights']*(tau/.08)*self.config['gain']
                omega=packet.payload['omega']+np.clip(correction[:3],-2,2)
                g=predict_imu(packet.payload,age,self.horizons[stream])['g']+np.clip(correction[3:],-.35,.35)
                norm=np.linalg.norm(g)
                if norm<1e-10:raise FloatingPointError('Degenerate gravity prediction')
                result[stream]={'omega':omega,'g':g/norm}
        return result
