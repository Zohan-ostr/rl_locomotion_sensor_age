"""Ridge training on whole development episodes; physical truth is a target only."""
import json
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from linear_compensator import FeatureHistory
from sensor_age import Packet, predict_imu
from run_h1_baseline import sha

ROOT=Path('results/m3_h1')

def dataset(actions):
    features={'joints':[],'imu':[]};targets={'joints':[],'imu':[]};taus={'joints':[],'imu':[]};sources=[]
    folders=sorted((ROOT/'training').glob('seed_*/*/*'))
    assert len(folders)==84
    for folder in folders:
        summary=json.loads((folder/'summary.json').read_text());assert 20<=summary['seed']<=25
        # NPZ members are lazy: cache once rather than decompressing a full array per row.
        with np.load(folder/'signals.npz') as stored:
            x={key:stored[key] for key in stored.files}
        dt=summary['upstream_config']['simulation_dt'];history=FeatureHistory(actions)
        initial={s:Packet(s,0,0,{k:x[k+'_true'][0] for k in (('q','dq') if s=='joints' else ('omega','g'))},dt) for s in ('joints','imu')}
        history.update(initial,0)
        for i,step in enumerate(x['decision_steps']):
            packets={s:Packet(s,int(x['selected_sample_steps'][i,j]),int(x['selected_delivery_steps'][i,j]),
                       {k:x[k+'_delivered'][i] for k in (('q','dq') if s=='joints' else ('omega','g'))},dt) for j,s in enumerate(('joints','imu'))}
            history.update(packets,int(step))
            history.context(int(step),x['actions'][i-1] if i else np.zeros(10),summary['command'],(step*dt%.8)/.8)
            t=step*dt
            if t<2.02 or (summary['fall_time_s'] is not None and t>=summary['fall_time_s']):continue
            index=int(np.searchsorted(x['sample_steps'],step));assert x['sample_steps'][index]==step
            for j,s in enumerate(('joints','imu')):
                age=x['ages_s'][i,j]
                if age<=0:continue
                assert age<=.08000001
                features[s].append(history.features(s,int(step),age));taus[s].append(age)
                if s=='joints':
                    targets[s].append(np.r_[x['q_true'][index]-x['q_delivered'][i]-x['dq_delivered'][i]*age,
                                           x['dq_true'][index]-x['dq_delivered'][i]])
                else:
                    base=predict_imu(packets[s].payload,age,.08)
                    targets[s].append(np.r_[x['omega_true'][index]-x['omega_delivered'][i],x['g_true'][index]-base['g']]/(age/.08))
        sources.append({'folder':str(folder),'summary_sha256':sha(folder/'summary.json'),'signals_sha256':sha(folder/'signals.npz')})
    return {s:np.asarray(v) for s,v in features.items()},{s:np.asarray(v) for s,v in targets.items()},{s:np.asarray(v) for s,v in taus.items()},sources


def main():
    assert (ROOT/'training/completed.json').exists()
    output=ROOT/'models';output.mkdir(exist_ok=True)
    stats=[]
    for actions in (False,True):
        target=output/f'actions_{int(actions)}.npz'
        if target.exists():raise FileExistsError(target)
        started=time.perf_counter();features,targets,taus,sources=dataset(actions);model={};details={}
        for stream in ('joints','imu'):
            raw=features[stream];mean=raw.mean(axis=0);std=raw.std(axis=0);std[std<1e-6]=1
            phi=np.column_stack([(raw-mean)/std,np.ones(len(raw))]);penalty=np.eye(phi.shape[1]);penalty[-1,-1]=0
            if stream=='joints':
                s=taus[stream]/.08;f=phi.shape[1]
                q=np.column_stack([phi*s[:,None]**2,phi*s[:,None]**3])/.05
                dq=np.column_stack([phi*2*s[:,None]/.08,phi*3*s[:,None]**2/.08])
                design=np.row_stack([q,dq]);y=np.row_stack([targets[stream][:,:10]/.05,targets[stream][:,10:]])
                penalty=np.kron(np.eye(2),penalty)
            else:design=phi;y=targets[stream]
            w=np.linalg.solve(design.T@design+penalty,design.T@y)
            model[stream+'_mean']=mean;model[stream+'_std']=std;model[stream+'_weights']=w
            details[stream]={'rows':len(raw),'features':phi.shape[1],'training_loss_rmse':float(np.sqrt(np.mean((design@w-y)**2)))}
        meta={'actions':actions,'train_seeds':list(range(20,26)),'history_packets':3,'history_actions':4,'ridge':1.,'horizon_s':.08,
              'joint_position_loss_scale_rad':.05,'joint_velocity_loss_scale_rad_s':1.,'feature_truth_access':False,
              'target':'current physical q/dq and omega/g; joint residuals fitted jointly on cubic segment',
              'sources':sources,'statistics':details}
        model['metadata']=json.dumps(meta);np.savez_compressed(target,**model)
        stats.append({'actions':actions,'model':str(target),'sha256':sha(target),'elapsed_s':time.perf_counter()-started,'statistics':details})
        print(json.dumps(stats[-1]),flush=True)
    (output/'training.json').write_text(json.dumps(stats,indent=2)+'\n')

if __name__=='__main__':main()
