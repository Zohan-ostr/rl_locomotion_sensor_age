"""Sequential, preregistered CPU study; each stage keeps real inputs and results."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
from run_sensor_episode import configure_threads, run_episode
from sensor_age import make_schedule, Packet, PacketQueue
from check_sensor_age import audit_episode
from run_h1_baseline import sha

ROOT = Path('results/m3_h1')
PRIMARY = ['J20','J40','J80','I20','I40','I80']
SINGLE = PRIMARY + ['J60','I60']
JOINT = ['J40I40','J60I40','J40I60']


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=text: raise FileExistsError(path)
    else:path.write_text(text)


def condition(name):
    import re
    out={'id':name}
    for prefix,d in re.findall(r'([JI])(\d+)',name):out['delay_'+('joints' if prefix=='J' else 'imu')+'_ms']=int(d)
    return out


def prepare(seeds, names):
    model=mujoco.MjModel.from_xml_path('vendor/unitree_rl_gym/resources/robots/h1/scene.xml')
    for seed in seeds:
        p=ROOT/f'inputs/seed_{seed}'
        delta=np.random.default_rng(np.random.SeedSequence([seed,20261009])).uniform(-.003,.003,10)
        qpos=model.qpos0.copy();qpos[7:]+=delta
        write(p/'initial.json',{'seed':seed,'qpos':qpos.tolist(),'qvel':np.zeros(model.nv).tolist(),'joint_delta_rad':delta.tolist()})
        for name in names:write(p/(name+'.json'),make_schedule(seed,condition(name)))


def execute(folder,seed,name,method='B1',params=None,oracle=(),diagnostics=False):
    folder=Path(folder)
    if (folder/'summary.json').exists():return json.loads((folder/'summary.json').read_text())
    inputs=ROOT/f'inputs/seed_{seed}'
    if method=='M2':
        profile=json.loads(Path('configs/m2_profile.json').read_text())
        params=profile['adapter_config'];h=(120,20)
    elif method=='M3':h=(80,80)
    else:h=(20,20)
    call={'function':'run_sensor_episode.run_episode','output':str(folder),'seed':seed,'condition':name,
          'method':method,'horizons_ms':h,'adapter_config':params,'oracle_components':oracle,
          'record_diagnostics':diagnostics,'started_unix':time.time()}
    with (ROOT/'runs.jsonl').open('a') as f:f.write(json.dumps(call)+'\n')
    s=run_episode(folder,inputs/(name+'.json'),inputs/'initial.json',method,h,
                  adapter_config=params,oracle_components=oracle,record_diagnostics=diagnostics)
    if not oracle:
        audit=audit_episode(folder)
        if method=='M3':
            x=np.load(folder/'signals.npz')
            for key,index in [('q',0),('dq',0),('omega',1),('g',1)]:
                fresh=x['ages_s'][:,index]==0
                np.testing.assert_array_equal(x[key+'_predicted'][fresh],x[key+'_delivered'][fresh])
            np.testing.assert_allclose(np.linalg.norm(x['g_predicted'],axis=1),1,atol=1e-8)
    else:audit={'oracle_only':True,'not_operational_adapter':True}
    write(folder/'audit.json',audit)
    print(json.dumps({'completed':str(folder),'fallen':s['fallen'],'alive':s['time_alive_s'],'rmse':s['velocity_rmse_after_warmup_before_fall_m_s']}),flush=True)
    return s


def diagnostic():
    prepare([20,21],SINGLE+['J50','J70']+JOINT+['clean'])
    # Actual queue observation times, independent of nominal names.
    signatures={}
    for name in ['J50','J60','J70','J80','I50','I60','I70','I80']:
        sched=make_schedule(20,condition(name));queues={s:PacketQueue(s) for s in ('joints','imu')};steps=[]
        ages=[]
        for k in range(1001):
            step=k*10;selected=[]
            for stream in queues:
                entry=sched['streams'][stream][k]
                if not entry['drop']:queues[stream].enqueue(Packet(stream,step,entry['delivery_step'],{'ramp':np.array([step])}),step)
                packet,_=queues[stream].deliver(step);selected.append(packet.sample_step)
            steps.append(selected);ages.append([(step-t)*.002 for t in selected])
        signatures[name]=np.array(steps)
        write(ROOT/'ages'/f'{name}.json',{'nominal':name,'observed_age_ms_after_2_2s':(np.unique(np.array(ages)[110:],axis=0)*1000).tolist(),'units':'milliseconds'})
    equivalence={a+'='+b:bool(np.array_equal(signatures[a],signatures[b])) for a,b in [('J50','J60'),('J70','J80'),('I50','I60'),('I70','I80')]}
    assert all(equivalence.values());write(ROOT/'ages/equivalence.json',equivalence)
    results=[]
    for seed in (20,21):
        for name,keys in [('J80',('q','dq')),('I80',('omega','g'))]:
            results.append(execute(ROOT/f'diagnostics/seed_{seed}/{name}/B1',seed,name,diagnostics=True))
            for comps in [(keys[0],),(keys[1],),keys]:
                results.append(execute(ROOT/f'diagnostics/seed_{seed}/{name}/oracle_{"_".join(comps)}',seed,name,oracle=comps))
    # Demonstrate noninvasive diagnostic forwards and nominal alias equivalence in physical closed loop.
    execute(ROOT/'diagnostics/invariance_plain',20,'J80')
    with np.load(ROOT/'diagnostics/invariance_plain/signals.npz') as a,np.load(ROOT/'diagnostics/seed_20/J80/B1/signals.npz') as b:
        for key in ('obs','actions','q_true','dq_true','omega_true','g_true'):np.testing.assert_array_equal(a[key],b[key])
    for name in ('J50','J60'):execute(ROOT/f'diagnostics/alias_{name}',20,name)
    with np.load(ROOT/'diagnostics/alias_J50/signals.npz') as a,np.load(ROOT/'diagnostics/alias_J60/signals.npz') as b:
        for key in ('obs','actions','q_true','dq_true','omega_true','g_true','ages_s','selected_sample_steps'):np.testing.assert_array_equal(a[key],b[key])
    output=[]
    for seed in (20,21):
        for name in ('J80','I80'):
            d=np.load(ROOT/f'diagnostics/seed_{seed}/{name}/B1/diagnostics.npz')
            for variant in ('B1','M','M2','oracle_q','oracle_dq','oracle_omega','oracle_g'):
                error=d['action_'+variant]-d['action_fresh']
                output.append({'seed':seed,'condition':name,'variant':variant,'action_rmse':float(np.sqrt(np.mean(error**2))),
                               'interpretation':'same B1 physical trajectory and copied working LSTM; not closed-loop outcome'})
    write(ROOT/'diagnostics/common_trajectory.json',output)
    write(ROOT/'diagnostics/closed_loop.json',[{'seed':s['seed'],'condition':s['condition']['id'],'oracle':s['oracle_components'],
          'fallen':s['fallen'],'alive_s':s['time_alive_s'],'velocity_rmse':s['velocity_rmse_after_warmup_before_fall_m_s']} for s in results])
    write(ROOT/'diagnostics/completed.json',{'closed_oracle_episodes':len(results),'noninvasive_forward_check':True,'nominal_alias_physics_check':True})


def collect():
    assert (ROOT/'diagnostics/completed.json').exists()
    names=['clean','J40','J60','J80','I40','I60','I80']
    prepare(range(20,26),names)
    for seed in range(20,26):
        for name in names:
            for method in ('B1','M2'):execute(ROOT/f'training/seed_{seed}/{name}/{method}',seed,name,method)
    write(ROOT/'training/completed.json',{'episodes':84,'seeds':list(range(20,26)),'targets_only_from_truth':True})


def tune():
    plan=json.loads((ROOT/'plan.json').read_text());prepare(range(26,30),SINGLE+JOINT+['clean'])
    for seed in range(26,30):
        for name in SINGLE:
            for method in ('B1','M','M2'):execute(ROOT/f'development/seed_{seed}/{name}/{method}',seed,name,method)
            for candidate in plan['candidates']:
                params={'family':'linear','model_path':str(ROOT/f'models/actions_{int(candidate["actions"])}.npz'),'gain':candidate['gain']}
                execute(ROOT/f'development/seed_{seed}/{name}/{candidate["id"]}',seed,name,'M3',params)
    scores=[]
    for candidate in plan['candidates']:
        values=[json.loads(p.read_text()) for p in (ROOT/'development').glob(f'seed_*/**/{candidate["id"]}/summary.json') if p.parent.parent.name in PRIMARY]
        assert len(values)==24
        scores.append({**candidate,'episodes':len(values),'falls':sum(s['fallen'] for s in values),'successes':sum(s['walking_success'] for s in values),
            'mean_alive_s':float(np.mean([s['time_alive_s'] for s in values])),
            'velocity_rmse':float(np.mean([s['velocity_rmse_after_warmup_before_fall_m_s'] for s in values]))})
    best=min(scores,key=lambda v:(v['falls'],-v['successes'],-v['mean_alive_s'],v['velocity_rmse'],v['id']))
    write(ROOT/'selection.json',{'candidates':scores,'selected':best,'selection_conditions':PRIMARY,'no_closed_seeds_used':True})
    for seed in range(26,30):
        for name in JOINT:
            for method in ('B1','M','M2','M3'):
                params={'family':'linear','model_path':str(ROOT/f'models/actions_{int(best["actions"])}.npz'),'gain':best['gain']} if method=='M3' else None
                execute(ROOT/f'development_joint/seed_{seed}/{name}/{method}',seed,name,method,params)
    # All clean variants must be bitwise identical, before freezing.
    for method in ('B1','M','M2','M3'):
        params={'family':'linear','model_path':str(ROOT/f'models/actions_{int(best["actions"])}.npz'),'gain':best['gain']} if method=='M3' else None
        execute(ROOT/f'checks/clean/{method}',26,'clean',method,params)
    reference=np.load(ROOT/'checks/clean/B1/signals.npz')
    for method in ('M','M2','M3'):
        other=np.load(ROOT/f'checks/clean/{method}/signals.npz')
        for key in ('obs','actions','q_true','dq_true','g_true','omega_true'):np.testing.assert_array_equal(reference[key],other[key])
    write(ROOT/'checks/clean_identity.json',{'methods':['B1','M','M2','M3'],'bitwise_identical':True})
    for name in ('J80','I80'):
        execute(ROOT/f'diagnostics/holdout_26/{name}/B1',26,name,diagnostics=True)
        with np.load(ROOT/f'diagnostics/holdout_26/{name}/B1/signals.npz') as a,np.load(ROOT/f'development/seed_26/{name}/B1/signals.npz') as b:
            for key in ('obs','actions','q_true','dq_true','g_true','omega_true'):np.testing.assert_array_equal(a[key],b[key])


def freeze():
    best=json.loads((ROOT/'selection.json').read_text())['selected']
    source=ROOT/f'models/actions_{int(best["actions"])}.npz';target=ROOT/'frozen/model.npz'
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copy2(source,target)
    profile={'status':'frozen_before_closed_test','horizon_joints_ms':80,'horizon_imu_ms':80,
       'adapter_config':{'family':'linear','model_path':str(target),'gain':best['gain']},'selected':best,
       'model_sha256':sha(target),'train_seeds':list(range(20,26)),'development_seeds':list(range(26,30)),
       'closed_test_seeds':list(range(100,140)),'primary_conditions':PRIMARY,'test_opened':False}
    write(ROOT/'frozen/profile.json',profile);write('configs/m3_profile.json',profile)
    files=['run_sensor_episode.py','sensor_age.py','linear_compensator.py','run_m3_study.py','train_m3.py','check_m3.py',
           'run_h1_baseline.py','hybrid_compensator.py','periodic_compensator.py']
    snapshot=ROOT/'frozen/source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in files:
        if not (snapshot/name).exists():shutil.copy2(Path('scripts')/name,snapshot/name)
    write(ROOT/'frozen/code_sha256.json',{name:sha(Path('scripts')/name) for name in files})
    write(ROOT/'frozen/test_plan.json',{'seeds':list(range(100,140)),'primary':PRIMARY,'extension_single':['J60','I60'],
       'joint_seeds':list(range(100,110)),'joint':JOINT,'methods':['B1','M','M2','M3'],'profile_sha256':sha(ROOT/'frozen/profile.json'),
       'model_sha256':sha(target),'primary_episodes':960,'extension_single_episodes':320,'joint_episodes':120,
       'bootstrap_resamples':5000,'bootstrap_rng_seed':20261009,'primary_minimum_improvement':.10,'retuning_allowed':False})


def test(extensions=False):
    profile=json.loads((ROOT/'frozen/profile.json').read_text())
    assert sha(Path(profile['adapter_config']['model_path']))==profile['model_sha256']
    for name,value in json.loads((ROOT/'frozen/code_sha256.json').read_text()).items():assert sha(Path('scripts')/name)==value,name
    if not (ROOT/'frozen/opened.json').exists():write(ROOT/'frozen/opened.json',{'opened_unix':time.time(),'profile_sha256':sha(ROOT/'frozen/profile.json'),'no_retuning':True})
    names=['J60','I60'] if extensions else PRIMARY
    prepare(range(100,140),names)
    for seed in range(100,140):
        for name in names:
            for method in ('B1','M','M2','M3'):
                execute(ROOT/f'{"closed_extensions" if extensions else "closed_primary"}/seed_{seed}/{name}/{method}',seed,name,method,
                        profile['adapter_config'] if method=='M3' else None)
    if extensions:
        prepare(range(100,110),JOINT)
        for seed in range(100,110):
            for name in JOINT:
                for method in ('B1','M','M2','M3'):execute(ROOT/f'closed_joint/seed_{seed}/{name}/{method}',seed,name,method,profile['adapter_config'] if method=='M3' else None)
    write(ROOT/f'{"closed_extensions" if extensions else "closed_primary"}/completed.json',{'completed':True,'no_retuning':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['diagnostics','collect','tune','freeze','test','extensions'])
    args=p.parse_args();configure_threads(1)
    if args.stage=='diagnostics':diagnostic()
    elif args.stage=='collect':collect()
    elif args.stage=='tune':tune()
    elif args.stage=='freeze':freeze()
    else:test(args.stage=='extensions')

if __name__=='__main__':main()
