"""Technical repeats: preserve old methods and verify the frozen M3 CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from run_sensor_episode import configure_threads,run_episode
from check_sensor_age import audit_episode

ROOT=Path('results/m3_h1/checks/reproduction')


def compare(reference,repeat):
    excluded={'adapter_ms','inference_ms','control_cycle_ms'}
    with np.load(reference/'signals.npz') as a,np.load(repeat/'signals.npz') as b:
        assert set(a.files)==set(b.files)
        keys=sorted(set(a.files)-excluded)
        for key in keys:np.testing.assert_array_equal(a[key],b[key])
    audit_episode(repeat)
    return {'reference':str(reference),'repeat':str(repeat),'bitwise_equal_signal_keys':keys,'packet_audit':'passed'}


def main():
    # Keep technical repeats out of statistical cohorts and out of runs.jsonl.
    assert Path('results/m3_h1/closed_extensions/completed.json').exists()
    configure_threads(1);ROOT.mkdir(parents=True,exist_ok=True)
    calls=[];results={}
    inputs=Path('results/m2_validation_h1/inputs/seed_10')
    profile=json.loads(Path('configs/m2_profile.json').read_text())
    for method in ('B1','M','M2'):
        target=ROOT/method;parameters=profile['adapter_config'] if method=='M2' else None
        horizons=(120,20) if method=='M2' else (20,20)
        calls.append({'function':'run_sensor_episode.run_episode','output':str(target),'schedule':str(inputs/'J40.json'),
                      'initial':str(inputs/'initial.json'),'method':method,'horizons_ms':horizons,'adapter_config':parameters})
        (ROOT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        if not (target/'summary.json').exists():
            run_episode(target,inputs/'J40.json',inputs/'initial.json',method,horizons,adapter_config=parameters)
        results[method]=compare(Path('results/m2_validation_h1/episodes/seed_10/J40')/method,target)
    target=ROOT/'M3_CLI'
    args=[sys.executable,'-I','scripts/run_sensor_episode.py','--initial','results/m3_h1/inputs/seed_100/initial.json',
          '--schedule','results/m3_h1/inputs/seed_100/I80.json','--method','M3','--output',str(target)]
    calls.append({'argv':args,'environment':{k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}})
    (ROOT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
    if not (target/'summary.json').exists():
        env=dict(os.environ);env.update(calls[-1]['environment'])
        with (ROOT/'cli.log').open('w') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
    results['M3_CLI']=compare(Path('results/m3_h1/closed_primary/seed_100/I80/M3'),target)
    summary=json.loads((target/'summary.json').read_text());frozen=json.loads(Path('results/m3_h1/frozen/profile.json').read_text())
    assert summary['adapter_model_sha256']==frozen['model_sha256']
    assert summary['horizon_joints_ms']==summary['horizon_imu_ms']==80
    (ROOT/'identity.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'passed':True,'methods':list(results),'extra_independent_seed_count':0},indent=2))


if __name__=='__main__':main()
