"""Measured paired statistics, common-trajectory diagnostics, plots and report."""
import argparse,csv,json,os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
os.environ.setdefault('MPLCONFIGDIR',str(Path('.cache/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from run_m3_study import ROOT,PRIMARY,JOINT,condition
from linear_compensator import LinearCompensator
from sensor_age import Packet
from run_sensor_episode import configure_threads
from run_h1_baseline import sha

METHODS=('B1','M','M2','M3')
COLORS={'B1':'#666666','M':'#dc7b11','M2':'#3976af','M3':'#178647'}

def write(name,value):
    path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,c-h),min(1.,c+h)]


def table(name,rows):
    p=ROOT/'tables';p.mkdir(exist_ok=True)
    keys=list(rows[0])
    with (p/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def save(fig,name):
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=160,bbox_inches='tight')
    plt.close(fig)


def load_records():
    rows=[];timings={};age_rows=[]
    frozen=json.loads((ROOT/'frozen/profile.json').read_text())
    frozen_sources=json.loads((ROOT/'frozen/code_sha256.json').read_text())
    for cohort in ('development','development_joint','closed_primary','closed_extensions','closed_joint'):
        for p in sorted((ROOT/cohort).glob('seed_*/*/*/summary.json')):
            s=json.loads(p.read_text());method=p.parent.name
            if cohort.startswith('closed_'):
                assert not s['oracle_components'] and not s['record_diagnostics']
                assert all(value==frozen_sources[name] for name,value in s['runner_sha256'].items() if name in frozen_sources)
                if method=='M3':
                    assert s['adapter_model_sha256']==frozen['model_sha256']
                    assert s['adapter_config']==frozen['adapter_config']
            with np.load(p.parent/'signals.npz') as saved:x={k:saved[k] for k in saved.files}
            t=x['decision_steps']*s['upstream_config']['simulation_dt']
            valid=(t>=2)&((t<s['fall_time_s']) if s['fall_time_s'] is not None else True)
            idx=np.searchsorted(x['sample_steps'],x['decision_steps']);assert np.array_equal(x['sample_steps'][idx],x['decision_steps'])
            errors={}
            for key in ('q','dq','omega'):
                errors[key+'_rmse']=float(np.sqrt(np.mean((x[key+'_predicted'][valid]-x[key+'_true'][idx][valid])**2)))
            keyset=('initial_state_sha256','schedule_sha256','checkpoint_sha256','scene_sha256','model_asset_sha256','upstream_config','command','fall_rule')
            rec={'cohort':cohort,'seed':s['seed'],'condition':s['condition']['id'],'method':method,
                'fallen':int(s['fallen']),'alive_s':s['time_alive_s'],'velocity_rmse':s['velocity_rmse_after_warmup_before_fall_m_s'],
                'success':int(s['walking_success']),**errors,'gravity_mean_angle_rad':s['gravity_prediction_mean_angle_rad'],
                'maximum_age_joints_ms':float(x['ages_s'][:,0].max()*1000),'maximum_age_imu_ms':float(x['ages_s'][:,1].max()*1000),
                'source':str(p),'source_sha256':sha(p),'paired_identity':{k:s[k] for k in keyset}}
            rows.append(rec)
            steady=t>=2.2
            assert steady.any()
            expected=condition(s['condition']['id'])
            for j,stream in enumerate(('joints','imu')):
                ages=x['ages_s'][steady,j]*1000
                nominal=expected.get('delay_'+stream+'_ms',0)
                actual=np.ceil(nominal/20)*20
                np.testing.assert_allclose(ages,actual,rtol=0,atol=1e-10)
                age_rows.append({'cohort':cohort,'seed':s['seed'],'condition':s['condition']['id'],'method':method,
                                 'stream':stream,'nominal_ms':nominal,'n_decisions_after_2_2s':int(sum(steady)),
                                 'median_ms':float(np.median(ages)),'p95_ms':float(np.percentile(ages,95)),
                                 'minimum_ms':float(ages.min()),'maximum_ms':float(ages.max())})
            if cohort=='closed_primary' and method in METHODS:
                for k in ('adapter_ms','inference_ms','control_cycle_ms'):
                    timings.setdefault(method,{}).setdefault(k,[]).extend(x[k].tolist())
    # Same schedule, states, physics, commands and fall rule across all compared methods.
    for cohort in ('closed_primary','closed_extensions','closed_joint'):
        values=[r for r in rows if r['cohort']==cohort]
        for seed,cond in sorted({(r['seed'],r['condition']) for r in values}):
            group=[r for r in values if r['seed']==seed and r['condition']==cond]
            assert len(group)==4 and set(r['method'] for r in group)==set(METHODS)
            assert all(r['paired_identity']==group[0]['paired_identity'] for r in group)
    write('paired_records.json',rows)
    table('episodes',[{k:v for k,v in r.items() if k!='paired_identity'} for r in rows])
    table('ages',age_rows)
    timing={m:{k:{'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'n_cycles':len(v)} for k,v in parts.items()} for m,parts in timings.items()}
    write('cpu_timing.json',timing)
    table('cpu_timing',[{'method':m,'component':k,**v} for m,parts in timing.items() for k,v in parts.items()])
    return rows,timing


def paired_stats(records):
    rows=[];groups=[]
    for cohort in ('closed_primary','closed_extensions','closed_joint'):
        subset=[r for r in records if r['cohort']==cohort]
        names=sorted(set(r['condition'] for r in subset))
        if not names:continue
        seeds=sorted(set(r['seed'] for r in subset))
        for cond in ['ALL']+names:
            values=[r for r in subset if cond=='ALL' or r['condition']==cond]
            for method in METHODS:
                a=[r for r in values if r['method']==method]
                n=len(a);falls=sum(r['fallen'] for r in a)
                groups.append({'cohort':cohort,'condition':cond,'method':method,'n_seed':len(seeds),'n_episodes':n,
                    'falls':falls,'fall_rate':falls/n,'mean_alive_s':float(np.mean([r['alive_s'] for r in a])),
                    'velocity_rmse':float(np.mean([r['velocity_rmse'] for r in a])),'successes':sum(r['success'] for r in a),
                    'q_rmse_rad':float(np.mean([r['q_rmse'] for r in a])),'dq_rmse_rad_s':float(np.mean([r['dq_rmse'] for r in a])),
                    'omega_rmse_rad_s':float(np.mean([r['omega_rmse'] for r in a])),'gravity_error_rad':float(np.mean([r['gravity_mean_angle_rad'] for r in a])),
                    'wilson_low':wilson(falls,n)[0] if cond!='ALL' else None,'wilson_high':wilson(falls,n)[1] if cond!='ALL' else None})
            for method in ('M','M2','M3'):
                for metric in ('fallen','alive_s','velocity_rmse','success','q_rmse','dq_rmse','omega_rmse','gravity_mean_angle_rad'):
                    diffs=[]
                    for seed in seeds:
                        a=[r for r in values if r['seed']==seed and r['method']==method]
                        b=[r for r in values if r['seed']==seed and r['method']=='B1']
                        assert len(a)==len(b)>0
                        diffs.append(float(np.mean([r[metric] for r in a])-np.mean([r[metric] for r in b])))
                    rng=np.random.default_rng(20261009)
                    boot=np.mean(rng.choice(diffs,size=(5000,len(seeds)),replace=True),axis=1)
                    lo,hi=np.quantile(boot,[.025,.975])
                    rows.append({'cohort':cohort,'condition':cond,'method':method,'metric':metric,'difference_method_minus_B1':float(np.mean(diffs)),
                                 'ci95_low':float(lo),'ci95_high':float(hi),'n_seed':len(seeds),'resamples':5000})
    table('groups',groups);table('paired_differences',rows);write('groups.json',groups);write('paired_statistics.json',rows)
    endpoint=next(r for r in rows if r['cohort']=='closed_primary' and r['condition']=='ALL' and r['method']=='M3' and r['metric']=='fallen')
    result={'fall_rate_improvement_B1_minus_M3':-endpoint['difference_method_minus_B1'],
            'paired95_interval':[-endpoint['ci95_high'],-endpoint['ci95_low']],
            'criterion_improvement_at_least_0_10':-endpoint['difference_method_minus_B1']>=.10-1e-12,
            'criterion_lower_bound_positive':-endpoint['ci95_high']>0,'n_seed':40,'conditions':PRIMARY,'primary_episode_count':960}
    result['main_hypothesis_criterion_met']=result['criterion_improvement_at_least_0_10'] and result['criterion_lower_bound_positive']
    write('primary_conclusion.json',result)
    return groups,rows,result


def common_diagnostics():
    profile=json.loads((ROOT/'frozen/profile.json').read_text())
    policy=torch.jit.load('vendor/unitree_rl_gym/deploy/pre_train/h1/motion.pt',map_location='cpu').eval()
    rows=[];timeline=[]
    # Use the development episode not used in regression fitting; selection still used its outcomes.
    for name in ('J80','I80'):
        folder=ROOT/f'diagnostics/holdout_26/{name}/B1';s=json.loads((folder/'summary.json').read_text())
        with np.load(folder/'signals.npz') as z:x={k:z[k] for k in z.files}
        with np.load(folder/'diagnostics.npz') as z:d={k:z[k] for k in z.files}
        import mujoco
        model=mujoco.MjModel.from_xml_path('vendor/unitree_rl_gym/resources/robots/h1/scene.xml')
        adapter=LinearCompensator(model.jnt_range[1:],{'joints':.08,'imu':.08}, {k:v for k,v in profile['adapter_config'].items() if k!='family'})
        dt=s['upstream_config']['simulation_dt'];cfg=s['upstream_config'];default=np.array(cfg['default_angles'],dtype=np.float32)
        initial={stream:Packet(stream,0,0,{k:x[k+'_true'][0] for k in (('q','dq') if stream=='joints' else ('omega','g'))},dt) for stream in ('joints','imu')}
        adapter.update(initial,0);m3_actions=[];m3_obs=[]
        with torch.inference_mode():
            for i,step in enumerate(x['decision_steps']):
                packets={stream:Packet(stream,int(x['selected_sample_steps'][i,j]),int(x['selected_delivery_steps'][i,j]),
                    {k:x[k+'_delivered'][i] for k in (('q','dq') if stream=='joints' else ('omega','g'))},dt) for j,stream in enumerate(('joints','imu'))}
                adapter.update(packets,int(step));adapter.context(int(step),x['actions'][i-1] if i else np.zeros(10),s['command'],(step*dt%.8)/.8)
                pred=adapter.predict(int(step));obs=x['obs'][i].copy()
                obs[:3]=pred['imu']['omega']*cfg['ang_vel_scale'];obs[3:6]=pred['imu']['g']
                obs[9:19]=(pred['joints']['q']-default)*cfg['dof_pos_scale'];obs[19:29]=pred['joints']['dq']*cfg['dof_vel_scale']
                for k,v in policy.named_buffers():v.copy_(torch.from_numpy(d['lstm_'+k][i]))
                m3_actions.append(policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze().copy());m3_obs.append(obs)
        d['action_M3']=np.array(m3_actions);d['obs_M3']=np.array(m3_obs)
        valid=x['decision_steps']*dt>=2
        for method in METHODS:
            obs=d['obs_'+method];fresh=d['obs_fresh'];act=d['action_'+method]-d['action_fresh']
            metrics={'q_error_rad':float(np.sqrt(np.mean(((obs[valid,9:19]-fresh[valid,9:19])/cfg['dof_pos_scale'])**2))),
                     'dq_error_rad_s':float(np.sqrt(np.mean(((obs[valid,19:29]-fresh[valid,19:29])/cfg['dof_vel_scale'])**2))),
                     'omega_error_rad_s':float(np.sqrt(np.mean(((obs[valid,:3]-fresh[valid,:3])/cfg['ang_vel_scale'])**2))),
                     'g_vector_error':float(np.sqrt(np.mean((obs[valid,3:6]-fresh[valid,3:6])**2))),
                     'action_rmse':float(np.sqrt(np.mean(act[valid]**2)))}
            rows.append({'seed':26,'condition':name,'method':method,'n_decisions':int(sum(valid)),**metrics})
            for i in np.flatnonzero(valid):timeline.append({'seed':26,'condition':name,'method':method,'time_s':x['decision_steps'][i]*dt,
                 'action_rmse':float(np.sqrt(np.mean(act[i]**2))),'age_joints_ms':x['ages_s'][i,0]*1000,'age_imu_ms':x['ages_s'][i,1]*1000})
    write('diagnostics/M3_common_trajectory.json',rows);table('common_trajectory',rows);table('common_action_timeline',timeline)
    return rows


def matched_velocity_windows(records):
    """Secondary sensitivity analysis; never substitutes the primary endpoint."""
    rows=[];comparisons=[]
    for cohort in ('closed_primary','closed_extensions','closed_joint'):
        subset=[r for r in records if r['cohort']==cohort]
        for seed,c in sorted({(r['seed'],r['condition']) for r in subset}):
            pair=[next(r for r in subset if r['seed']==seed and r['condition']==c and r['method']==m) for m in ('B1','M3')]
            end=min(r['alive_s'] for r in pair);values=[]
            for r in pair:
                x=np.genfromtxt(Path(r['source']).with_name('trajectory.csv'),delimiter=',',names=True)
                valid=(x['time_s']>=2)&(x['time_s']<end)
                error=(x['vx_yaw_m_s'][valid]-x['cmd_vx_m_s'][valid])**2+(x['vy_yaw_m_s'][valid]-x['cmd_vy_m_s'][valid])**2
                rmse=float(np.sqrt(error.mean()));values.append(rmse)
                rows.append({'cohort':cohort,'seed':seed,'condition':c,'method':r['method'],'window_start_s':2,'window_end_exclusive_s':end,
                             'n_decisions':int(sum(valid)),'velocity_rmse_m_s':rmse})
            comparisons.append({'cohort':cohort,'seed':seed,'condition':c,'difference':values[1]-values[0]})
    stats=[]
    for cohort in ('closed_primary','closed_extensions','closed_joint'):
        subset=[r for r in comparisons if r['cohort']==cohort];seeds=sorted({r['seed'] for r in subset})
        for c in ['ALL']+sorted({r['condition'] for r in subset}):
            diffs=[np.mean([r['difference'] for r in subset if r['seed']==seed and (c=='ALL' or r['condition']==c)]) for seed in seeds]
            rng=np.random.default_rng(20261009)
            lo,hi=np.quantile(np.mean(rng.choice(diffs,size=(5000,len(seeds)),replace=True),axis=1),[.025,.975])
            stats.append({'cohort':cohort,'condition':c,'n_seed':len(seeds),'difference_M3_minus_B1_m_s':float(np.mean(diffs)),
                          'ci95_low':float(lo),'ci95_high':float(hi),'posthoc_secondary':True})
    table('matched_velocity_windows',rows);table('matched_velocity_differences',stats)
    write('matched_velocity_differences.json',stats)
    return stats


def figures(records,groups,differences,timings,common):
    lookup={(r['cohort'],r['condition'],r['method']):r for r in groups}
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,prefix,label in zip(axes,('J','I'),('Суставы','IMU')):
        for j,m in enumerate(METHODS):
            v=[lookup[('closed_extensions' if d==60 else 'closed_primary',f'{prefix}{d}',m)] for d in (20,40,60,80)]
            y=np.array([r['fall_rate'] for r in v]);lo=np.array([r['wilson_low'] for r in v]);hi=np.array([r['wilson_high'] for r in v])
            ax.errorbar(np.array([20,40,60,80])+(j-1.5)*.6,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o-',capsize=3,color=COLORS[m],label=m)
        ax.set(title=label,xlabel='Фактический постоянный возраст после разгона, мс',ylabel='Доля падений',ylim=(-.04,1.04),xticks=[20,40,60,80]);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Закрытая проверка: N=40 seed/точку. Wilson 95%. Условие 60 мс — расширение.')
    save(fig,'falls')
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,metric,title,mult in zip(axes,('fallen','alive_s','velocity_rmse'),('Падения, п.п.','T_alive, с','RMSE скорости, м/с'),(100,1,1)):
        for j,m in enumerate(('M','M2','M3')):
            v=[next(r for r in differences if r['cohort']=='closed_primary' and r['condition']==c and r['method']==m and r['metric']==metric) for c in PRIMARY]
            y=np.array([r['difference_method_minus_B1'] for r in v])*mult
            lo=np.array([r['ci95_low'] for r in v])*mult;hi=np.array([r['ci95_high'] for r in v])*mult
            ax.errorbar(np.arange(6)+(j-1)*.12,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',capsize=3,color=COLORS[m],label=m)
        ax.axhline(0,color='black',lw=.8);ax.set(title=title+' — метод минус B1',xticks=range(6),xticklabels=PRIMARY);ax.tick_params(axis='x',labelrotation=45);ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Парный bootstrap 95%, seed целиком, N=40. RMSE усечён до падения.')
    save(fig,'paired_differences')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,c in zip(axes.flat,PRIMARY):
        for m in METHODS:
            v=sorted([r for r in records if r['cohort']=='closed_primary' and r['condition']==c and r['method']==m],key=lambda r:r['seed'])
            ax.plot([r['seed'] for r in v],[r['alive_s'] for r in v],'.-',lw=.6,markersize=3,color=COLORS[m],label=m)
        ax.set(title=c,xlabel='Seed',ylabel='T_alive, с',ylim=(0,21));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Время до подтверждённого падения: N=40. Значение 20 с — цензурирование.')
    save(fig,'alive_by_seed')
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,metric,title in zip(axes.flat,('q_error_rad','dq_error_rad_s','omega_error_rad_s','g_vector_error','action_rmse'),
                              ('Ошибка q, рад','Ошибка dq, рад/с','Ошибка omega, рад/с','Ошибка g, компоненты единичного вектора','Ошибка действий, безразм.')):
        for i,m in enumerate(METHODS):
            v=[next(r for r in common if r['condition']==c and r['method']==m)[metric] for c in ('J80','I80')]
            ax.bar(np.arange(2)+(i-1.5)*.2,v,.2,label=m,color=COLORS[m])
        ax.set(title=title,xticks=[0,1],xticklabels=['J80','I80'],ylim=(0,None));ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    axes[1,2].axis('off')
    axes[1,2].text(0,.8,'N=1 seed (26) на условие.\nПосле разгона 2 с до падения B1.\nОшибки действий — к свежему входу\nпри той же копии LSTM.\nБез оценки межэпизодной неопределённости.',va='top')
    fig.suptitle('Общая B1-траектория, одинаковая LSTM. Диагностика разработки, не закрытая проверка.')
    save(fig,'state_and_actions')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,metric,title in zip(axes,('fall_rate','velocity_rmse'),('Доля падений','RMSE скорости, м/с')):
        for i,m in enumerate(METHODS):
            v=[lookup[('closed_joint',c,m)] for c in JOINT]
            y=np.array([r[metric] for r in v])
            if metric=='fall_rate':
                lo=np.array([r['wilson_low'] for r in v]);hi=np.array([r['wilson_high'] for r in v])
                ax.errorbar(np.arange(3)+(i-1.5)*.1,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',capsize=3,color=COLORS[m],label=m)
            else:
                intervals=[]
                for c in JOINT:
                    values=np.array([r['velocity_rmse'] for r in records if r['cohort']=='closed_joint' and r['condition']==c and r['method']==m])
                    rng=np.random.default_rng(20261009)
                    intervals.append(np.quantile(np.mean(rng.choice(values,size=(5000,len(values)),replace=True),axis=1),[.025,.975]))
                lo,hi=np.array(intervals).T
                ax.errorbar(np.arange(3)+(i-1.5)*.1,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',capsize=3,color=COLORS[m],label=m)
            ax.set(title=title,xticks=range(3),xticklabels=JOINT,ylim=(-.04,1.04) if metric=='fall_rate' else (0,None));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Совместные задержки: N=10. Wilson / bootstrap 95%. RMSE усечён до падения.')
    save(fig,'joint_delays')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,prefix in zip(axes,('J','I')):
        for d in (20,40,60,80):
            cohort='closed_extensions' if d==60 else 'closed_primary'
            p=ROOT/f'{cohort}/seed_100/{prefix}{d}/B1/trajectory.csv'
            x=np.genfromtxt(p,delimiter=',',names=True)
            ax.step(x['time_s'],x['age_joints_s' if prefix=='J' else 'age_imu_s']*1000,where='post',label=f'{d} мс')
        ax.set(title=prefix+' — фактический возраст',xlabel='Симуляционное время, с',ylabel='Возраст, мс',xlim=(1.8,2.4),ylim=(-3,85));ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Seed 100/B1. Пакеты свежие до 2 с включительно; измерение в моменты решений 50 Гц.')
    save(fig,'actual_ages')
    fig,axes=plt.subplots(1,3,figsize=(11,4),constrained_layout=True)
    for ax,key,title in zip(axes,('adapter_ms','inference_ms','control_cycle_ms'),('Адаптер','Политика','Полный цикл')):
        for offset,stat in ((-.18,'median'),(.18,'p95')):ax.bar(np.arange(4)+offset,[timings[m][key][stat] for m in METHODS],.36,label=stat)
        ax.set(title=title,ylabel='мс',xticks=range(4),xticklabels=METHODS,ylim=(0,None));ax.legend()
    fig.suptitle('CPU, один поток. N циклов B1/M/M2/M3: '+ '/'.join(str(timings[m]['adapter_ms']['n_cycles']) for m in METHODS)+'\nЦикл без загрузки/сохранения/рендера.')
    save(fig,'cpu_timing')


def report(groups,conclusion,timings,common):
    before=(ROOT/'REPORT.md').read_text()
    # Keep the dated preregistration verbatim in one document.
    original=before.split('\n\n<a id="preregistered"></a>\n\n')[-1] if '<a id="preregistered">' in before else before
    selection=json.loads((ROOT/'selection.json').read_text());profile=json.loads((ROOT/'frozen/profile.json').read_text())
    initial_diag=json.loads((ROOT/'diagnostics/common_trajectory.json').read_text())
    def initial_action(c,m):return float(np.mean([r['action_rmse'] for r in initial_diag if r['condition']==c and r['variant']==m]))
    lines=['# M3: замкнутая проверка линейного адаптера H1 на CPU','',
           'Выполнены диагностика, обучение на seed 20–25, подбор на seed 26–29 и полная закрытая серия seed 100–139. Никакой подстройки после открытия закрытой серии не проводилось. Старые веса, физика, PD, нормализация, команда и критерий падения сохранены.','',
           'Объём: 16 диагностических B1/oracle-эпизодов и 3 технических повтора; 84 эпизода сбора; 224 эпизода подбора на отдельных потоках; 48 разработочных совместных; 4 clean и 2 общие диагностические траектории; 960 основных закрытых + 320 отдельных 60 мс + 120 совместных закрытых. Технические CLI/регрессионные повторы перечислены в checks/. Это объём вычислений, а не число независимых наблюдений.','',
           '## Основной результат','',
           f"Снижение частоты падений B1−M3 по исходным шести E2: **{conclusion['fall_rate_improvement_B1_minus_M3']*100:.2f} п.п.**, парный 95% интервал **[{conclusion['paired95_interval'][0]*100:.2f}; {conclusion['paired95_interval'][1]*100:.2f}] п.п.**. Основной критерий {'выполнен' if conclusion['main_hypothesis_criterion_met'] else 'не выполнен'}. Это результат выбранного H1 checkpoint в данной модели задержек, а не доказательство переноса на реального робота.",'',
           'Критерий: минимум 10 п.п. снижения падений и положительная нижняя граница парного интервала. 40 независимых seed, по шесть условий на seed. 960 эпизодов четырёх методов не являются 960 независимыми начальными состояниями.','',
           'Сравниваются сохранённые выбранные профили: B1 удерживает показания; M — кинематический прогноз с горизонтами 20/20 мс; M2 — прежний составной профиль 120/20 мс с периодом 0,8 с и весом 0,5; M3 — новый профиль 80/80 мс. Это сравнение управляющих методов целиком, не изолированный тест влияния горизонта.','',
           '## Что изменено и почему','',
           'M3 — ridge-прогноз по трём доставленным пакетам каждого потока, возрастам и промежуткам sample-time, известной команде и фазе. Полный вариант имеет четыре предыдущих выданных действия; текущий результат политики недоступен. Это объявленное расширение интерфейса относительно M/M2. Сравнение без действий сохраняет команду/фазу и доставленную историю.','',
           'Абляция исключает историю действий только из признаков адаптера. Штатное предыдущее действие в 41-мерном входе политики сохранено во всех вариантах. Команда в обучении постоянная; наличие её признаков само по себе не обучает перенос на другие скорости.','',
           'Для суставов регрессия совместно минимизирует ошибку q (масштаб 0,05 рад) и dq (масштаб 1 рад/с). Кривая: q=q_packet+dq_packet·tau+a·s²+b·s³, dq=dq_packet+(2a·s+3b·s²)/0,08, s=tau/0,08. Коэффициенты линейны по стандартизованным признакам. Вес поправки применяется к a/b совместно. Ограничения коэффициентов действуют на весь сегмент; XML-clipping задаёт нулевую производную на насыщенном участке. Согласованность относится к одному прогнозному сегменту при фиксированных признаках, не обещает глобально гладкой последовательности переоценок.','',
           'IMU: линейные остаточные поправки к omega и кинематически вращённой g, затем нормализация. Совместная динамическая согласованность omega/g не накладывается. При age=0 — побитовое копирование. Горизонты 80/80 мс; положения ограничены XML. Ограничение горизонта не превращает старые данные в свежие. Фаза остаётся фазой штатной политики (период 0,8 с); адаптивная оценка периода не заявляется.','',
           f"Выбран **{selection['selected']['id']}**, ridge=1 после стандартизации. Модель **{profile['model_sha256']}**. [Профиль](frozen/profile.json), [план закрытой проверки](frozen/test_plan.json), [хеши кода](frozen/code_sha256.json), [модель](frozen/model.npz).",'',
           'Доверие по прошлым прогнозам не добавлялось: диагностика обосновала приоритет omega и согласованного q/dq, но не дала отдельного основания расширять заранее ограниченный подбор дополнительным механизмом доверия. Возврат к удержанию показаний не называется безопасным режимом.','',
           '## Диагностика и доступ к истине','',
           'На seed 20/21 J80 оба B1 упали; отдельный oracle q или dq убрал падения, но RMSE оставался выше порога 0,2. При I80 oracle omega убрал оба падения, oracle g вызвал более ранние падения (4,670/4,504 с против B1 6,328/6,332 с). Oracle обоих компонентов потока воспроизвёл штатную ходьбу. Это только N=2, физические траектории oracle различаются.','',
           'На общей записанной B1-траектории замена omega сильнее снижала ошибку действий в I80, а q — в J80. Каждая проверка использовала одну и ту же копию состояния LSTM; перед рабочим вызовом buffers восстановлены. B1 с дополнительными вызовами побитово совпал с обычным запуском. После обучения добавлена общая траектория seed 26: он не входил в ridge-fit, но участвовал в подборе, поэтому это диагностика разработки, не независимая проверка.','',
           f"Первый приоритет — omega: средняя ошибка действий в I80 на общей истории B1 снизилась {initial_action('I80','B1'):.3f} → {initial_action('I80','oracle_omega'):.3f}; oracle omega убрал оба падения. Следующий приоритет — q вместе с согласованной dq: при J80 oracle q снизил ошибку действий {initial_action('J80','B1'):.3f} → {initial_action('J80','oracle_q'):.3f}, а отдельные oracle q/dq каждый убрали падения. Эти начальные ошибки действий включают свежий разгон и усечены падением B1; N=2 seed, единицы — безразмерный выход политики. Причина различий замкнутых траекторий этим не доказана.",'',
           '[Oracle-эпизоды](diagnostics/closed_loop.json), [начальная диагностика действий](diagnostics/common_trajectory.json), [M3 на общей траектории](diagnostics/M3_common_trajectory.json). Истина применяется только к oracle и обучающим целям, не входит в FeatureHistory или рабочий LinearCompensator. Начальный t=0 пакет с физической истиной в сборке обучения допустим, потому что он реально доставлен без задержки.','',
           '## Все кандидаты подбора','', '| Кандидат | Действия | Вес | Падения / 24 | Успехи / 24 | T_alive, с | RMSE, м/с |','|---|---|---:|---:|---:|---:|---:|']
    for r in selection['candidates']:lines.append(f"| {r['id']} | {r['actions']} | {r['gain']} | {r['falls']} | {r['successes']} | {r['mean_alive_s']:.3f} | {r['velocity_rmse']:.3f} |")
    lines+=['','Четыре кандидата записаны до запусков. Подбор — только исходные E2, четыре seed целиком. 84 обучающих эпизода B1/M2 на шести seed, 25919 суставных и 24364 IMU обучающих строк; соседние строки не делились между обучением и подбором. Ошибка внутреннего смешанного ridge-loss не считается физической RMSE. Прерванный первый медленный проход подготовки NPZ и его причина сохранены в models/preparation_issue.json; после кеширования массивов обучение заняло около 5 с на модель. Неудачные кандидаты не удалены.','',
        '## Закрытая серия: исходные шесть E2','', '| Условие | Метод | Падения / N | T_alive, с | RMSE, м/с | Успешная ходьба / N |','|---|---|---:|---:|---:|---:|']
    for c in ['ALL']+PRIMARY:
        for m in METHODS:
            r=next(v for v in groups if v['cohort']=='closed_primary' and v['condition']==c and v['method']==m)
            lines.append(f"| {c} | {m} | {r['falls']}/{r['n_episodes']} | {r['mean_alive_s']:.3f} | {r['velocity_rmse']:.3f} | {r['successes']}/{r['n_episodes']} |")
    lines+=['','RMSE усечён до подтверждённого падения после 2 с. Собственные траектории методов различаются. Улучшение RMSE аварийного эпизода не доказывает устойчивости; успех требует одновременно отсутствия падения и RMSE≤0,2. T_alive ограничено 20 с, не павшие эпизоды цензурированы. Bootstrap 5000, RNG 20261009, повторная выборка seed целиком. Wilson для отдельных условий; для ALL не применялся наивный Wilson по 240 коррелированным эпизодам.','']
    def get(c,m):return next(r for r in groups if r['cohort']=='closed_primary' and r['condition']==c and r['method']==m)
    bi,mi,bj,mj=get('I80','B1'),get('I80','M3'),get('J80','B1'),get('J80','M3')
    lines += [f"При I80 падения B1/M3: {bi['falls']}/40 → {mi['falls']}/40; RMSE {bi['velocity_rmse']:.3f} → {mi['velocity_rmse']:.3f} м/с, успешная ходьба M3 {mi['successes']}/40. При J80 падения {bj['falls']}/40 → {mj['falls']}/40, среднее T_alive {bj['mean_alive_s']:.3f} → {mj['mean_alive_s']:.3f} с. Эти условия показывают разные управляющие последствия, которые нельзя скрывать одним усреднённым RMSE.", '',
              'Отслеживание команды в условиях без падений:','', '| Условие | B1 RMSE, м/с | M3 RMSE, м/с | Изменение, % |','|---|---:|---:|---:|']
    for c in PRIMARY:
        b,m=get(c,'B1'),get(c,'M3')
        if b['falls']==m['falls']==0:lines.append(f"| {c} | {b['velocity_rmse']:.4f} | {m['velocity_rmse']:.4f} | {(m['velocity_rmse']/b['velocity_rmse']-1)*100:+.2f} |")
    lines+=['','Положительный процент здесь означает ухудшение. Основной критерий падений не содержит требования улучшить каждую ошибку скорости, поэтому выполнение критерия не означает универсального улучшения отслеживания.','',
            '### Ошибки восстановления на собственных траекториях','',
            '| Условие | Метод | q RMSE, рад | dq RMSE, рад/с | omega RMSE, рад/с | g, средний угол, рад |','|---|---|---:|---:|---:|---:|']
    for c in PRIMARY:
        for m in METHODS:
            r=get(c,m);lines.append(f"| {c} | {m} | {r['q_rmse_rad']:.4f} | {r['dq_rmse_rad_s']:.4f} | {r['omega_rmse_rad_s']:.4f} | {r['gravity_error_rad']:.4f} |")
    lines+=['','Средние эпизодных ошибок после 2 с до падения, N=40 seed на условие. Сравнение не изолирует ошибки входов: физические движения методов различаются. Парные 95% интервалы ошибок сохранены в tables/paired_differences.csv; дополнительные интервалы описательные, без поправки на множественные сравнения. Для изоляции воздействия входов используется общая B1-траектория ниже.','',
            '### Общая записанная траектория: восстановление и действия','',
            '| Условие | Метод | q RMSE, рад | dq RMSE, рад/с | omega RMSE, рад/с | g RMSE компоненты | Действия RMSE |','|---|---|---:|---:|---:|---:|---:|']
    for r in common:
        lines.append(f"| {r['condition']} | {r['method']} | {r['q_error_rad']:.4f} | {r['dq_error_rad_s']:.4f} | {r['omega_error_rad_s']:.4f} | {r['g_vector_error']:.4f} | {r['action_rmse']:.4f} |")
    cb,cm=[next(r for r in common if r['condition']=='J80' and r['method']==m) for m in ('B1','M3')]
    lines += ['',f"В диагностическом J80 M3 уменьшил dq RMSE {cb['dq_error_rad_s']:.3f} → {cm['dq_error_rad_s']:.3f} рад/с и ошибку действий {cb['action_rmse']:.3f} → {cm['action_rmse']:.3f}. Однако в реальных закрытых J80 его q RMSE выросла {bj['q_rmse_rad']:.3f} → {mj['q_rmse_rad']:.3f} рад, и падение наступало раньше. Локальное улучшение на чужой фиксированной траектории не гарантирует устойчивости собственного замкнутого движения. Причина расхождения этими данными не установлена."]
    lines+=['','Одна диагностическая траектория seed 26 на условие, после 2 с до падения B1. Свежий вход и все методы используют одинаковые предыдущие действия B1, фазу и копию LSTM перед каждым решением. Свежая проверка заменяет текущий вход, но не накопленную память B1 на память идеального контроллера; это локальная чувствительность на данной истории. Действия безразмерные, до штатного масштаба 0,25. Межэпизодная неопределённость здесь не оценивается.','',
            '### Вторичный анализ одинакового окна скорости','',
            'Дополнительная проверка чувствительности к усечению: для B1/M3 одинаковое окно [2 с; min(T_alive B1, T_alive M3)), включая цензурирование 20 с. Анализ добавлен после фиксации метода, не участвовал в выборе и не заменяет основной критерий. Траектории всё ещё различаются.','',
            '| Условие | M3−B1 RMSE, м/с | Парный 95% интервал |','|---|---:|---|']
    for r in json.loads((ROOT/'matched_velocity_differences.json').read_text()):
        if r['cohort']=='closed_primary':lines.append(f"| {r['condition']} | {r['difference_M3_minus_B1_m_s']:+.4f} | [{r['ci95_low']:+.4f}; {r['ci95_high']:+.4f}] |")
    lines += ['','## Расширения: 60 мс и совместные задержки','', '| Серия | Условие | Метод | Падения / N | T_alive, с | RMSE, м/с | Успехи / N |','|---|---|---|---:|---:|---:|---:|']
    for r in groups:
        if r['cohort']!='closed_primary' and r['condition']!='ALL':lines.append(f"| {r['cohort']} | {r['condition']} | {r['method']} | {r['falls']}/{r['n_episodes']} | {r['mean_alive_s']:.3f} | {r['velocity_rmse']:.3f} | {r['successes']}/{r['n_episodes']} |")
    lines+=['','J60/I60: все 40 закрытых seed, 320 эпизодов. Совместные J40I40/J60I40/J40I60: ограниченные seed 100–109, 120 эпизодов, отдельный анализ. Эти условия не заменяют исходный основной критерий. Обучение проводилось на отдельных потоках, совместный режим — дополнительная проверка вне обучающего распределения.','',
        'Фактическое квантование: 50/60 мс дают одинаковые sample-time и возраст 60 мс; 70/80 — 80 мс. Время доставки пакета в metadata может отличаться, но показания на решениях одинаковы. J50/J60 дополнительно проверены побитово в реальной замкнутой симуляции; все четыре пары проверены очередями. Они не считаются независимыми условиями. [Аудит](ages/equivalence.json).','',
        '## CPU и проверки','', '| Метод | N циклов | Адаптер median/p95, мс | Политика median/p95, мс | Цикл median/p95, мс |','|---|---:|---|---|---|']
    for m in METHODS:
        parts=timings[m];values=[f"{parts[k]['median']:.4f}/{parts[k]['p95']:.4f}" for k in ('adapter_ms','inference_ms','control_cycle_ms')]
        lines.append('| '+m+' | '+str(parts['adapter_ms']['n_cycles'])+' | '+' | '.join(values)+' |')
    lines+=['','Intel i9-12900H, Python 3.10.12, Torch 2.5.1+cpu, MuJoCo 3.2.7, NumPy 1.26.4; один поток. Времена объединены по реальным тактам 240 закрытых E2-эпизодов каждого метода; адаптер включает update и context/predict. Цикл исключает setup, финальную запись и рендер. Headless без sleep. Возможность укладываться в 20 мс относится к данному компьютеру и нагрузке, не гарантирует real-time в произвольной среде.','',
        'Прошли 8 новых unit-проверок M3 и 14 прежних пакетов: нулевой возраст, причинность, старые приходы, рост возраста, постоянная скорость, знак/норма g, предел горизонта, аналитическая dq включая ограничения коэффициентов и XML-clipping, исключение действий в абляции. Clean B1/M/M2/M3 побитово одинаков. Для каждого обычного эпизода проведён аудит очередей; M3 дополнительно проверяет свежие блоки и норму g. Во всех закрытых парах проверены хеши состояний/расписаний/ассетов, команда и fall_rule. [Финальный аудит](final_audit.json).','',
        'Технические повторы B1/M/M2 на прежнем seed 10/J40 побитово совпали со всеми прежними нетайминговыми сигналами. CLI M3 с автоматически прочитанным профилем побитово совпал с закрытым seed 100/I80. [Проверка](checks/reproduction/identity.json), реальные вызовы и CLI-лог рядом. Эти четыре повтора не добавлены в статистические когорты.','',
        '## Графики и таблицы','',
        '- [Падения](figures/falls.png): N=40, Wilson 95%, масштаб доли 0–1.',
        '- [Парные разности к B1](figures/paired_differences.png): bootstrap 95%, seed целиком.',
        '- [T_alive по seed](figures/alive_by_seed.png): 20 с означает цензурирование.',
        '- [Ошибки состояния и действий](figures/state_and_actions.png): общая траектория, N=1 диагностический seed 26.',
        '- [Фактические возраста](figures/actual_ages.png): seed 100, сетка решения 20 мс.',
        '- [CPU](figures/cpu_timing.png): median/p95.',
        '- [Совместные задержки](figures/joint_delays.png): N=10, отдельно от основной проверки.',
        '- [CSV](tables/): episodes, groups, paired_differences, cpu_timing, ages, common_trajectory, common_action_timeline и matched_velocity_*.',
        '','Рядом с каждым PNG есть PDF. Масштабы ошибок начинаются с нуля; кривые задержек соединяют только измеренные точки и не доказывают поведения между ними. Точки разных методов слегка разнесены по горизонтали для видимости; реальные возраста — 20/40/60/80 мс. Время по seed показано на шкале 0–20 с.','',
        '## Ограничения и продолжение','',
        'Снижение падений, точность восстановления и отслеживание скорости оцениваются отдельно. Ограничения: одна политика H1, одна скорость, малые начальные возмущения, ровный пол; PD получает свежую обратную связь. Oracle не является реализуемым контроллером. Совместная omega/g динамика и адаптивный период не решены. После просмотра seed 100–139 они считаются открытыми и не могут снова служить закрытой проверкой новой настройки. Для M4 понадобятся новые данные разработки и новая независимая серия.','',
        'M3 в этой серии не проверен при jitter, выпадениях пакетов, возрасте больше 80 мс, смене команд и стоянии. Математические проверки причинности и ограничения горизонта не заменяют замкнутую проверку устойчивости в этих режимах.','',
        'Пакеты формируются на 50 Гц: суставной блок содержит q/dq десяти управляемых суставов, IMU — body omega и направление g из ориентации MuJoCo. Сырые ускорения, дрейф/фильтры реальной IMU и задержка локального PD не моделируются. Остальные признаки — команда, прежнее действие и фаза — передаются штатно.','',
        'Интервалы отражают только вариацию заданных seed (начальные суставные возмущения ±0,003 рад, генератор SeedSequence([seed,20261009])). Постоянные задержки не дают случайной вариации расписания. Вырожденный bootstrap-интервал возможен при одинаковой парной разности исходов у всех seed; он не означает нулевой неопределённости при смене скорости, пола, checkpoint или реальной аппаратуры.','',
        'Дальше имеет смысл отдельно выбирать суставной и IMU-модули на новых данных разработки: сравнить learned omega с прежними суставными M/M2 и новым согласованным q/dq, затем зафиксировать следующий вариант до новой проверки. Это гипотеза M4, а не дополнительная настройка M3 по открытым seed. Также нужны совместные задержки в обучающих эпизодах, переходы команд и другие готовые политики.','',
        'Возможное доверие по прошлым ошибкам требует заранее сохранить прогноз на момент sample-time до прихода проверяемого пакета; при его доставке сравнить сохранённый прогноз именно с этим временем, не переоценивать его с использованием нового пакета. Затем отдельно проверить, снижает ли такой механизм падения; удержание старых данных не объявляется безопасным. Любой положительный эффект здесь не гарантирует переноса на реального робота.','',
        '## Воспроизведение','',
        'Из корня проекта, в существующей .venv, новый каталог вывода:', '', '```bash',
        'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\',
        '  .venv/bin/python -I scripts/run_sensor_episode.py \\',
        '  --initial results/m3_h1/inputs/seed_100/initial.json \\',
        '  --schedule results/m3_h1/inputs/seed_100/I80.json \\',
        '  --method M3 --output results/user_runs/M3_seed100_I80', '```','',
        'Профиль по умолчанию — configs/m3_profile.json. Канонический замороженный профиль — frozen/profile.json. Фактические вызовы эпизодов — runs.jsonl; команды стадий — commands.json; логи — logs/. Анализ: `.venv/bin/python -I scripts/analyze_m3.py`. Технические повторы не увеличивают N. Публикация и отправка материалов не выполнялись.','',
        '<a id="preregistered"></a>','',original]
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    for file in ('closed_primary/completed.json','closed_extensions/completed.json'):
        assert (ROOT/file).exists(),'Cannot report an unfinished full study'
    configure_threads(1)
    records,timings=load_records();groups,differences,conclusion=paired_stats(records);common=common_diagnostics()
    matched_velocity_windows(records)
    assert len([r for r in records if r['cohort']=='closed_primary'])==960
    assert len([r for r in records if r['cohort']=='closed_extensions'])==320
    assert len([r for r in records if r['cohort']=='closed_joint'])==120
    hashes=json.loads((ROOT/'frozen/code_sha256.json').read_text())
    code_matches={name:sha(Path('scripts')/name)==expected for name,expected in hashes.items()}
    assert all(code_matches.values()),'Frozen method/workflow source changed'
    profile=json.loads((ROOT/'frozen/profile.json').read_text())
    assert sha(ROOT/'frozen/model.npz')==profile['model_sha256']
    assert sha(Path('configs/m3_profile.json'))==sha(ROOT/'frozen/profile.json')
    audit_count=0
    for r in records:
        if not r['cohort'].startswith('closed_'):continue
        a=json.loads(Path(r['source']).with_name('audit.json').read_text())
        assert a['causality_and_payload_checks']=='passed'
        audit_count+=1
    assert audit_count==1400
    reproduction=json.loads((ROOT/'checks/reproduction/identity.json').read_text())
    assert set(reproduction)=={'B1','M','M2','M3_CLI'}
    assert all(r['packet_audit']=='passed' and r['bitwise_equal_signal_keys'] for r in reproduction.values())
    opened=json.loads((ROOT/'frozen/opened.json').read_text())
    closed_calls=[json.loads(line) for line in (ROOT/'runs.jsonl').read_text().splitlines() if 100<=json.loads(line)['seed']<140]
    assert len(closed_calls)==1400
    assert min(r['started_unix'] for r in closed_calls)>=opened['opened_unix']
    assert opened['profile_sha256']==sha(ROOT/'frozen/profile.json')
    training=next(iter(np.load(ROOT/'frozen/model.npz',allow_pickle=False)['metadata'].reshape(-1)))
    metadata=json.loads(str(training));assert metadata['train_seeds']==list(range(20,26))
    write('final_audit.json',{'closed_primary_episodes':960,'closed_single_extensions':320,'closed_joint_episodes':120,
          'no_retuning':True,'paired_identity_checks':'passed','per_episode_packet_audits':'passed',
          'audited_closed_episodes':audit_count,'frozen_code_checks':code_matches,'default_profile_matches_frozen':True,
          'steady_observation_ages_match_conditions':True,
          'closed_episodes_use_no_oracle_or_diagnostic_forwards':True,'per_episode_frozen_source_model_checks':True,
          'legacy_B1_M_M2_and_M3_cli_bitwise_repeats':'passed',
          'all_closed_calls_after_opening':True,'frozen_training_seeds':metadata['train_seeds'],
          'model_hash_matches_frozen':True})
    figures(records,groups,differences,timings,common);report(groups,conclusion,timings,common)
    write('analysis_manifest.json',{'analysis_source':'scripts/analyze_m3.py','analysis_sha256':sha(Path('scripts/analyze_m3.py')),
          'verification_source_sha256':sha(Path('scripts/check_m3_reproduction.py')),
          'profile_sha256':sha(ROOT/'frozen/profile.json'),'model_sha256':sha(ROOT/'frozen/model.npz'),
          'files':{str(p.relative_to(ROOT)):sha(p) for folder in ('tables','figures') for p in sorted((ROOT/folder).glob('*')) if p.is_file()}})
    print(json.dumps(conclusion,indent=2),flush=True)

if __name__=='__main__':main()
