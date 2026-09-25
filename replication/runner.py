"""Generate a frozen plan, run selected cells, and resume without discarding failures."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import multiprocessing
import os
import platform
import time
import traceback
import numpy as np
from .io import ROOT,safe_relative,stable_seed,write_json,read_json,write_npz,implementation_hashes
from .configuration import selected_specs

TERMINAL={'ok','failed','inapplicable'}


def _plan_config(config):
    return {k:v for k,v in config.items() if k not in ('jobs','plot')}


def prepare(config):
    from .environments import (generate_e1,generate_e1_piecewise,generate_transpose,
                               generate_random_paper,generate_random_switching)
    from .calibration import sample_calibration
    from .algorithms import applicability,Mechanism
    output=safe_relative(config['output']);output.mkdir(parents=True,exist_ok=True)
    path=output/'manifest.json'
    if path.exists():
        manifest=read_json(path)
        if manifest['plan_configuration']!=_plan_config(config):
            raise ValueError('Existing output has a different configuration; choose a new output folder')
        if manifest['implementation_hashes']!=implementation_hashes():
            raise ValueError('Implementation changed since this plan was frozen; choose a new output folder')
        # Presentation choices can change without invalidating computed results.
        manifest['configuration']['plot']=config['plot'];write_json(path,manifest)
        print(f"Existing plan: {len(manifest['tasks'])} cells in {config['output']}",flush=True)
        return output
    if any(output.iterdir()):
        raise ValueError('A new output must be empty; use another output path')
    if config['experiment']=='random':
        rc=config['random']
        generation={k:v for k,v in rc.items() if k not in ('switching_count','selection_seed','L0','gamma')}
        instances=generate_random_paper(horizon=config['horizon'],config=generation,
                                       joint_diagnostics=rc.get('joint_diagnostics',True))
        instances+=generate_random_switching(instances,count=rc.get('switching_count',30),
                      selection_seed=rc.get('selection_seed',2026092401),horizon=config['horizon'],
                      L0=rc.get('L0',50),gamma=rc.get('gamma',1.6))
    elif config['experiment']=='transpose':
        tc=config['transpose']
        instances=[generate_transpose(n,orientation,horizon=config['horizon'],
                     baseline=tc.get('baseline',.35),minimum_action_gap=tc.get('minimum_action_gap',.02),
                     environment_seed=tc.get('environment_seed',2026092307))
                   for n in tc['n_values'] for orientation in tc['orientations']]
    elif config['experiment']=='star':
        from .star import load_star_instance
        instances=[load_star_instance(**config['star'],horizon=config['horizon'])]
    else:
        instances=[]
        for A,Z in config['dimensions']:
            for epsilon in config['epsilons']:
                instances.extend(generate_e1(A,Z,epsilon,config['horizon'],**config['e1']))
                if 'random_baseline_piecewise' in config['modes']:
                    instances.extend(generate_e1_piecewise(A,Z,seed,epsilon=epsilon,
                                     horizon=config['horizon'],**config['e1']) for seed in config['seeds'])
        instances=[x for x in instances if x['metadata']['mode'] in config['modes']]
    if not instances:raise ValueError('The selection contains no environments')
    specs=selected_specs(config);tasks=[];calibrations={};identities=set()
    for inst in instances:
        sid=inst['scenario_id']
        if sid in identities:raise ValueError('Duplicate environment ID')
        identities.add(sid)
        arrays={k:np.asarray(inst[k]) for k in ('M','vectors','starts')}
        for k in ('empirical_action','empirical_mediator','empirical_loss','empirical_mu'):
            if k in inst:arrays[k]=np.asarray(inst[k])
        if inst.get('baseline_groups') is not None:arrays['baseline_groups']=np.asarray(inst['baseline_groups'])
        fingerprint=hashlib.sha256()
        for k,v in sorted(arrays.items()):
            fingerprint.update(k.encode());fingerprint.update(str(v.shape).encode());fingerprint.update(v.tobytes())
        env_hash=fingerprint.hexdigest()
        write_npz(output/'instances'/f'{sid}.npz',**arrays)
        meta=dict(inst['metadata'],scenario_id=sid,environment_hash=env_hash)
        write_json(output/'instances'/f'{sid}.json',meta)
        conditions=[('true',0)]
        if config['experiment']=='online':
            sizes=sorted(set(config['initial_pairs']))
            if any(type(n) is not int or n<0 for n in sizes):raise ValueError('Invalid initial sample sizes')
            conditions += [('online_'+str(n),n) for n in sizes]
        for seed in config['seeds']:
            if inst.get('trajectory_seed',meta.get('trajectory_seed',seed))!=seed:continue
            calibration_paths={}
            if config['experiment']=='online':
                key=(meta['n_actions'],meta['n_contexts'],meta['epsilon'],seed)
                if key not in calibrations:
                    a,z=sample_calibration(arrays['M'],seed,n_max=max(sizes),epsilon=meta['epsilon'])
                    token=hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
                    write_npz(output/'calibration'/f'{token}_pairs.npz',action=a,context=z)
                    fits={}
                    for n in sizes:
                        counts=np.bincount(z[:n]*meta['n_actions']+a[:n],minlength=arrays['M'].size).reshape(arrays['M'].shape)
                        relative=f'calibration/{token}_n{n}.npz'
                        write_npz(output/relative,counts=counts)
                        fits[n]=relative
                    calibrations[key]=fits
                calibration_paths=calibrations[key]
            for spec in specs:
                for condition,n in conditions:
                    online=condition!='true'
                    supplied=arrays['M'] if not online else (np.ones_like(arrays['M'])/meta['n_contexts'])
                    # Positive smoothed estimates meet CUCB2's support requirement.
                    reason=applicability(spec,Mechanism(supplied,arrays.get('baseline_groups')))
                    task=dict(experiment=config['experiment'],scenario_id=sid,environment_hash=env_hash,
                        algorithm=spec['id'],spec=spec,seed=seed,horizon=config['horizon'],
                        checkpoints=config['checkpoints'],mechanism_mode=condition,online=online,
                        calibration_n=n,calibration_path=calibration_paths.get(n) if online else None,
                        prior_strength=config.get('prior_strength',1.),applicable=reason is None,
                        applicability_reason=reason)
                    task['rng_scenario_id']=meta.get('rng_scenario_id',sid)
                    task['sampling']=meta.get('sampling','bernoulli_mediator')
                    if task['sampling']=='empirical_rows':task['dataset']=meta['dataset']
                    identity={k:v for k,v in task.items() if k not in ('experiment','calibration_path')}
                    task['task_id']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
                    tasks.append(task)
    tasks.sort(key=lambda t:(t['seed'],stable_seed('environment_queue',t['scenario_id']),
                             stable_seed('algorithm_queue',t['algorithm']),t['mechanism_mode']))
    manifest=dict(schema_version=1,configuration=deepcopy(config),plan_configuration=_plan_config(config),
        implementation_hashes=implementation_hashes(),algorithms=specs,tasks=tasks,
        instances=sorted(identities),versions=dict(python=platform.python_version(),numpy=np.__version__),
        randomness=('Common action/within-arm row uniforms per trajectory seed; independent learner RNG.'
                    if config['experiment']=='star' else
                    'Common action/context/loss uniforms per environment and trajectory seed; independent learner RNG.'),
        calibration=('Not used in the empirical STAR benchmark.' if config['experiment']=='star' else
                     'Uniform actions; mediator conditional on true M; nested initial samples shared by methods and loss regimes.'),
        online_rule='Issue Mhat_t from t-1 past online observations; update loss estimates before observing the new pair.',
        failure_policy='Retain failures and N/A. No automatic outcome-dependent retries, replacement, or imputation.')
    write_json(path,manifest)
    for task in tasks:
        if not task['applicable']:
            write_json(output/'raw'/f"{task['task_id']}.json",dict(task,status='inapplicable',stopped_round=0,
                checkpoints={},policy_checkpoints={},reason=task['applicability_reason']))
    print(f"Prepared {len(instances)} environments, {len(tasks)} cells ({sum(t['applicable'] for t in tasks)} applicable), output: {config['output']}",flush=True)
    return output


def run_one(task,output,save_full_trajectories=False,snapshot_interval=1000,limit_rounds=None):
    from .algorithms import instantiate,Mechanism,resolved_parameters
    from .calibration import OnlineMechanismEstimate
    output=Path(output);path=output/'raw'/f"{task['task_id']}.json"
    prior=read_json(path) if path.exists() else None
    if prior and prior['status'] in TERMINAL:return prior['status']
    if prior and (any(prior.get(k)!=v for k,v in task.items() if k!='checkpoints')
                  or prior.get('planned_checkpoints')!=task['checkpoints']):
        raise ValueError('Task changed during resume')
    if prior and limit_rounds is not None and limit_rounds<prior['stopped_round']:
        raise ValueError('Verification limit cannot shorten an existing saved prefix')
    record=dict(task,status='running',stopped_round=0,planned_checkpoints=task['checkpoints'],
                checkpoints={},policy_checkpoints={},curve=[],prefix_digest=None)
    started=time.perf_counter();learner=estimator=None;T=task['horizon']
    arrays={k:np.empty(T,dtype=np.float64) for k in ('cumulative_regret','cumulative_policy_regret')}
    arrays.update({k:np.empty(T,dtype=np.int32) for k in ('action','context')})
    arrays['loss']=np.empty(T,dtype=np.float64)
    policy_t=[];policies=[];model_snapshots=[];hasher=hashlib.sha256()
    cumulative=policy_total=0.;prefix_verified=not prior or prior['stopped_round']==0

    def persist():
        n=record['stopped_round'];record['runtime_seconds']=time.perf_counter()-started
        record['prefix_digest']=hasher.hexdigest()
        if learner is not None:record['learner_diagnostics']=learner.diagnostics()
        record['prefix_verified']=prefix_verified
        if save_full_trajectories:
            write_npz(output/'traces'/f"{task['task_id']}.npz",**{k:v[:n] for k,v in arrays.items()},
                      policy_t=np.asarray(policy_t,dtype=np.int32),p=np.asarray(policies))
        if estimator is not None:
            write_npz(output/'estimates'/f"{task['task_id']}.npz",offline_counts=estimator.initial_counts,
                      final_counts=estimator.counts,model_t=np.asarray(policy_t,dtype=np.int32),
                      M_hat=np.asarray(model_snapshots))
        write_json(path,record)

    try:
        with np.load(output/'instances'/f"{task['scenario_id']}.npz",allow_pickle=False) as data:
            M,vectors,starts=data['M'],data['vectors'],data['starts']
            groups=data['baseline_groups'] if 'baseline_groups' in data else None
            empirical=task.get('sampling')=='empirical_rows'
            if empirical:
                row_actions=data['empirical_action'];row_mediators=data['empirical_mediator']
                row_losses=data['empirical_loss'];empirical_mu=data['empirical_mu']
                row_pools=[np.flatnonzero(row_actions==a) for a in range(M.shape[1])]
                arrays['row_index']=np.empty(T,dtype=np.int32)
        supplied=M
        if task['online']:
            with np.load(output/task['calibration_path'],allow_pickle=False) as data:counts=data['counts']
            estimator=OnlineMechanismEstimate(counts,prior_strength=task['prior_strength'])
            supplied=estimator.matrix(1)
        paired=stable_seed('e1_random30_v1',task.get('rng_scenario_id',task['scenario_id']),task['seed'])
        learner_seed=stable_seed(paired,'learner')
        if empirical:
            paired=stable_seed('raw_realdata_v1',task['dataset'],task['seed'])
            learner_seed=stable_seed(task['dataset'],task['seed'],'learner')
        uniforms=np.random.default_rng(paired).random((T,2 if empirical else 3))
        learner=instantiate(task['spec'],Mechanism(supplied,groups),T,
                            np.random.default_rng(learner_seed),online=task['online'])
        record['resolved_parameters']=resolved_parameters(task['spec'],learner)
        record['paired_rng_seed']=paired
        cdf=np.cumsum(M,axis=0);cdf[-1]=1.;phase=-1
        boundaries=set(map(int,starts))|{int(t)-1 for t in starts[1:]}
        write_json(path,record)
        for t in range(1,T+1):
            if (t==1 or t%250==1) and (output/'STOP').exists():
                record['status']='stopped';break
            if limit_rounds is not None and t>limit_rounds:
                record['status']='verification_prefix';break
            if phase<0 or (phase+1<len(starts) and t>=starts[phase+1]):
                phase+=1;g=vectors[phase];mu=empirical_mu if empirical else M.T@g;gaps=mu-mu.min()
            if estimator is not None:
                supplied=estimator.matrix(t)
                if t>1:learner.update_mechanism(supplied)
            p=learner.probabilities(t)
            if not np.all(np.isfinite(p)) or np.any(p<0) or abs(float(p.sum())-1)>1e-9:
                raise ValueError('Invalid action distribution')
            action_cdf=np.cumsum(p);action_cdf[-1]=1.
            a=int(np.searchsorted(action_cdf,uniforms[t-1,0],side='right'))
            if empirical:
                pool=row_pools[a]
                row=int(pool[min(int(uniforms[t-1,1]*len(pool)),len(pool)-1)])
                z=int(row_mediators[row]);loss=float(row_losses[row])
                arrays['row_index'][t-1]=row
            else:
                z=int(np.searchsorted(cdf[:,a],uniforms[t-1,1],side='right'))
                loss=int(uniforms[t-1,2]<g[z])
            cumulative+=float(gaps[a]);policy_total+=float(p@gaps)
            for k,value in dict(action=a,context=z,loss=loss,cumulative_regret=cumulative,cumulative_policy_regret=policy_total).items():arrays[k][t-1]=value
            if t==1 or t%snapshot_interval==0 or t in boundaries or t==T:
                policy_t.append(t);policies.append(p.copy())
                if estimator is not None:model_snapshots.append(supplied.copy())
            # Record the incurred observation even if the subsequent update fails.
            if empirical:
                hasher.update(np.asarray([a,z,row],dtype=np.int64).tobytes())
                hasher.update(np.asarray([loss],dtype=np.float64).tobytes())
            else:hasher.update(np.asarray([a,z,loss],dtype=np.int64).tobytes())
            hasher.update(np.asarray([cumulative,policy_total],dtype=np.float64).tobytes())
            record['stopped_round']=t
            if prior and t==prior['stopped_round']:
                if hasher.hexdigest()!=prior['prefix_digest']:raise ValueError('Replayed prefix differs from saved observations')
                prefix_verified=True
            if t%1000==0 or t in boundaries or t==T:
                record['curve'].append(dict(t=t,regret=cumulative,policy_regret=policy_total))
            if t in task['checkpoints']:
                record['checkpoints'][str(t)]=cumulative;record['policy_checkpoints'][str(t)]=policy_total
            learner.update(t,a,z,float(loss),p)
            if estimator is not None:
                estimator.observe(t,a,z)
                if t in task['checkpoints']:
                    record.setdefault('estimation_diagnostics',{})[str(t)]=estimator.diagnostics(M)
            if t%5000==0:persist()
        else:record['status']='ok'
    except Exception as exc:
        record.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
        # Relative source locations in error text keep exported records portable.
        record['traceback']=traceback.format_exc().replace(str(ROOT),'.')
    record.update(final_regret=cumulative,final_policy_regret=policy_total)
    persist();return record['status']


def run_plan(output,jobs=4,max_tasks=None,limit_rounds=None):
    """Completed and failed runs are retained. Interrupted prefixes replay exactly."""
    output=Path(output);manifest=read_json(output/'manifest.json')
    if manifest['implementation_hashes']!=implementation_hashes():raise ValueError('Frozen implementation differs')
    if (output/'STOP').exists():raise ValueError('STOP exists; remove it deliberately before resuming')
    lock=output/'RUNNING.lock'
    try:
        handle=lock.open('x');handle.write(str(os.getpid()));handle.close()
    except FileExistsError:
        raise RuntimeError('RUNNING.lock exists. Check that no worker is active before removing a stale lock.')
    def status_counts():
        return Counter(read_json(p)['status'] for p in (output/'raw').glob('*.json'))
    try:
        pending=[]
        for task in manifest['tasks']:
            path=output/'raw'/f"{task['task_id']}.json"
            if not path.exists() or read_json(path)['status'] not in TERMINAL:pending.append(task)
        if max_tasks is not None:pending=pending[:max_tasks]
        cfg=manifest['configuration'];completed=0;total=len(pending)
        print(f'Running {total} unfinished cells with {jobs} worker(s).',flush=True)
        with ProcessPoolExecutor(max_workers=jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
            queue=iter(pending);active={}
            def submit():
                if (output/'STOP').exists():return False
                task=next(queue,None)
                if task is None:return False
                f=pool.submit(run_one,task,str(output),cfg.get('save_full_trajectories',False),
                              cfg.get('policy_snapshot_interval',1000),limit_rounds)
                active[f]=task;return True
            for _ in range(jobs):submit()
            while active:
                done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    task=active.pop(future)
                    try:state=future.result()
                    except Exception as exc:
                        write_json(output/'raw'/f"{task['task_id']}.json",dict(task,status='failed',
                            stopped_round=0,checkpoints={},policy_checkpoints={},reason=str(exc)))
                        state='failed'
                    completed+=1
                    print(f'{completed}/{total}: {task["algorithm"]} {task["mechanism_mode"]} seed={task["seed"]}: {state}',flush=True)
                    submit()
                if done:write_json(output/'progress.json',dict(completed_this_invocation=completed,counts=status_counts(),active=len(active),planned=len(manifest['tasks'])))
        write_json(output/'progress.json',dict(completed_this_invocation=completed,counts=status_counts(),active=0,planned=len(manifest['tasks'])))
    finally:lock.unlink(missing_ok=True)
