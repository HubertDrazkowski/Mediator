"""Load editable experiment settings and resolve current algorithm names."""
from copy import deepcopy
from .io import ROOT,read_json,safe_relative


def load_config(path):
    from pathlib import Path
    path=Path(path)
    config=read_json(ROOT/path if not path.is_absolute() else path)
    if config.get('experiment') not in ('e1','random','online','transpose','star'):
        raise ValueError('experiment must be e1, random, online, transpose, or star')
    for field in ('horizon','jobs'):
        if type(config.get(field)) is not int or config[field]<1:
            raise ValueError(field+' must be a positive integer')
    safe_relative(config['output'])
    config['checkpoints']=sorted(set(int(t) for t in config['checkpoints']+[config['horizon']] if 0<int(t)<=config['horizon']))
    if not config.get('seeds') or any(type(s) is not int or s<0 for s in config['seeds']):
        raise ValueError('seeds must be a nonempty list of nonnegative integers')
    if len(set(config['seeds']))!=len(config['seeds']):raise ValueError('Duplicate seeds')
    if config.get('policy_snapshot_interval',1000)<1:raise ValueError('policy_snapshot_interval must be positive')
    if config['experiment'] in ('e1','online'):
        if len(config.get('epsilons',[]))!=1:
            raise ValueError('Choose one epsilon per configuration/output folder for unambiguous panels')
        if not config.get('modes') or set(config['modes'])-{'stationary','baseline_switching','random_baseline_piecewise'}:
            raise ValueError('Unknown E1 loss regime')
        if len(set(config['modes']))!=len(config['modes']):raise ValueError('Duplicate modes')
    if config['experiment']=='online':
        if 'random_baseline_piecewise' in config['modes']:
            raise ValueError('The paper online comparison uses stationary and deterministic switching only')
        if not config.get('initial_pairs'):raise ValueError('Select initial calibration sample sizes')
        if any(type(n) is not int or n<0 for n in config['initial_pairs']):raise ValueError('Invalid initial_pairs')
    if config['experiment']=='star':
        safe_relative(config['star']['source_path'])
    if config['experiment']=='transpose':
        if not config['transpose']['n_values'] or any(type(n) is not int or n<3 for n in config['transpose']['n_values']):
            raise ValueError('transpose.n_values must contain integers >=3')
        if not config['transpose']['orientations'] or set(config['transpose']['orientations'])-{'vertex','edge'}:
            raise ValueError('transpose.orientations must select vertex and/or edge')
    return config


def selected_specs(config):
    from .algorithms import default_specs,DISPLAY_NAMES
    specs={s['id']:s for s in default_specs()}
    aliases={v.replace('\n',' '):k for k,v in DISPLAY_NAMES.items()}
    identifiers=[aliases.get(name,name) for name in config['algorithms']]
    if len(set(identifiers))!=len(identifiers):raise ValueError('Duplicate algorithms')
    if not identifiers:raise ValueError('Select at least one algorithm')
    overrides={}
    for name,value in config.get('algorithm_overrides',{}).items():
        key=aliases.get(name,name)
        if key not in identifiers:raise ValueError('Override refers to an unselected algorithm: '+name)
        if key in overrides:raise ValueError('Duplicate override aliases')
        overrides[key]=value
    chosen=[]
    for identifier in identifiers:
        if identifier not in specs:raise ValueError('Unknown algorithm: '+identifier)
        spec=deepcopy(specs[identifier])
        override=deepcopy(overrides.get(identifier,{}))
        unknown=set(override)-{'c','rho','cutoff','cutoff_rule','alpha','options'}
        if unknown:raise ValueError('Unknown algorithm override fields: '+str(unknown))
        calibrated=spec['backend']=='calibrated_tsallis'
        if calibrated:
            if 'options' in override:raise ValueError('Use c/rho/alpha/cutoff for CTsallis; options are unused')
            native=identifier in ('Tsallis-INF','CTsallis-Action','CTsallis-Context')
            if native and 'alpha' in override:raise ValueError('Native standalone methods have no alpha term')
            spec.update(override)
            if spec['cutoff_rule'] not in ('fixed','scaled'):raise ValueError('cutoff_rule must be fixed or scaled')
            if spec['cutoff_rule']=='scaled' and 'cutoff' not in override:
                if spec['c']<=0:raise ValueError('c must be positive')
                spec['cutoff']=16./spec['c']**2
        else:
            if set(override)-{'options'}:raise ValueError('Baseline overrides belong under options (RW uses rv_cutoff)')
            spec['options']={**spec.get('options',{}),**override.get('options',{})}
            if identifier=='EXP4MF-RV':spec['cutoff']=spec['options']['rv_cutoff']
        if override:spec['parameter_provenance']='Explicit configuration override; not the paper default'
        from .algorithms.registry import resolve_spec
        spec=resolve_spec(spec)
        chosen.append(spec)
    return chosen


def smoke_config(config):
    """A distinct, short pipeline check. It is not a paper-result rerun."""
    cfg=deepcopy(config)
    cfg.update(horizon=200,checkpoints=[50,200],seeds=[cfg['seeds'][0]],jobs=1)
    cfg['output']='results/smoke_'+cfg['experiment']
    cfg['dimensions']=[[30,80]]
    cfg['plot']['checkpoint']=200
    cfg['plot']['focus_dimensions']=[30,80]
    if cfg['experiment']=='random':
        cfg['random']['n_graphs']=1
        cfg['random']['extension_n_graphs']=0
        cfg['random']['switching_count']=1
    if cfg['experiment']=='transpose':cfg['transpose']['n_values']=[4]
    return cfg
