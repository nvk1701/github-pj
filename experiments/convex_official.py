"""Versioned CP03 orchestration. Dry-run plans work without training or W&B."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from core.convex_eval import evaluate_history
from core.convex_population import PopulationEvaluator
from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.tracking import DEFAULT_ENTITY, DEFAULT_PROJECT, init_run
from core.trainer import OnlineSGDTrainer
from core.utils import load_config, set_seed
from experiments.convex import validate_config

PROTOCOL = 'convex-cp03-v1'
EXTRA = {'protocol_version','run_category','variance_profile','oracle_tolerance','log_interval'}
LOCKED = dict(steps=2000,batch_size=256,eval_batch_size=512,label_noise_std=float(np.sqrt(.1)),
    L=.25,gradient_variance_per_dimension=.25,D_max_per_dimension=1.,projection_radius_per_dimension=.5,
    eta_init=1.,constant_lr=.1,inverse_sqrt_lr0=.1,shift_mode='estimated',ema_beta=.9,eval_seed_stream=1,
    a=1.,b=-1.,k=4.,ell=8)
CLASSIFICATIONS = dict(distribution='PAPER-SPECIFIED',schedule_profile='PAPER-SPECIFIED',
    radial_oracle='MATHEMATICALLY-DERIVED',population_checks='EXPERIMENTALLY-VERIFIED',
    noise_units_init_EMA_radius_baselines='PROJECT-ASSUMPTION',author_hidden_evaluator='UNKNOWN')


def digest(value):
    """Hash canonical finite JSON for a resolved configuration."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def validate_official(config):
    """Reject schema conflicts and changes to the frozen reproduction profile."""
    if not isinstance(config,dict) or not EXTRA <= set(config):
        raise ValueError('all CP03 metadata keys are required')
    if config['protocol_version'] != PROTOCOL:
        raise ValueError('unsupported protocol version')
    if config['run_category'] not in ('reproduction','pilot','smoke','sensitivity'):
        raise ValueError('unknown run category')
    if config['mode'] not in ('disabled','offline','online'):
        raise ValueError('invalid W&B mode')
    if config['variance_profile'] != 'paper':
        raise ValueError('main CP03 runner uses paper variance, safe diagnostics remain CP02')
    tolerance = config['oracle_tolerance']
    if isinstance(tolerance,bool) or not isinstance(tolerance,(float,int)) or not 1e-13 <= tolerance <= 1e-6:
        raise ValueError('oracle tolerance must be in [1e-13,1e-6]')
    interval = config['log_interval']
    if type(interval) is not int or interval<1:
        raise ValueError('log_interval must be a positive integer')
    base = {k:v for k,v in config.items() if k not in EXTRA}
    base['mode'] = 'disabled'  # Reuse legacy scientific validation without changing its online policy.
    cfg = validate_config(base)
    cfg.update({k:config[k] for k in EXTRA},mode=config['mode'])
    for key,value in LOCKED.items():
        deviations = {'steps','batch_size','eval_batch_size'} if cfg['run_category'] in ('pilot','smoke') else set()
        if cfg['run_category']=='sensitivity':
            deviations = {'ell','label_noise_std'}
        if key not in deviations and cfg[key] != value:
            raise ValueError(f'{key} conflicts with the frozen protocol')
    if not set(cfg['dimensions']) <= {2,8,32,128,512} or not set(cfg['seeds']) <= {0,1,2}:
        raise ValueError('select only frozen dimensions and evaluation seeds')
    return cfg


def build_plan(config):
    """Return deterministic unique run keys/paths without creating any output."""
    cfg = validate_official(config)
    root = Path(cfg['output_dir'])
    root = root.resolve() if root.is_absolute() else (ROOT/root).resolve()
    runs = [dict(key=f'{PROTOCOL}-d{d}-{method}-seed{seed}',dimension=d,method=method,seed=seed,
                 path=str(root/f'd{d}'/method/f'seed{seed}'))
            for d in cfg['dimensions'] for seed in cfg['seeds'] for method in cfg['methods']]
    return dict(protocol_version=PROTOCOL,config=cfg,config_sha256=digest(cfg),classifications=CLASSIFICATIONS,
        comparators=dict(primary='B: deterministic population loss to numerical approximate oracle',
                         secondary='A: deterministic population loss to generating reference'),
        gamma_semantics='EMA of completed learner updates, not true future oracle shift',
        baseline_policy='predeclared untuned LR0=0.1',seed_policy='train seed; eval SeedSequence([seed,1,d])',
        integration_errors='estimated, not certified',
        oracle_policy='Gaussian 1D quadrature, exact-norm cache, never fitted to evaluation samples',
        wandb_requirements=['lab1','protocol/category/method/seed/dimension','source/config hashes','actual endpoint and URL','separate local/remote status'],
        acceptance_gates=['regressions and numerical checks','finite d128/d512 pilots','online API smoke or isolated blocker','clean committed source'],
        unknowns=['author hidden evaluator/noise/init/gamma','paper extraction uncertainty','full paper reproduction fidelity'],
        source=_provenance(),run_count=len(runs),runs=runs)


def _json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def _provenance():
    def git(*args):
        return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    files = ['core/convex_population.py','core/convex_eval.py','core/schedules.py','core/trainer.py',
             'core/drift.py','core/utils.py','core/tracking.py','experiments/convex_official.py',
             'experiments/convex_figures.py','configs/convex_cp03.yaml','docs/convex_cp03_protocol.md']
    endpoint = urlsplit(os.environ.get('WANDB_BASE_URL','https://api.wandb.ai'))
    if endpoint.username or endpoint.password:
        raise ValueError('credentials must not be embedded in endpoint URLs')
    return dict(head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),
        source_uncommitted=bool(git('status','--porcelain')),remote_tracking_tip=git('rev-parse','origin/feat/convex'),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        python=sys.version,packages={p:version(p) for p in ('numpy','scipy','matplotlib','wandb')},
        requested_api_endpoint=endpoint.geturl())


def run_official(config):
    """Run selected cells once, exporting full arrays and explicit local/remote state."""
    from experiments.convex_figures import export_figures, plot_run
    import wandb
    plan = build_plan(config)
    cfg = plan['config']
    provenance = plan['source']
    if cfg['run_category']=='reproduction' and provenance['source_uncommitted']:
        raise ValueError('official reproduction requires committed clean source')
    root = Path(plan['runs'][0]['path']).parents[2]
    root.mkdir(parents=True,exist_ok=False)
    _json(root/'run_manifest.json',dict(plan,source=provenance))
    records,profiles = [],{}
    for cell in plan['runs']:
        directory = Path(cell['path'])
        directory.mkdir(parents=True)
        run = None
        _json(directory/'status.json',dict(local_status='running',remote_verified=False))
        metadata = dict(cfg,**cell,**provenance,config_sha256=plan['config_sha256'],
                        comparator_definitions=plan['comparators'],baseline_policy=plan['baseline_policy'])
        _json(directory/'config.json',metadata)
        try:
            started = time.perf_counter()
            name = ('cp03-online-smoke-' if cfg['run_category']=='smoke' else '')+cell['key']
            run = init_run(dict(experiment='convex',method=cell['method'],seed=cell['seed'],
                dimension=cell['dimension'],mode=cfg['mode'],entity=DEFAULT_ENTITY,project=DEFAULT_PROJECT,
                name=name,tags=['lab1',cfg['run_category'],PROTOCOL,'estimated','paper'],resolved=metadata))
            actual_endpoint=run.settings.base_url
            metadata['actual_api_endpoint']=actual_endpoint
            run.config.update({'actual_api_endpoint':actual_endpoint})
            _json(directory/'config.json',metadata)
            tracking_init = time.perf_counter()-started
            d,seed,method = cell['dimension'],cell['seed'],cell['method']
            oracle_start = time.perf_counter()
            if d not in profiles:
                reference,gamma = generate_spiral_trajectory(cfg['steps'],d,cfg['a'],cfg['b'],cfg['k'],cfg['ell'])
                profiles[d] = (reference,gamma,PopulationEvaluator(reference,cfg['label_noise_std'],tolerance=cfg['oracle_tolerance']))
            reference,gamma,population = profiles[d]
            oracle_seconds = time.perf_counter()-oracle_start
            set_seed(seed)
            schedule = (ConvexSchedule(cfg['L'],d*cfg['D_max_per_dimension'],d*cfg['gradient_variance_per_dimension'],cfg['eta_init'])
                        if method=='convex' else ConstantSchedule(cfg['constant_lr']) if method=='constant'
                        else InverseSqrtSchedule(cfg['inverse_sqrt_lr0']))
            trainer = OnlineSGDTrainer(d,schedule,loss_type='logistic',batch_size=cfg['batch_size'],
                noise_std=cfg['label_noise_std'],max_norm=d*cfg['projection_radius_per_dimension'],
                use_oracle_gamma=False,ema_beta=cfg['ema_beta'],use_wandb=False)
            tick = time.perf_counter()
            history = trainer.run(reference,gamma,seed=seed)
            train_seconds = time.perf_counter()-tick
            eval_seed = int(np.random.SeedSequence([seed,cfg['eval_seed_stream'],d]).generate_state(1)[0])
            tick = time.perf_counter()
            metrics = evaluate_history(history,batch_size=cfg['batch_size'],label_noise_std=cfg['label_noise_std'],
                eval_batch_size=cfg['eval_batch_size'],eval_seed=eval_seed,approx_oracle_path=population.oracle_path)
            mc_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            metrics.update(population.evaluate(history.theta_path),step=np.arange(1,cfg['steps']+1),
                lr=history.lr_history,gamma_history=history.gamma_used,gamma_schedule=history.gamma_used)
            population_seconds = time.perf_counter()-tick
            tick = time.perf_counter()
            np.savez_compressed(directory/'history.npz',theta=history.theta_path,reference=reference,
                                approx_oracle=population.oracle_path,**metrics)
            with (directory/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream)
                writer.writerow(metrics)
                writer.writerows(zip(*metrics.values()))
            serialize_seconds = time.perf_counter()-tick
            tick=time.perf_counter()
            plot_run(directory,history,metrics,population.oracle_path)
            plot_seconds=time.perf_counter()-tick
            tick = time.perf_counter()
            for i in range(cfg['steps']):
                if i%cfg['log_interval']==0 or i==cfg['steps']-1:
                    run.log({k:float(v[i]) for k,v in metrics.items() if k!='step'},step=i+1)
            if cfg['mode']!='disabled':
                artifact = wandb.Artifact(cell['key'],type='convex-evidence',metadata={'config_sha256':plan['config_sha256']})
                for filename in ('history.npz','metrics.csv','config.json','figure.png'):
                    artifact.add_file(str(directory/filename))
                run.log_artifact(artifact)
            logging_seconds = time.perf_counter()-tick
            url = run.url if cfg['mode']=='online' else None
            record = dict(**cell,local_status='complete',run_id=run.id,wandb_url=url,mode=cfg['mode'],
                entity=run.entity,project=run.project,tags=list(run.tags),api_endpoint=actual_endpoint,remote_verified=False,
                uploaded=None if cfg['mode']=='online' else False,
                final_population_A=float(metrics['population_cum_A'][-1]),final_population_B=float(metrics['population_cum_B'][-1]),
                train_seconds=train_seconds,mc_seconds=mc_seconds,population_seconds=population_seconds,
                oracle_seconds=oracle_seconds,serialize_seconds=serialize_seconds,plot_seconds=plot_seconds,logging_seconds=logging_seconds,
                tracking_init_seconds=tracking_init,cache_entries=len(population.cache))
            run.summary.update(final_population_B=record['final_population_B'],completed=True,protocol_version=PROTOCOL)
            run.finish()
            record['total_seconds']=time.perf_counter()-started
            _json(directory/'status.json',record)
            records.append(record)
        except BaseException as error:
            _json(directory/'status.json',dict(local_status='failed',error_type=type(error).__name__,remote_verified=False))
            if run is not None:
                try:
                    run.summary['completed']=False
                except BaseException:
                    pass
                try:
                    run.finish(exit_code=1)
                except BaseException:
                    pass
            raise
    tick=time.perf_counter()
    try:
        export_figures(root,records,cfg)
    except BaseException:
        _json(root/'status.json',dict(local_status='failed_plot_export',remote_verified=False))
        raise
    _json(root/'plot_timing.json',dict(seconds=time.perf_counter()-tick))
    _json(root/'summary.json',dict(protocol_version=PROTOCOL,runs=records))
    _json(root/'status.json',dict(local_status='complete',remote_verified=False))
    return records


def main(argv=None):
    """Print a dry plan or run explicitly selected cells with frozen defaults."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/convex_cp03.yaml')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--mode',choices=['disabled','offline','online'])
    parser.add_argument('--category',choices=['reproduction','pilot','smoke','sensitivity'])
    parser.add_argument('--output-dir')
    for key in ('dimensions','seeds','methods'):
        parser.add_argument('--'+key,nargs='+',type=str if key=='methods' else int)
    for key in ('steps','batch-size','eval-batch-size','ell'):
        parser.add_argument('--'+key,type=int)
    parser.add_argument('--label-noise-std',type=float)
    args=parser.parse_args(argv)
    config=load_config(args.config)
    for key,value in vars(args).items():
        if key not in ('config','dry_run') and value is not None:
            config['run_category' if key=='category' else key]=value
    print(json.dumps(build_plan(config) if args.dry_run else run_official(config),indent=2))


if __name__=='__main__':
    main()
