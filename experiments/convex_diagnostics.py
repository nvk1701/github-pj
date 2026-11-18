"""Small CP02 comparator diagnostics using the unchanged shared SGD trainer."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.convex_eval import evaluate_history
from core.convex_population import gaussian_population_loss_gradient, gaussian_population_oracle
from core.drift import generate_spiral_trajectory
from core.schedules import ConstantSchedule, ConvexSchedule, InverseSqrtSchedule
from core.tracking import init_run
from core.trainer import OnlineSGDTrainer
from core.utils import load_config, set_seed
from experiments.convex import validate_config

PROTOCOL_VERSION = 'convex-cp02-v1-provisional'


class _DiameterBoundSchedule(ConvexSchedule):
    """Use the known domain diameter as a future-free upper bound on shift."""

    def step(self, t, **kwargs):
        kwargs.pop('gamma_prev', None)
        kwargs.pop('gamma', None)
        return super().step(t, gamma=self.D_max, **kwargs)


def validate_protocol_config(config, *, gamma_mode='estimated', variance_profile='paper'):
    """Validate a small diagnostic profile and resolve its explicit variance.

    Diameter-bound mode requires v=d, the proved conservative bound.
    Paper v=d/4 is a reproduction profile without a uniform guarantee.
    Online logging and full reproduction are outside this diagnostic entrypoint.
    """
    if gamma_mode not in ('estimated', 'lagged_reference', 'diameter_bound'):
        raise ValueError('unknown gamma mode')
    if variance_profile not in ('paper', 'safe'):
        raise ValueError('unknown variance profile')
    if gamma_mode == 'diameter_bound' and variance_profile != 'safe':
        raise ValueError('diameter-bound protocol requires safe variance v=d')
    cfg = dict(config)
    cfg['gradient_variance_per_dimension'] = .25 if variance_profile == 'paper' else 1.
    cfg['shift_mode'] = 'lagged_reference' if gamma_mode == 'lagged_reference' else 'estimated'
    cfg = validate_config(cfg)
    if cfg['L'] < .25:
        raise ValueError('Gaussian population smoothness requires an upper bound L>=1/4')
    if cfg['steps'] > 128 or max(cfg['dimensions']) > 32:
        raise ValueError('CP02 diagnostics are limited to T<=128 and d<=32')
    return cfg


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def run_diagnostics(config, *, gamma_mode='estimated', variance_profile='paper'):
    """Export reference and approximate-oracle metrics for the same trajectories.

    Both comparisons reuse each paired evaluation batch. Oracle construction
    and population quadrature are evaluator-only and never enter the learner.
    Each run records numerical comparator error separately from MC SE.
    """
    cfg = validate_protocol_config(config, gamma_mode=gamma_mode, variance_profile=variance_profile)
    output = Path(cfg['output_dir'])
    if not output.is_absolute():
        output = ROOT/output
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dirty = bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip())
    remote_tip = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','origin/feat/convex'],text=True).strip()
    included = subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',head,remote_tip],check=False)
    if included.returncode not in (0,1):
        raise RuntimeError('cannot determine source ancestry against the remote-tracking tip')
    source_files = ['core/convex_population.py','core/convex_eval.py','core/schedules.py','core/trainer.py',
                    'core/drift.py','core/tracking.py','configs/convex.yaml','experiments/convex_diagnostics.py',
                    'docs/convex_protocol.md']
    hashes = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files}
    summaries = []
    for d in cfg['dimensions']:
        reference, gamma = generate_spiral_trajectory(cfg['steps'],d,cfg['a'],cfg['b'],cfg['k'],cfg['ell'])
        oracles = [gaussian_population_oracle(ref,cfg['label_noise_std']) for ref in reference]
        oracle_path = np.array([entry['theta'] for entry in oracles])
        radius = d*cfg['projection_radius_per_dimension']
        if np.max(np.linalg.norm(oracle_path,axis=1)) > radius+1e-12:
            raise ValueError('approximate population comparator lies outside the projection ball')
        for seed in cfg['seeds']:
            eval_seed = int(np.random.SeedSequence([seed,cfg['eval_seed_stream'],d]).generate_state(1)[0])
            for method in cfg['methods']:
                run_dir = output/f'd{d}'/method/f'seed{seed}'
                run_dir.mkdir(parents=True)
                set_seed(seed)
                if method == 'convex':
                    schedule_class = _DiameterBoundSchedule if gamma_mode == 'diameter_bound' else ConvexSchedule
                    schedule = schedule_class(cfg['L'],d*cfg['D_max_per_dimension'],
                                              d*cfg['gradient_variance_per_dimension'],cfg['eta_init'])
                elif method == 'constant':
                    schedule = ConstantSchedule(cfg['constant_lr'])
                else:
                    schedule = InverseSqrtSchedule(cfg['inverse_sqrt_lr0'])
                trainer = OnlineSGDTrainer(d,schedule,loss_type='logistic',batch_size=cfg['batch_size'],
                    noise_std=cfg['label_noise_std'],max_norm=radius,ema_beta=cfg['ema_beta'],
                    use_oracle_gamma=gamma_mode=='lagged_reference',use_wandb=False)
                history = trainer.run(reference,gamma,seed=seed)
                metrics = evaluate_history(history,batch_size=cfg['batch_size'],
                    label_noise_std=cfg['label_noise_std'],eval_batch_size=cfg['eval_batch_size'],
                    eval_seed=eval_seed,approx_oracle_path=oracle_path)
                reference_quad, oracle_quad, errors = [], [], []
                for w,ref,oracle in zip(history.theta_path,reference,oracle_path):
                    loss,_,a = gaussian_population_loss_gradient(w,ref,cfg['label_noise_std'])
                    ref_loss,_,b = gaussian_population_loss_gradient(ref,ref,cfg['label_noise_std'])
                    oracle_loss,_,c = gaussian_population_loss_gradient(oracle,ref,cfg['label_noise_std'])
                    reference_quad.append(loss-ref_loss)
                    oracle_quad.append(loss-oracle_loss)
                    errors.append(a['loss']+max(b['loss'],c['loss']))
                metrics.update(step=np.arange(1,cfg['steps']+1), lr=history.lr_history,
                    gamma_history=history.gamma_used,
                    gamma_schedule=(np.full(cfg['steps'],d*cfg['D_max_per_dimension'])
                                    if gamma_mode=='diameter_bound' and method=='convex' else history.gamma_used),
                    population_reference_excess_quad=np.array(reference_quad),
                    population_approx_oracle_excess_quad=np.array(oracle_quad),
                    population_gap_error_estimate=np.array(errors),
                    oracle_risk_suboptimality_estimate=np.array([o['risk_suboptimality_estimate'] for o in oracles]),
                    oracle_parameter_error_estimate=np.array([o['parameter_error_estimate'] for o in oracles]))
                if not all(np.isfinite(v).all() for v in metrics.values()):
                    raise ValueError('nonfinite diagnostic refused')
                metadata = dict(cfg, dimension=d,seed=seed,eval_seed=eval_seed,method=method,
                    protocol_version=PROTOCOL_VERSION,checkpoint='CP02',run_category='diagnostic',
                    comparison_protocols=['A-generating-reference','B-approximate-population-oracle'],
                    metric_definition='signed paired excess, reference and approximate oracle reported separately',
                    gamma_mode=gamma_mode,variance_profile=variance_profile,head=head,
                    python=sys.version,packages={name:version(name) for name in ('numpy','scipy','wandb')},
                    source_uncommitted=dirty,remote_tracking_tip=remote_tip,source_commit_not_pushed=included.returncode==1,
                    source_sha256=hashes,quadrature_error_status='estimated, not interval-certified')
                _save_json(run_dir/'config.json',metadata)
                with (run_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(metrics)
                    writer.writerows(zip(*metrics.values()))
                np.savez_compressed(run_dir/'history.npz',theta=history.theta_path,reference=reference,
                                    approx_oracle=oracle_path,**metrics)
                run = init_run(dict(experiment='convex-cp02',method=method,seed=seed,mode=cfg['mode'],
                    name=f'cp02-{gamma_mode}-{variance_profile}-d{d}-{method}-seed{seed}',
                    tags=['lab1','diagnostic','CP02',PROTOCOL_VERSION,gamma_mode,variance_profile],resolved=metadata))
                try:
                    for i in range(cfg['steps']):
                        run.log({key:float(value[i]) for key,value in metrics.items() if key!='step'},step=i+1)
                    summary = dict(dimension=d,seed=seed,method=method,gamma_mode=gamma_mode,
                        final_reference_mc=float(metrics['eval_cum_mc_excess'][-1]),
                        final_approx_oracle_mc=float(metrics['eval_cum_mc_approx_oracle_excess'][-1]),
                        final_reference_quad=float(np.sum(reference_quad)),
                        final_approx_oracle_quad=float(np.sum(oracle_quad)),
                        cum_quad_error_estimate=float(np.sum(errors)),
                        max_oracle_parameter_error_estimate=float(np.max(metrics['oracle_parameter_error_estimate'])),
                        run_id=run.id,mode=cfg['mode'],uploaded=False,run_url=None,tags=list(run.tags))
                    run.summary.update(summary)
                except BaseException:
                    try:
                        run.finish(exit_code=1)
                    except BaseException:
                        pass
                    raise
                else:
                    run.finish()
                _save_json(run_dir/'summary.json',summary)
                summaries.append(summary)
    _save_json(output/'summary.json',dict(protocol_version=PROTOCOL_VERSION,runs=summaries))
    return summaries


def main():
    """Run a limited offline/disabled diagnostic matrix, never official reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--mode',choices=['disabled','offline'],default='offline')
    parser.add_argument('--gamma-mode',choices=['estimated','lagged_reference','diameter_bound'],default='estimated')
    parser.add_argument('--variance-profile',choices=['paper','safe'],default='paper')
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--methods',nargs='+',choices=['convex','constant','inverse_sqrt'],default=['convex','constant','inverse_sqrt'])
    args = parser.parse_args()
    cfg = load_config(ROOT/'configs/convex.yaml')
    cfg.update(steps=32,dimensions=[2],batch_size=32,eval_batch_size=1024,seeds=args.seeds,
               methods=args.methods,mode=args.mode,output_dir=args.output_dir)
    print(json.dumps(run_diagnostics(cfg,gamma_mode=args.gamma_mode,variance_profile=args.variance_profile),indent=2))


if __name__ == '__main__':
    main()
