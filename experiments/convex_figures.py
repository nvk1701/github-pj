"""Ve hinh va mapping Sheet 4 tu cac mang CP03 da du coverage."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_run(directory, history, metrics, oracle):
    """Luu population B va quy dao co nhan dung learner/reference/oracle."""
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(metrics['step'],metrics['population_cum_B'])
    axes[0].set(xlabel='Pre-update step',ylabel='Cumulative population B',title='Numerical approximate-oracle comparator')
    for path,label in ((history.theta_path,'learner'),(history.theta_star_path,'generating reference'),(oracle,'approximate population oracle')):
        axes[1].plot(path[:,0],path[:,1],label=label)
    axes[1].legend(fontsize=8)
    axes[1].set(title='First two coordinates',xlabel='Coordinate 1',ylabel='Coordinate 2')
    fig.tight_layout()
    fig.savefig(Path(directory)/'figure.png',dpi=150)
    plt.close(fig)


def export_figures(root, records, config):
    """Gop mean/sample SD theo seed, xuat CSV va mapping dung pham vi.

    Hinh Figure-5-style thuoc giao thuc du an, khong phai ket qua goc tac gia.
    Quy dao dung seed nho nhat da khai bao, ghi rieng hai comparator.
    Paper scalar va relative deviation giu N/A khi chua the doi chieu.
    """
    root=Path(root)
    curves,mapping=[],[]
    fig_a,ax_a=plt.subplots(figsize=(7,4))
    fig_c,ax_c=plt.subplots(figsize=(7,4))
    target=128 if 128 in config['dimensions'] else max(config['dimensions'])
    for d in config['dimensions']:
        for method in config['methods']:
            selected=[r for r in records if r['dimension']==d and r['method']==method]
            if len(selected)!=len(config['seeds']) or any(r['local_status']!='complete' for r in selected):
                raise ValueError('plots require all selected seeds to be complete')
            paths=[]
            for row in selected:
                with np.load(Path(row['path'])/'history.npz') as data:
                    paths.append(data['population_cum_B'])
            values=np.array(paths)
            mean=np.mean(values,axis=0)
            sd=np.std(values,axis=0,ddof=1) if len(values)>1 else None
            steps=np.arange(1,len(mean)+1)
            for t,value in enumerate(mean):
                curves.append(dict(dimension=d,method=method,step=t+1,mean=float(value),
                                   sample_std=float(sd[t]) if sd is not None else None))
            if method=='convex':
                ax_a.plot(steps,mean,label=f'd={d}')
                if sd is not None:
                    ax_a.fill_between(steps,mean-sd,mean+sd,alpha=.12)
            if d==target:
                ax_c.plot(steps,mean,label=method)
                if sd is not None:
                    ax_c.fill_between(steps,mean-sd,mean+sd,alpha=.12)
            mapping.append(dict(experiment_id=f"{config['protocol_version']}-d{d}-{method}",
                figure_or_table='Figure 5 bottom-left style / T15 baseline comparison',
                metric_name='final cumulative population B',comparator='numerical approximate population oracle',
                dimension=d,steps=config['steps'],n_seeds=len(selected),mean=float(mean[-1]),
                sample_std=float(sd[-1]) if sd is not None else 'N/A',paper_value='N/A',
                paper_value_source='No validated comparable scalar extraction',relative_deviation_percent='N/A',
                wandb_url=json.dumps([r['wandb_url'] for r in selected]),local_artifact=str(root/'population_curves.csv'),
                slide_number=26,scientific_status=config['run_category'],
                notes='Mean +/- sample SD, ddof=1. Untuned baselines. No normalization by T.'))
    ax_a.set(xlabel='Step',ylabel='Cumulative population B',title=f"Figure 5-style Convex dimensions ({config['run_category']})")
    ax_c.set(xlabel='Step',ylabel='Cumulative population B',title=f'T15 project baselines, d={target}, untuned LR0=0.1')
    for fig,ax,name in ((fig_a,ax_a,'figure5_left.png'),(fig_c,ax_c,'baseline_comparison.png')):
        handles,_=ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        fig.tight_layout()
        fig.savefig(root/name,dpi=160)
        plt.close(fig)
    if 128 in config['dimensions'] and 'convex' in config['methods']:
        row=next(r for r in records if r['dimension']==128 and r['method']=='convex' and r['seed']==min(config['seeds']))
        with np.load(Path(row['path'])/'history.npz') as data:
            fig,ax=plt.subplots(figsize=(6,5))
            paths=[data[k][:,:2] for k in ('theta','reference','approx_oracle')]
            for path,label in zip(paths,('learner, pre-update','generating reference','approximate population oracle')):
                ax.plot(path[:,0],path[:,1],label=label)
            ax.legend(fontsize=8)
            ax.set(title=f"Figure 5-style d128, seed{row['seed']} ({config['run_category']})",xlabel='Coordinate 1',ylabel='Coordinate 2')
            fig.tight_layout()
            fig.savefig(root/'figure5_right.png',dpi=160)
            plt.close(fig)
            with (root/'trajectory_d128.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream)
                writer.writerow(['step','learner_1','learner_2','reference_1','reference_2','oracle_1','oracle_2'])
                writer.writerows((t+1,*paths[0][t],*paths[1][t],*paths[2][t]) for t in range(len(paths[0])))
        mapping.append(dict(mapping[0],experiment_id='convex-d128-trajectory',figure_or_table='Figure 5 bottom-right style',
            metric_name='two-coordinate trajectory',dimension=128,n_seeds=1,mean='N/A',sample_std='N/A',
            paper_value_source='Trajectory has no directly comparable scalar paper value',wandb_url=row['wandb_url'],
            local_artifact=str(root/'trajectory_d128.csv'),notes='Selected seed, not seed-averaged path.'))
    mapping.append(dict(mapping[0],experiment_id='T17-outside-this-export',figure_or_table='T17 sensitivity',metric_name='cumulative population B',
        mean='N/A',sample_std='N/A',paper_value_source='Separate CP05 exploratory one-factor results; not aggregated here',
        dimension=128,steps=2000,n_seeds=3,wandb_url='N/A',local_artifact='configs/convex_cp03_sensitivity.yaml',slide_number=28,
        scientific_status='not-covered-by-this-export',notes='See docs/results/cp04-cp05/RESULTS.md for CP05 sensitivity evidence. Never retune main protocol.'))
    for filename,rows in (('population_curves.csv',curves),('sheet4_mapping.csv',mapping)):
        with (root/filename).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
