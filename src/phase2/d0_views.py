"""Aggregate-only D0 tables and figures; no checkpoints, cohort access, or fitting."""
from pathlib import Path
import numpy as np
from .analysis_views import read_csv

NAMES=['geometry','coordinates','functional','directions','checks','baseline_metrics',
       'shared_encoder','shared_scaled','intervention_summary','effect_distribution',
       'paired_interventions','coordinate_ablation']
MODELS=['A','B','B_frozen_probe','B_classical']
COLORS={'A':'#355C9A','B':'#BA4A32','B_frozen_probe':'#8B5FA4','B_classical':'#27735C'}
FAMILIES=['mean','zero','scramble','local_coordinate','local_pc']


def load_tables(path):
    path=Path(path)
    missing=[str(path/f'{name}.csv') for name in NAMES if not (path/f'{name}.csv').is_file()]
    if missing:
        raise FileNotFoundError('Missing saved D0 summaries. Set QARL_D0_ANALYSIS_DIR. No diagnostics or training will run.\n'+'\n'.join(missing))
    return {name:read_csv(path/f'{name}.csv') for name in NAMES}


def plot_geometry(table):
    import matplotlib.pyplot as plt
    data=table.query("checkpoint=='final' and split=='validation'")
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    for ax,name in zip(axes,['A','B','B_classical']):
        part=data[data.model==name]
        for (fold,seed),g in part.groupby(['fold','seed']):
            g=g.set_index('space')
            ax.plot([0,1],[g.loc['z','variance_entropy_rank'],g.loc['angle','variance_entropy_rank']],
                    color=COLORS[name],alpha=.45,marker='o',markersize=3)
        ax.set_xticks([0,1],['Pre-angular z','Angular transform'])
        ax.set_title(name+(' · hypothetical map' if name=='B_classical' else ''))
        ax.set_ylim(.8,4.2);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Variance entropy rank')
    fig.suptitle('Variation across patients · final validation embeddings · changes do not establish causal information loss')
    return fig


def plot_decomposition(coordinates,fold=0,seed=1729):
    import matplotlib.pyplot as plt
    data=coordinates.query("model=='B' and checkpoint=='final'")
    data=data[(data.fold==fold)&(data.seed==seed)]
    measures=[('z_sd','SD of z'),('map_derivative_relative_max_mean','Mean map derivative / (π/2)'),
              ('angle_expectation_gradient_rms','RMS ∂f/∂angle'),('z_expectation_gradient_rms','RMS ∂f/∂z')]
    fig,axes=plt.subplots(1,4,figsize=(14,4),layout='constrained')
    for ax,(key,label) in zip(axes,measures):
        for split,style in [('train','o-'),('validation','s--')]:
            part=data[data.split==split].sort_values('coordinate')
            ax.plot(part.coordinate,part[key],style,label=split)
        ax.set_xlabel('Local coordinate index');ax.set_xticks(range(4));ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(f'B · fold {fold}, seed {seed} · variation and sensitivities have separate meanings and units')
    return fig


def plot_shared_sensitivity(functional):
    import matplotlib.pyplot as plt
    data=functional.query("checkpoint=='final' and split=='validation' and derivative_kind=='probability_z'")
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,key,title in zip(axes,['gradient_norm_rms','training_sd_scaled_gradient_norm_rms'],
            ['Unscaled probability response','Response scaled by training coordinate SD']):
        part=data[data.model.isin(['B','B_frozen_probe'])]
        for (fold,seed),g in part.groupby(['fold','seed']):
            by=g.set_index('model')
            ax.plot([0,1],[by.loc['B',key],by.loc['B_frozen_probe',key]],'o-',color='#777777',alpha=.5,markersize=4)
        ax.set_xticks([0,1],['B','B_frozen_probe']);ax.set_ylabel('RMS gradient norm');ax.set_title(title);ax.grid(alpha=.2)
    fig.suptitle('Matched final encoder · each line is one fold/seed · sensitivity does not establish useful information')
    return fig


def plot_interventions(table):
    import matplotlib.pyplot as plt
    data=table[(table.split=='validation')&((table.alpha==0)|(table.alpha==.1))]
    fig,axes=plt.subplots(2,5,figsize=(16,6),layout='constrained')
    for col,family in enumerate(FAMILIES):
        for row,metric in enumerate(['delta_roc_auc_mean','delta_log_loss_mean']):
            ax=axes[row,col]
            for i,name in enumerate(MODELS):
                part=data[(data.family==family)&(data.model==name)]
                ax.scatter(i+(part.fold-2)*.06,part[metric],color=COLORS[name],s=18,alpha=.7)
            ax.axhline(0,color='black',linewidth=.7)
            ax.set_xticks(range(4),MODELS,rotation=55,ha='right');ax.grid(axis='y',alpha=.2)
            ax.set_title(family.replace('_',' ')+(' · α=0.1' if family.startswith('local') else ''))
            ax.set_ylabel('Δ ROC-AUC' if row==0 else 'Δ log loss (nats)')
    fig.suptitle('Frozen interventions · mean of fixed design cells per fold/seed · independent panel scales; not retraining effects')
    return fig


def plot_constant_ablation(ablation):
    import matplotlib.pyplot as plt
    data=ablation[(ablation.split=='validation')&ablation.model.isin(['B','B_frozen_probe'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True,layout='constrained')
    for ax,name in zip(axes,['B','B_frozen_probe']):
        for family,marker,color in [('mean','o','#355C9A'),('zero','x','#BA4A32')]:
            part=data[(data.model==name)&(data.family==family)]
            ax.scatter(np.maximum(part.relative_training_sd,1e-12),np.maximum(part.probability_change_abs_mean,1e-16),
                       marker=marker,color=color,label=family,s=23,alpha=.7)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Coordinate SD / largest coordinate SD in its training run')
        ax.set_title(name);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean absolute probability change')
    fig.suptitle('Same B embedding · mean versus zero replacement · display floors x=1e−12, y=1e−16')
    return fig


def plot_support(table):
    import matplotlib.pyplot as plt
    data=table[(table.split=='validation')&((table.alpha==0)|(table.alpha==.1))]
    fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
    for ax,key,label in zip(axes,['z_l2_mean_mean','angle_l2_mean_mean','outside_training_bounds_after_mean'],
            ['Mean L2 displacement in z','Mean L2 angular displacement (radians)', 'Fraction outside training coordinate bounds']):
        for name,offset in [('B',-.10),('B_classical',.10)]:
            for i,family in enumerate(FAMILIES):
                part=data[(data.family==family)&(data.model==name)]
                ax.scatter(np.full(len(part),i+offset),part[key],color=COLORS[name],s=18,alpha=.6,label=name if i==0 else None)
        ax.set_xticks(range(5),[x.replace('_','\n') for x in FAMILIES]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Final validation · local α=0.1 · B_classical uses its own representation; marginal bounds do not establish support')
    return fig


def plot_local_approximation(table):
    import matplotlib.pyplot as plt
    data=table[(table.split=='validation')&table.family.str.startswith('local')]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,family in zip(axes,['local_coordinate','local_pc']):
        for name in ['B','B_frozen_probe']:
            part=data[(data.family==family)&(data.model==name)]
            means=part.groupby('alpha').linearization_error_abs_mean_mean.mean()
            ax.plot(means.index,means,marker='o',color=COLORS[name],label=name)
            ax.scatter(part.alpha,part.linearization_error_abs_mean_mean,color=COLORS[name],s=9,alpha=.3)
        ax.set(xlabel='α × training directional SD',ylabel='Mean absolute first-order probability error',title=family.replace('_',' '))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('All fold/seed effects retained · lines are descriptive means, without independent-sample intervals')
    return fig
