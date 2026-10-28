"""Shared calculations and figures for saved development outputs.

No fitting, model construction, checkpoint loading, or data-cohort access.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def read_csv(path):
    # Preserve extreme classifier probabilities exactly across CSV round trips.
    return pd.read_csv(path, float_precision='round_trip')


MODELS = ['A', 'A_classical', 'B', 'B_frozen_probe', 'B_classical']
MEASURES = ['roc_auc','ap_er_positive','ap_er_negative','balanced_accuracy','accuracy','log_loss']


def paired_predictions(b, probe):
    joined = b.merge(probe, on='case_id', suffixes=('_B','_probe'), validate='one_to_one')
    if len(joined) != len(b) or len(joined) != len(probe):
        raise ValueError('Different validation patients')
    if not joined.er_label_B.equals(joined.er_label_probe):
        raise ValueError('Label mismatch')
    rho = float(spearmanr(joined.probability_er_positive_B, joined.probability_er_positive_probe).statistic)
    if not np.isfinite(rho):
        raise ValueError('Undefined Spearman correlation')
    return joined, rho




def summarize_metrics(table, measures):
    summary = table.groupby('experiment', sort=False)[list(measures)+['seconds']].agg(['mean','std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    return summary


def spectrum_values(z):
    singular = np.linalg.svd(z-z.mean(axis=0), compute_uv=False)
    return dict(rank_relative_1e6=int((singular > singular[0]*1e-6).sum()),
                **{f'singular_{i}': float(value) for i,value in enumerate(singular)})


def angular_values(z):
    # Offline NumPy calculation, shared by the official script and notebook.
    # Same formula/tolerance as diagnostics.angular_summary; no Torch execution
    # is needed to summarize saved arrays (avoids native backend/thread coupling).
    t = np.tanh(z / 2)
    derivative = .5 * (1 - t**2)
    return dict(z_min=float(z.min()),z_max=float(z.max()),
                z_mean=float(z.mean()),z_std=float(z.std()),
                angle_q05=float(np.quantile(np.pi*t,.05)),
                angle_median=float(np.median(np.pi*t)),
                angle_q95=float(np.quantile(np.pi*t,.95)),
                angle_abs_mean=float(np.abs(np.pi*t).mean()),
                angle_abs_max=float(np.abs(np.pi*t).max()),
                angular_saturation_95=float((np.abs(t)>.95).mean()),
                angular_saturation_99=float((np.abs(t)>.99).mean()),
                tanh_derivative_mean=float(derivative.mean()),
                tanh_derivative_below_001=float((derivative<.01).mean()))


def paired_values(by_name, seed, rho, n_validation, measures):
    return dict(seed=seed,n_validation=n_validation,spearman_B_probe=rho,
        **{f'B_minus_A_{m}':float(by_name.loc['B',m]-by_name.loc['A',m]) for m in measures},
        **{f'probe_minus_B_{m}':float(by_name.loc['B_frozen_probe',m]-by_name.loc['B',m]) for m in measures})


def generalization_table(curves):
    """A descriptive online gap; the two losses use different parameter states."""
    result = curves[['experiment','seed','epoch','train_loss','validation_loss']].copy()
    result['validation_minus_online_train_bce'] = result.validation_loss-result.train_loss
    return result


def plot_generalization(curves):
    import matplotlib.pyplot as plt
    data = generalization_table(curves)
    names = list(dict.fromkeys(data.experiment))
    figure, axes = plt.subplots(1,len(names),figsize=(4.2*len(names),3.7),
                               sharex=True,sharey=True,squeeze=False,layout='constrained')
    seeds = sorted(data.seed.unique())
    colors = dict(zip(seeds,['#355C9A','#BA4A32','#27735C']))
    styles = ['-','--',':']
    for ax,name in zip(axes.ravel(),names):
        for seed,style in zip(seeds,styles):
            part = data[(data.experiment==name)&(data.seed==seed)]
            ax.plot(part.epoch,part.validation_minus_online_train_bce,
                    color=colors[seed],linestyle=style,label=str(seed))
        ax.axhline(0,color='#777777',linewidth=.8)
        ax.set_title(name);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Validation BCE − online training BCE')
    axes[0,0].legend(title='Seed',fontsize=8)
    figure.suptitle('Fold 0 · descriptive gap; losses measured at different parameter states',fontsize=11)
    return figure


def plot_latent_spectrum(spectrum, split='train', checkpoint='final'):
    import matplotlib.pyplot as plt
    data = spectrum[(spectrum.split==split)&(spectrum.checkpoint==checkpoint)]
    names = list(dict.fromkeys(data.experiment))
    if data.empty:
        raise ValueError(f'No spectrum for split={split}, checkpoint={checkpoint}')
    figure,axes = plt.subplots(1,len(names),figsize=(4.2*len(names),3.7),sharex=True,sharey=True,
                              squeeze=False,layout='constrained')
    colors = dict(zip(sorted(data.seed.unique()),['#355C9A','#BA4A32','#27735C']))
    for ax,name in zip(axes.ravel(),names):
        for row in data[data.experiment==name].itertuples():
            singular = np.array([getattr(row,f'singular_{i}') for i in range(4)])
            relative = singular/singular[0] if singular[0] > 0 else np.zeros(4)
            ax.semilogy(range(1,5),np.maximum(relative,1e-12),marker='o',
                        color=colors[row.seed],label=f'{row.seed}: rank {row.rank_relative_1e6}')
        ax.axhline(1e-6,color='#777777',linestyle='--',linewidth=.8)
        ax.set_title(name);ax.set_xlabel('Singular value index');ax.set_xticks(range(1,5))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set_ylabel('Singular value / largest singular value')
    figure.suptitle(f'Centered latent representation · {split}, {checkpoint} · relative threshold 1e−6',fontsize=11)
    return figure


def plot_pilot_figures(curves):
    """Render convergence, gradient, and saturation panels from saved curve CSVs."""
    import matplotlib.pyplot as plt
    seeds = sorted({seed for _, seed in curves})
    colors = dict(A='#355C9A', B='#BA4A32', B_classical='#27735C')
    figures = {}
    def save(figure, name):
        figures[name] = figure
    fig, axes = plt.subplots(3,3,figsize=(12,9),sharex=True,sharey=True,layout='constrained')
    for col,name in enumerate(['A','B','B_classical']):
        for row,seed in enumerate(seeds):
            c = curves[name,seed]
            ax=axes[row,col]
            ax.plot(c.epoch,c.train_loss,'--',color=colors[name],label='Train (online)')
            ax.plot(c.epoch,c.validation_loss,color=colors[name],label='Validation')
            ax.set_title(f'{name} · seed {seed}')
            ax.set_xlabel('Epoch'); ax.set_ylabel('BCE')
            ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    maximum_loss = max(float(c[['train_loss','validation_loss']].to_numpy().max()) for c in curves.values())
    axes[0,0].set_ylim(0, maximum_loss * 1.05)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Fold 0 · fixed 50 epochs · final checkpoint is primary')
    save(fig,'convergence')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,(name,head) in zip(axes,[('A','head'),('B','head'),('B','encoder')]):
        for seed in seeds:
            c=curves[name,seed]
            ax.semilogy(c.epoch,c[f'{head}_gradient_norm'],label=str(seed))
        ax.set_title(f'{name}: {head} gradient norm')
        ax.set_xlabel('Epoch');ax.set_ylabel('Mean minibatch L2 norm');ax.grid(alpha=.2)
    axes[0].legend(title='Seed')
    save(fig,'gradient_evolution')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,name in zip(axes,['A','B','B_classical']):
        for seed in seeds:
            c=curves[name,seed]
            ax.plot(c.epoch,100*c.train_angular_saturation_95,label=f'{seed} train')
            ax.plot(c.epoch,100*c.validation_angular_saturation_95,'--',label=f'{seed} val')
        ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('|tanh(z/2)| > 0.95 (%)');ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,ncol=2)
    save(fig,'angular_saturation')
    parameter_cols=[f'gradient_l{l}_q{q}_{g}' for l in range(2) for q in range(4) for g in ['RY','RZ']]
    heat=np.array([pd.concat([curves[name,s] for s in seeds])[parameter_cols].mean().to_numpy()
                   for name in ['A','B']])
    fig,ax=plt.subplots(figsize=(12,2.8),layout='constrained')
    im=ax.imshow(np.log10(np.maximum(heat,1e-18)),vmin=-18,vmax=0,aspect='auto',cmap='viridis')
    ax.set_yticks([0,1],['A','B']);ax.set_xticks(range(16),[x.replace('gradient_','') for x in parameter_cols],rotation=45,ha='right')
    ax.set_title('Mean epoch RMS gradient by parameter · all three seeds')
    fig.colorbar(im,ax=ax,label='log10 RMS gradient')
    save(fig,'parameter_gradients')
    return figures


def plot_private_diagnostics(scores, paired, run_path, seeds):
    """Render exact histograms and paired scores from private development outputs."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    run_path = Path(run_path)
    colors = dict(A='#355C9A', B='#BA4A32', B_classical='#27735C')
    figures = {}
    def save(figure, name):
        figures[name] = figure
    fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True,sharey=True,layout='constrained')
    for row,name in enumerate(['A','B']):
        for col,seed in enumerate(seeds):
            folder=run_path/f'fold_0_{name}'/f'{name}_seed_{seed}'
            with np.load(folder/'latent_diagnostics.npz', allow_pickle=False) as z:
                ax=axes[row,col]
                bins=np.linspace(-np.pi,np.pi,33)
                for key,label,color,style in [('initial_train','Initial train','#888888','--'),
                        ('final_train','Final train',colors[name],'-'),
                        ('final_validation','Final validation','#27735C',':')]:
                    ax.hist((np.pi*np.tanh(z[key]/2)).ravel(),bins=bins,density=True,
                            histtype='step',label=label,color=color,linestyle=style,linewidth=1.5)
                ax.axvline(.95*np.pi,color='#999999',alpha=.5)
                ax.axvline(-.95*np.pi,color='#999999',alpha=.5)
                ax.set_title(f'{name} · seed {seed}');ax.set_xlabel('Angle (rad)');ax.set_ylabel('Density')
    axes[0,0].legend(fontsize=8)
    save(fig,'angle_distributions')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True,layout='constrained')
    for ax,seed in zip(axes,seeds):
        j=scores[seed]
        for label,color in [(0,'#355C9A'),(1,'#BA4A32')]:
            sub=j[j.er_label_B==label]
            ax.scatter(sub.probability_er_positive_B,sub.probability_er_positive_probe,
                       s=13,alpha=.65,color=color,label='ER+' if label else 'ER−')
        rho=paired.loc[paired.seed==seed,'spearman_B_probe'].iloc[0]
        ax.plot([0,1],[0,1],'--',color='#999999',linewidth=1)
        ax.set_title(f'Seed {seed} · Spearman ρ={rho:.3f}')
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('B: p(ER+)');ax.set_ylabel('Frozen probe: p(ER+)')
    axes[0].legend()
    save(fig,'B_probe_scores')
    return figures
