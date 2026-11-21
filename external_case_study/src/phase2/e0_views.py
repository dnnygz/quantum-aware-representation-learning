"""Read-only aggregate E0 figures; no model construction, cohort access, or fitting."""
from pathlib import Path
import numpy as np
from .analysis_views import read_csv

NAMES=['metrics','head_results','e01_paired','e01_diagonal','e02_paired','summary',
       'primary_summary','fold_primary','initialization_spread','curves','costs']
MODELS=['B_historical','VQC_refit','LR_z_historical','LR_angle']
LABELS=['Historical B','VQC refit','Historical LR-z','LR-angle']
COLORS=['#355C9A','#BA4A32','#8B5FA4','#27735C']


def load_tables(path):
    path=Path(path)
    missing=[str(path/f'{name}.csv') for name in NAMES if not (path/f'{name}.csv').is_file()]
    if missing:
        raise FileNotFoundError('Missing saved E0 summaries. Set QARL_E0_ANALYSIS_DIR. No training will run.\n'+'\n'.join(missing))
    return {name:read_csv(path/f'{name}.csv') for name in NAMES}


def plot_e01(paired):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for fold,group in paired.groupby('fold'):
        for diagonal,marker in [(False,'o'),(True,'s')]:
            part=group[group.diagonal==diagonal]
            ax.scatter(part.train_log_loss_minus_B,part.validation_log_loss_minus_B,
                color=plt.cm.tab10(fold),marker=marker,s=40 if diagonal else 25,alpha=.8,
                label=f'Fold {fold}' if not diagonal else None)
    ax.axhline(0,color='black',linewidth=.8);ax.axvline(0,color='black',linewidth=.8)
    ax.set(xlabel='Refit − historical training log loss (nats)',
           ylabel='Refit − historical validation log loss (nats)',
           title='E0.1 primary contrasts · 45 dependent head fits · squares: h=s')
    ax.legend();ax.grid(alpha=.2)
    return fig


def plot_pair_matrix(table,columns,titles,heading):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(columns),figsize=(5.3*len(columns),4.8),layout='constrained',squeeze=False)
    for ax,column,title in zip(axes.ravel(),columns,titles):
        data=table.pivot(index='fold',columns='encoder_seed',values=column)
        bound=max(float(np.abs(data.to_numpy()).max()),1e-8)
        im=ax.imshow(data,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        for i in range(5):
            for j in range(3):
                value=data.iloc[i,j]
                ax.text(j,i,f'{value:+.4f}',ha='center',va='center',fontsize=9,
                    color='white' if abs(value)>.65*bound else 'black')
        ax.set_xticks(range(3),data.columns);ax.set_yticks(range(5),data.index)
        ax.set(xlabel='Encoder seed',ylabel='Fold',title=title)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle(heading+' · independent color ranges')
    return fig


def plot_diagonal(table):
    return plot_pair_matrix(table,['train_log_loss_minus_B','validation_log_loss_minus_B'],
        ['Δ training log loss (nats)','Δ validation log loss (nats)'],
        'Prespecified diagonal h=s · refit − corresponding historical B · not a pure initialization contrast')


def plot_e02(table):
    return plot_pair_matrix(table,['validation_roc_auc_minus_LR_z','validation_log_loss_minus_LR_z'],
        ['Δ validation ROC-AUC','Δ validation log loss (nats)'],
        'E0.2 primary paired contrasts · LR-angle − historical LR-z')


def plot_initializations(paired):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    keys=sorted(set(zip(paired.fold,paired.encoder_seed)));positions={key:i for i,key in enumerate(keys)}
    colors=['#355C9A','#BA4A32','#27735C']
    for ax,split in zip(axes,['train','validation']):
        for h,color,offset in zip([1729,1730,1731],colors,[-.18,0,.18]):
            part=paired[paired.head_seed==h]
            ax.scatter([positions[f,s]+offset for f,s in zip(part.fold,part.encoder_seed)],part[f'{split}_log_loss'],
                color=color,s=30,label=f'Head {h}',zorder=3)
        for key,g in paired.groupby(['fold','encoder_seed']):
            i=positions[key];historical=float(g[f'{split}_log_loss_reference_B'].iloc[0])
            ax.plot([i-.3,i+.3],[historical,historical],color='black',linewidth=1.3)
        ax.set_ylabel(f'{split.capitalize()} log loss (nats)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xticks(range(15),[f'{f}/{s}' for f,s in keys],rotation=50,ha='right')
    axes[-1].set_xlabel('Fold / encoder seed')
    fig.suptitle('All prespecified starts · black segments: historical B · no best-head selection')
    return fig


def plot_metrics(table):
    import matplotlib.pyplot as plt
    data=table[table.split=='validation'].groupby(['fold','encoder_seed','model']).mean(numeric_only=True).reset_index()
    measures=['roc_auc','ap_er_negative','balanced_accuracy','log_loss','brier']
    fig,axes=plt.subplots(1,5,figsize=(17,4.6),layout='constrained')
    for ax,metric in zip(axes,measures):
        for i,(model,color) in enumerate(zip(MODELS,COLORS)):
            part=data[data.model==model]
            ax.scatter(i+(part.fold-2)*.07,part[metric],color=color,s=22,alpha=.65)
            means=part.groupby('fold')[metric].mean()
            ax.scatter(np.full(5,i),means,marker='_',color='black',s=90)
        ax.set_xticks(range(4),LABELS,rotation=50,ha='right');ax.set_title(metric);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('Loss (nats)' if metric=='log_loss' else 'Score')
    fig.suptitle('Validation · VQC heads averaged within encoder first · dots: encoder results; black: fold means')
    return fig


def plot_curves(curves,fold=0):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors=['#355C9A','#BA4A32','#27735C']
    for ax,s in zip(axes,[1729,1730,1731]):
        for h,color in zip([1729,1730,1731],colors):
            part=curves[(curves.fold==fold)&(curves.encoder_seed==s)&(curves.head_seed==h)]
            ax.plot(part.epoch,part.validation_loss,color=color,label=f'h={h}')
            ax.plot(part.epoch,part.train_loss,'--',color=color)
        ax.set(title=f'Encoder seed {s}',xlabel='Epoch',ylabel='BCE (nats)');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Fold {fold} · solid: validation; dashed: online training · independent y scales · fixed epoch 50')
    return fig


def plot_generalization(wide):
    import matplotlib.pyplot as plt
    data=wide.groupby(['fold','encoder_seed','model']).mean(numeric_only=True).reset_index()
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    for model,label,color,marker in zip(MODELS,LABELS,COLORS,['o','s','^','D']):
        part=data[data.model==model]
        ax.scatter(part.train_log_loss,part.validation_log_loss,label=label,color=color,marker=marker,s=32,alpha=.75)
    bound=float(data[['train_log_loss','validation_log_loss']].to_numpy().max())*1.05
    ax.plot([0,bound],[0,bound],'--',color='gray');ax.set(xlabel='Final training log loss (nats)',
        ylabel='Final validation log loss (nats)',title='Same-checkpoint generalization · head means within each encoder')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    return fig


def plot_cost(costs):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.4),layout='constrained')
    for i,(model,label,color) in enumerate(zip(['VQC_refit','LR_angle'],['VQC · 50 epochs','LR-angle · L-BFGS'],['#BA4A32','#27735C'])):
        part=costs[costs.model==model]
        values=part.fit_update_seconds+part.epoch_validation_seconds
        ax.scatter(i+(part.fold-2)*.025,values,color=color,s=16,alpha=.6)
        ax.plot([i-.18,i+.18],[values.mean()]*2,color='black')
    ax.set_xticks([0,1],['VQC · 50 epochs','LR-angle · L-BFGS']);ax.set_yscale('log')
    ax.set(ylabel='Fit + epoch validation time (seconds; log scale)',
        title='Conditional head cost · optimizer budgets differ; original encoder cost excluded')
    ax.grid(axis='y',alpha=.2)
    return fig
