"""Read-only figures for verified five-fold development summaries."""
from pathlib import Path
import numpy as np
from .analysis_views import MODELS, read_csv

TABLES=['metrics','summary','paired','paired_fold_means','curves','latent','angles','generalization','calibration']
COLORS=['#355C9A','#6B7685','#BA4A32','#8B5FA4','#27735C']


def load_tables(directory):
    directory=Path(directory)
    missing=[str(directory/f'{name}.csv') for name in TABLES if not (directory/f'{name}.csv').is_file()]
    if missing:
        raise FileNotFoundError('Missing saved CV summaries. Set QARL_CV_ANALYSIS_DIR. No training is performed.\n'+'\n'.join(missing))
    return {name:read_csv(directory/f'{name}.csv') for name in TABLES}


def plot_metrics(table):
    import matplotlib.pyplot as plt
    measures=['roc_auc','ap_er_positive','ap_er_negative','balanced_accuracy','log_loss']
    fig,axes=plt.subplots(1,5,figsize=(17,4.3),layout='constrained')
    for ax,measure in zip(axes,measures):
        for i,(name,color) in enumerate(zip(MODELS,COLORS)):
            part=table[table.experiment==name]
            for fold,g in part.groupby('fold'):
                ax.scatter(np.full(len(g),i+(fold-2)*.11),g[measure],s=15,color=color,alpha=.55)
            means=part.groupby('fold')[measure].mean()
            ax.scatter([i]*5,means,marker='_',s=100,color='black',zorder=3)
        ax.set_xticks(range(5),MODELS,rotation=55,ha='right')
        ax.set_title(measure);ax.set_ylabel('Loss (nats)' if measure=='log_loss' else 'Score')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Development CV · dots: fold × seed; black marks: seed mean within each fold · descriptive spread')
    return fig


def plot_pairs(pairs,comparison='B_minus_A'):
    import matplotlib.pyplot as plt
    measures=['roc_auc','ap_er_positive','ap_er_negative','balanced_accuracy','log_loss']
    fig,axes=plt.subplots(1,5,figsize=(17,4.2),layout='constrained')
    for ax,m in zip(axes,measures):
        data=pairs.pivot(index='fold',columns='seed',values=f'{comparison}_{m}')
        bound=max(np.abs(data.to_numpy()).max(),1e-8)
        im=ax.imshow(data,vmin=-bound,vmax=bound,cmap='RdBu_r',aspect='auto')
        for i in range(5):
            for j in range(3):
                v=data.iloc[i,j]
                ax.text(j,i,f'{v:+.3f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(v)>.65*bound else 'black')
        ax.set_xticks(range(3),data.columns,rotation=45);ax.set_yticks(range(5),data.index)
        ax.set_xlabel('Seed');ax.set_ylabel('Fold');ax.set_title(m)
        fig.colorbar(im,ax=ax,shrink=.6)
    fig.suptitle(f'{comparison.replace("_minus_"," − ")} · paired differences; lower log loss is better · separate color ranges')
    return fig


def plot_spectrum(latent):
    import matplotlib.pyplot as plt
    data=latent[(latent.checkpoint=='final')&(latent.split=='train')]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True,layout='constrained')
    for ax,name in zip(axes,['A','B','B_classical']):
        for row in data[data.experiment==name].itertuples():
            s=np.array([getattr(row,f'singular_{i}') for i in range(4)])
            ax.semilogy(range(1,5),np.maximum(s/s[0],1e-12),color=plt.cm.tab10(row.fold),
                        linestyle=['-','--',':'][row.seed-1729],alpha=.7)
        ax.axhline(1e-6,color='black',linewidth=.7)
        ax.set_xticks(range(1,5));ax.set_xlabel('Singular value index');ax.set_title(name);ax.grid(alpha=.2)
    axes[0].set_ylabel('Singular value / largest singular value')
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color=plt.cm.tab10(f),label=f'Fold {f}') for f in range(5)],loc='outside lower center',ncol=5)
    fig.suptitle('Centered final training spectrum · line styles: seeds 1729 / 1730 / 1731 · floor 1e−12')
    return fig


def plot_coordinates(latent):
    import matplotlib.pyplot as plt
    data=latent[(latent.checkpoint=='final')&(latent.split=='train')]
    fig,axes=plt.subplots(1,3,figsize=(12,6),layout='constrained')
    for ax,name in zip(axes,['A','B','B_classical']):
        part=data[data.experiment==name].sort_values(['fold','seed'])
        values=part[[f'coordinate_{i}_std' for i in range(4)]].to_numpy()
        im=ax.imshow(np.log10(np.maximum(values,1e-12)),vmin=-12,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(4));ax.set_xlabel('Coordinate (zero-based)')
        ax.set_yticks(range(15),[f'{r.fold}/{r.seed}' for r in part.itertuples()]);ax.set_title(name)
    axes[0].set_ylabel('Fold / seed');fig.colorbar(im,ax=axes,label='log10 coordinate SD (latent units)')
    fig.suptitle('Final training coordinate variability · SD uses ddof=0; display floor 1e−12')
    return fig


def plot_learning(curves,value='loss'):
    import matplotlib.pyplot as plt
    names=['A','B','B_classical']
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    for ax,name in zip(axes,names):
        data=curves[curves.experiment==name]
        for fold,part in data.groupby('fold'):
            avg=part.groupby('epoch').mean(numeric_only=True)
            color=plt.cm.tab10(fold)
            if value=='loss':
                ax.plot(avg.index,avg.validation_loss,color=color,label=f'Fold {fold}')
                ax.plot(avg.index,avg.train_loss,'--',color=color)
                ax.set_ylabel('BCE (nats)')
            elif value=='gradient':
                ax.semilogy(avg.index,np.maximum(avg.head_gradient_norm,1e-18),color=color,label=f'Fold {fold}')
                if name!='A':
                    ax.semilogy(avg.index,np.maximum(avg.encoder_gradient_norm,1e-18),'--',color=color)
                ax.set_ylabel('Mean minibatch L2 gradient norm')
            else:
                ax.plot(avg.index,100*avg.validation_angular_saturation_95,color=color,label=f'Fold {fold}')
                ax.plot(avg.index,100*avg.train_angular_saturation_95,'--',color=color)
                ax.set_ylabel('Saturated patient × coordinate values (%)')
        ax.set_title(name);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    detail={'loss':'solid: validation; dashed: online training',
            'gradient':'solid: head; dashed: encoder; independent y scales',
            'saturation':'solid: validation; dashed: training; B_classical angles are hypothetical'}[value]
    fig.suptitle(f'Seed means within fold · {detail}')
    return fig


def plot_generalization(gaps):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,metric in zip(axes,['log_loss','roc_auc']):
        for i,(name,color) in enumerate(zip(MODELS,COLORS)):
            part=gaps[gaps.experiment==name]
            ax.scatter(i+(part.fold-2)*.1,part[f'validation_minus_train_{metric}'],color=color,s=22,alpha=.7)
        ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(5),MODELS,rotation=40,ha='right')
        ax.set_ylabel(f'Validation − training {metric}');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same final checkpoint · all 15 fold/seed results per model · dependent observations')
    return fig


def plot_reliability(calibration):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    for name,color in zip(MODELS,COLORS):
        part=calibration[(calibration.experiment==name)&(calibration.n>0)].copy()
        part['psum']=part.n*part.mean_probability;part['ysum']=part.n*part.observed_positive_fraction
        grouped=part.groupby('bin')[['n','psum','ysum']].sum()
        ax.plot(grouped.psum/grouped.n,grouped.ysum/grouped.n,marker='o',color=color,label=name)
    ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),
        xlabel='Mean predicted ER+ probability',ylabel='Observed ER+ fraction',
        title='Descriptive reliability · 10 fixed bins · pooled OOF repetitions')
    ax.legend();ax.grid(alpha=.2)
    return fig
