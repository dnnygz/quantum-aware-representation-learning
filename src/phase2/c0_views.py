"""Read-only aggregate loading and notebook plots for completed C0-v1 results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ARMS=['C0-NQE-Q','C0-NQE-E','C0-Hybrid-Q','C0-Hybrid-E']
COLORS={'Q':'#225B8E','E':'#B76B17'}


def load_results(folder):
    folder=Path(folder)
    names=['metrics','paired_Q_minus_E','paired_fold_means','primary_summary','score_summary',
           'geometry','frozen_observable_geometry','curves','costs','gradient_components',
           'generalization','same_encoder_controls','contextual_historical','historical_B_geometry','historical_metrics']
    missing=[name+'.csv' for name in names if not (folder/(name+'.csv')).is_file()]
    if missing:raise FileNotFoundError('Missing completed C0 aggregate artifacts: '+', '.join(missing)+'. No training will be performed.')
    result={name:pd.read_csv(folder/(name+'.csv')) for name in names}
    result['integrity']=json.loads((folder/'integrity.json').read_text())
    return result


def style():
    plt.rcParams.update({'figure.figsize':(10,4.8),'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'normal','figure.dpi':110})


def _label(arm):return arm.removeprefix('C0-')


def primary_contrasts(data):
    frame=data['paired_fold_means'].query("split=='validation'")
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,metric,label in zip(axes,['roc_auc_Q_minus_E','log_loss_Q_minus_E'],
                               ['ROC-AUC difference (Q − E)','Log loss difference (Q − E; nats)']):
        for i,schedule in enumerate(['NQE','Hybrid']):
            subset=frame[frame.schedule==schedule]
            ax.scatter(subset[metric],i+np.linspace(-.16,.16,len(subset)),color=COLORS['Q'],s=38,
                       label='Five fold means' if i==0 else None)
            ax.scatter(subset[metric].mean(),i,marker='D',color='#222222',s=55,
                       label='Mean across folds' if i==0 else None)
            for j,row in enumerate(subset.itertuples()):
                ax.annotate(str(row.fold),(getattr(row,metric),i+np.linspace(-.16,.16,len(subset))[j]),
                            xytext=(4,3),textcoords='offset points',fontsize=8)
        ax.axvline(0,color='#777777',linewidth=1);ax.set_yticks([0,1],['NQE','Hybrid'])
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    axes[0].set_title('Discrimination; positive favors Q')
    axes[1].set_title('Probability loss; negative favors Q')
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Primary development contrasts: seeds averaged within each fold')
    fig.tight_layout();return fig


def predictive_scores(data):
    frame=data['metrics'].query("snapshot=='final' and readout=='VQC' and split=='validation'")
    means=frame.groupby(['arm','fold'])[['roc_auc','log_loss']].mean().groupby('arm').agg(['mean','std'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,m,label,b in zip(axes,['roc_auc','log_loss'],['Validation ROC-AUC','Validation log loss (nats)'],[data['historical_metrics'].query("experiment=='B'")[m].mean() for m in ['roc_auc','log_loss']]):
        for i,arm in enumerate(ARMS):
            row=means.loc[arm,m];ax.errorbar(row['mean'],i,xerr=row['std'],fmt='o',color=COLORS[arm[-1]],capsize=3)
        ax.axvline(b,color='#555555',linestyle='--',label='Historical B (context only)')
        ax.set_yticks(range(4),[_label(a) for a in ARMS]);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Final VQC scores; bars show descriptive SD of five fold means, not confidence intervals')
    fig.tight_layout();return fig


def learning_curves(data):
    frame=data['curves'];fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for arm in ARMS:
        subset=frame[frame.arm==arm]
        primary=subset[subset.stage==('head' if 'NQE' in arm else 'joint')]
        auxiliary=subset[subset.stage==('geometry' if 'NQE' in arm else 'joint')]
        line='-' if 'NQE' in arm else '--'
        for ax,source,column in [(axes[0],auxiliary,'online_auxiliary'),(axes[1],primary,'online_bce'),(axes[2],primary,'validation_bce')]:
            means=source.groupby(['fold','epoch'])[column].mean().groupby('epoch').mean()
            ax.plot(means.index,means.values,color=COLORS[arm[-1]],linestyle=line,label=_label(arm))
            ax.set_xlabel('Within-stage epoch');ax.grid(alpha=.2)
    axes[0].set_title('Online pair objective');axes[0].set_ylabel('Mean squared residual')
    axes[1].set_title('Online training BCE');axes[1].set_ylabel('BCE (nats)')
    axes[2].set_title('End-of-epoch validation BCE');axes[2].set_ylabel('BCE (nats)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed 50-epoch stages; NQE has an additional 50-epoch geometry stage')
    fig.tight_layout();return fig


def geometry_transfer(data):
    g=data['geometry'];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,column,label in zip(axes,['Q_balanced_pair_risk','trace_distance'],
                              ['Balanced Q pair risk (lower is better)','Balanced class-state trace distance']):
        for i,arm in enumerate(ARMS):
            subset=g[g.arm==arm].groupby('split')[column].mean()
            ax.plot([0,1],[subset['train'],subset['validation']],color=COLORS[arm[-1]],
                    linestyle='-' if 'NQE' in arm else '--',marker='o',label=_label(arm))
        ax.set_xticks([0,1],['Training','Validation']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);axes[0].set_title('State-pair separation transfers incompletely')
    axes[1].set_title('Trace distance is a distinct ensemble descriptor')
    fig.tight_layout();return fig


def readout_controls(data):
    frame=data['metrics'].query("snapshot=='final' and split=='validation'")
    readouts=['frozen_initial_VQC','VQC','LR_z','LR_angle','class_reference_Q','class_reference_E']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,metric,cmap in zip(axes,['roc_auc','log_loss'],['Blues','YlOrBr']):
        values=frame.groupby(['arm','readout'])[metric].mean().unstack().loc[ARMS,readouts]
        im=ax.imshow(values,cmap=cmap,aspect='auto',vmin=0,vmax=1 if metric=='roc_auc' else None)
        ax.set_xticks(range(6),['Frozen v₀','VQC','LR-z','LR-angle','Q overlap','E overlap'],rotation=35,ha='right')
        ax.set_yticks(range(4),[_label(a) for a in ARMS])
        for i in range(4):
            for j in range(6):ax.text(j,i,f'{values.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9,
                                      color='white' if im.norm(values.iloc[i,j])>.65 else '#222222')
        ax.set_title('Validation ROC-AUC' if metric=='roc_auc' else 'Validation log loss (nats)')
        fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    fig.suptitle('Same final encoder, different fixed or training-only fitted readouts')
    fig.tight_layout();return fig


def generalization(data):
    frame=data['generalization'];fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for i,arm in enumerate(ARMS):
        subset=frame[frame.arm==arm]
        for ax,column,label in zip(axes,['log_loss','roc_auc'],['Log loss (nats)','ROC-AUC']):
            values=[subset['train_'+column].mean(),subset['validation_'+column].mean()]
            ax.plot([0,1],values,color=COLORS[arm[-1]],linestyle='-' if 'NQE' in arm else '--',marker='o',label=_label(arm))
            ax.set_xticks([0,1],['Training','Validation']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Same final checkpoint: training specialization and held-out behavior')
    fig.tight_layout();return fig


def accessibility(data):
    g=data['geometry'].query("snapshot=='final' and split=='validation'")
    m=data['metrics'].query("snapshot=='final' and split=='validation' and readout=='VQC'")
    values=g.merge(m,on=['arm','fold','seed','snapshot','split'],validate='one_to_one')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for arm in ARMS:
        subset=values[values.arm==arm]
        for ax,x,title in zip(axes,['trace_distance','accessibility_ratio'],
                              ['Balanced class-state trace distance','Absolute raw contrast / trace distance']):
            ax.scatter(subset[x],subset.roc_auc,color=COLORS[arm[-1]],marker='o' if 'NQE' in arm else '^',
                       facecolors='none' if 'Hybrid' in arm else COLORS[arm[-1]],label=_label(arm),alpha=.8)
            ax.set_xlabel(title);ax.set_ylabel('Validation ROC-AUC');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Distinct descriptors; ratio near one does not establish complete predictive accessibility')
    fig.tight_layout();return fig


def gradient_components(data):
    frame=data['gradient_components'];fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for arm in ['C0-Hybrid-Q','C0-Hybrid-E']:
        subset=frame[frame.arm==arm].groupby(['fold','epoch'])[['ratio','cosine']].mean().groupby('epoch').mean()
        for ax,key,label in zip(axes,['ratio','cosine'],['Auxiliary / BCE encoder-gradient norm','Encoder-gradient cosine']):
            ax.plot(subset.index,subset[key],marker='o',color=COLORS[arm[-1]],label=_label(arm));ax.set_xlabel('Joint epoch');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[1].axhline(0,color='#777777',linewidth=1);axes[0].legend(fontsize=8)
    fig.suptitle('Fixed training-only diagnostic batches; descriptive means, not causal proof')
    fig.tight_layout();return fig


def costs(data):
    frame=data['costs'].groupby('arm').mean(numeric_only=True).reindex(ARMS)
    fig,ax=plt.subplots(figsize=(9,4.5));left=np.zeros(4)
    for name,color in [('geometry_seconds','#7B9CB6'),('head_seconds','#555555'),('joint_seconds','#B76B17'),('diagnostic_seconds','#9E87AC')]:
        values=frame[name].fillna(0).to_numpy();ax.barh(range(4),values,left=left,label=name.removesuffix('_seconds').replace('_',' '),color=color);left+=values
    ax.set_yticks(range(4),[_label(a) for a in ARMS]);ax.set_xlabel('Mean recorded seconds per pipeline (partial timer decomposition)')
    ax.set_xlim(left=0);ax.legend(ncols=4,fontsize=8);ax.set_title('Stage timers exclude some input, scoring, frozen-readout and serialization overhead')
    fig.tight_layout();return fig


PLOTS={'primary_contrasts':primary_contrasts,'predictive_scores':predictive_scores,
       'learning_curves':learning_curves,'geometry_transfer':geometry_transfer,
       'readout_controls':readout_controls,'generalization':generalization,
       'accessibility':accessibility,'gradient_components':gradient_components,'costs':costs}
