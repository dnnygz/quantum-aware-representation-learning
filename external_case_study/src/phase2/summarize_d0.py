"""Export non-patient-level coordinate ablations from the completed D0 archive."""
import argparse
from pathlib import Path
import numpy as np
from .analysis_views import read_csv


def export_ablation(run, summaries):
    run,summaries=Path(run),Path(summaries)
    destination=summaries/'coordinate_ablation.csv'
    if destination.exists():
        raise FileExistsError(destination)
    detail=read_csv(run/'interventions.csv')
    coordinates=read_csv(summaries/'coordinates.csv').query("checkpoint=='final'")
    reference=coordinates[['fold','seed','model','split','coordinate','training_sd']].copy()
    reference['largest_training_sd']=reference.groupby(['fold','seed','model','split']).training_sd.transform('max')
    reference['relative_training_sd']=reference.training_sd/reference.largest_training_sd
    ablation=detail[detail.family.isin(['mean','zero'])].copy()
    ablation['coordinate']=ablation.direction.str.rsplit('_',n=1).str[-1].astype(int)
    ablation=ablation.merge(reference,on=['fold','seed','model','split','coordinate'],validate='many_to_one')
    # No IDs, per-patient arrays, checkpoints, fitting, or raw cohort access.
    ablation.to_csv(destination,index=False)
    return ablation


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--summaries',required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--shared-scaled-only',action='store_true')
    mode.add_argument('--distributions-only',action='store_true')
    args=parser.parse_args()
    table=(export_shared_scaled(args.run,args.summaries) if args.shared_scaled_only
           else export_distributions(args.summaries) if args.distributions_only
           else export_ablation(args.run,args.summaries))
    print(f'Exported {len(table)} aggregate diagnostic rows')


def export_shared_scaled(run, summaries):
    """Compare both heads in shared coordinates, before/after training-SD scaling."""
    import pandas as pd
    run,summaries=Path(run),Path(summaries)
    destination=summaries/'shared_scaled.csv'
    if destination.exists():
        raise FileExistsError(destination)
    rows=[]
    for folder in sorted(run.glob('fold_*_seed_*')):
        _,fold,_,seed=folder.name.split('_')
        with np.load(folder/'B_final_training_reference.npz',allow_pickle=False) as archive:
            sd=archive['sd'].copy()
        for split in ['train','validation']:
            with np.load(folder/f'B_final_{split}_derivatives.npz',allow_pickle=False) as archive:
                b=archive['probability_gradient'].copy()
            with np.load(folder/f'B_frozen_probe_final_{split}_derivatives.npz',allow_pickle=False) as archive:
                probe=archive['probability_gradient'].copy()
            for scale,weights in [('raw',np.ones(4)),('training_sd',sd)]:
                left,right=b*weights,probe*weights
                norm=np.linalg.norm(left,axis=1)*np.linalg.norm(right,axis=1)
                defined=norm>1e-15
                cosine=np.sum(left[defined]*right[defined],axis=1)/norm[defined]
                rows.append(dict(fold=int(fold),seed=int(seed),split=split,scale=scale,
                    cosine_mean=float(cosine.mean()),cosine_q05=float(np.quantile(cosine,.05)),
                    cosine_median=float(np.median(cosine)),cosine_q95=float(np.quantile(cosine,.95)),
                    defined_fraction=float(defined.mean())))
    table=pd.DataFrame(rows)
    table.to_csv(destination,index=False)
    return table


def export_distributions(summaries):
    """Separate fold and seed variation after averaging the fixed intervention cells."""
    import pandas as pd
    summaries=Path(summaries);destination=summaries/'effect_distribution.csv'
    if destination.exists():
        raise FileExistsError(destination)
    table=read_csv(summaries/'intervention_summary.csv')
    measures=[c for c in table if c.endswith('_mean')]
    rows=[]
    for (model,split,family,alpha),group in table.groupby(['model','split','family','alpha']):
        for measure in measures:
            fold_means=group.groupby('fold')[measure].mean()
            rows.append(dict(model=model,split=split,family=family,alpha=alpha,measure=measure,
                mean=fold_means.mean(),between_fold_sd=fold_means.std(ddof=1),
                mean_within_fold_seed_sd=group.groupby('fold')[measure].std(ddof=1).mean(),
                minimum=group[measure].min(),maximum=group[measure].max()))
    result=pd.DataFrame(rows);result.to_csv(destination,index=False)
    return result


if __name__=='__main__':
    main()
