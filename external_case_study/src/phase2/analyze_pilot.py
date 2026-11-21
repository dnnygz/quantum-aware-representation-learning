"""Reproducible descriptive analysis of a completed, fixed fold-0 pilot.

Reads saved outputs only. Does not fit models, access raw data, or select epochs.
"""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis_views import (MEASURES, read_csv, paired_predictions,
                             plot_generalization, plot_latent_spectrum)
from .analysis_artifacts import load_analysis_artifacts
from .run import write_json
from src.data.utils import sha256_file
import torch


def markdown(frame, decimals=4):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.{decimals}f}'
        return str(x)
    return ('| '+' | '.join(map(str,frame.columns))+' |\n|'+ '|'.join(['---']*len(frame.columns))+'|\n'
            +'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)))


def analyze(run_path, output, preflight, smoke):
    run_path, output = Path(run_path), Path(output)
    artifacts = load_analysis_artifacts(run_path)
    config, tables = artifacts['config'], artifacts['tables']
    table = tables['metrics_by_seed']
    summary = tables['descriptive_summary'].set_index('experiment')
    paired = tables['paired_comparisons']
    rank_table, angle_table = tables['latent_spectrum'], tables['angular_diagnostics']
    curves = dict(tuple(tables['learning_curves'].groupby(['experiment', 'seed'], sort=False)))
    output.mkdir(parents=True, exist_ok=False)
    figures = output/'figures'
    figures.mkdir()
    for name, frame in tables.items():
        frame.to_csv(output/f'{name}.csv', index=False)
    initialization, solution_states, scores = [], [], {}
    for seed in config['seeds']:
        a_initial = torch.load(run_path/f'fold_0_A/A_seed_{seed}/initial_model.pt', weights_only=True)
        b_initial = torch.load(run_path/f'fold_0_B/B_seed_{seed}/initial_model.pt', weights_only=True)
        if not torch.equal(a_initial['head.weights'], b_initial['head.weights']):
            raise ValueError('Unpaired quantum initialization')
        initialization.append(dict(seed=seed, quantum_weights_identical=True))
        final_state = torch.load(run_path/f'fold_0_B/B_seed_{seed}/model.pt', weights_only=True)
        solution_states.append(dict(seed=seed,
            encoder_output_weight_row_norms=final_state['encoder.2.weight'].norm(dim=1).tolist(),
            encoder_output_bias=final_state['encoder.2.bias'].tolist(),
            quantum_weights=final_state['head.weights'].tolist()))
        base = run_path/'fold_0_B'
        scores[seed], _ = paired_predictions(
            read_csv(base/f'B_seed_{seed}'/'validation_predictions.csv'),
            read_csv(base/f'B_frozen_probe_seed_{seed}'/'validation_predictions.csv'))
    write_json(output/'initialization_checks.json',initialization)
    write_json(output/'solution_state_diagnostics.json',solution_states)
    shutil.copy2(preflight,output/'preflight_diagnostics.json')
    diagnostics = json.loads(Path(preflight).read_text())
    # Shared dynamic views; static exports remain optional local artifacts.
    from .analysis_views import plot_pilot_figures, plot_private_diagnostics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':120,'savefig.dpi':160})
    rendered = plot_pilot_figures(curves)
    rendered.update(plot_private_diagnostics(scores, paired, run_path, config['seeds']))
    rendered['generalization'] = plot_generalization(tables['learning_curves'])
    rendered['latent_spectrum'] = plot_latent_spectrum(rank_table)
    for name, figure in rendered.items():
        figure.savefig(figures/f'{name}.png', bbox_inches='tight')
        figure.savefig(figures/f'{name}.svg', bbox_inches='tight')
        plt.close(figure)
    # Same-seed A has unchanged initialization and optimization; its first two
    # epochs should reproduce the old smoke. B changed initialization policy.
    smoke=Path(smoke)
    old_a=read_csv(smoke/'comparison.csv').query("experiment == 'A'").iloc[0]
    old_curve=read_csv(smoke/'fold_0_A/A_seed_1729/curves.csv')
    c=curves['A',1729]
    first_two_match=np.allclose(c[['train_loss','validation_loss']].iloc[:2],
                               old_curve[['train_loss','validation_loss']],atol=1e-12,rtol=0)
    write_json(output/'smoke_comparison.json',dict(A_first_two_epochs_match=bool(first_two_match),
        smoke_roc_auc=float(old_a.roc_auc),pilot_roc_auc=float(table.query("experiment=='A' and seed==1729").roc_auc.iloc[0]),
        smoke_validation_loss=float(old_curve.validation_loss.iloc[-1]),
        pilot_validation_loss=float(c.validation_loss.iloc[-1]),
        B_smoke_not_directly_comparable='quantum initialization policy changed'))
    resources=json.loads((run_path/'resources.json').read_text())
    sums=summary.reset_index()
    compact=pd.DataFrame({'Model':sums.experiment,
        **{m:[f'{a:.4f} ± {b:.4f}' for a,b in zip(sums[m+'_mean'],sums[m+'_std'])]
           for m in ['roc_auc','ap_er_positive','ap_er_negative','balanced_accuracy','log_loss']}})
    from string import Template
    template_path = Path(__file__).resolve().parents[2]/'reports/templates/phase2_pilot_report.md'
    text = Template(template_path.read_text()).substitute(
        summary_table=markdown(compact), metrics_table=markdown(table[['experiment','seed']+MEASURES]),
        paired_table=markdown(paired[['seed','B_minus_A_roc_auc','probe_minus_B_roc_auc','spearman_B_probe']]),
        angles_table=markdown(angle_table.query("checkpoint=='final' and coordinate=='pooled'")[[
            'experiment','seed','split','angular_saturation_95','angular_saturation_99','tanh_derivative_below_001']]),
        spectrum_table=markdown(rank_table.query("checkpoint=='final' and split=='train'")[[
            'experiment','seed','rank_relative_1e6','singular_0','singular_1','singular_2','singular_3']]),
        cost_table=markdown(table[['experiment','seed','seconds','preprocessing_seconds',
                                  'qnode_calls','circuit_sample_evaluations']].fillna('n/a')),
        sensitivity_error=f"{max(d['max_finite_difference_error'] for d in diagnostics['sensitivity']):.3g}",
        synthetic_initial_loss=f"{diagnostics['synthetic']['initial_loss']:.6f}",
        synthetic_final_loss=f"{diagnostics['synthetic']['final_loss']:.6f}",
        synthetic_accuracy=f"{diagnostics['synthetic']['accuracy']:.1f}",
        wall_seconds=f"{resources['wall_seconds']:.2f}",rss_gib=f"{resources['process_peak_rss']/1024**3:.2f}",
        cost_B_over_A=f"{summary.loc['B','seconds_mean']/summary.loc['A','seconds_mean']:.2f}",
        cost_B_over_classical=f"{summary.loc['B','seconds_mean']/summary.loc['B_classical','seconds_mean']:.1f}",
        first_two_match=str(bool(first_two_match)),smoke_auc=f"{old_a.roc_auc:.4f}",
        final_a_auc=f"{table.query("experiment=='A' and seed==1729").roc_auc.iloc[0]:.4f}",
        smoke_validation_loss=f"{old_curve.validation_loss.iloc[-1]:.4f}",
        final_a_validation_loss=f"{curves['A',1729].validation_loss.iloc[-1]:.4f}",
        delta_mean=f"{paired.B_minus_A_roc_auc.mean():.4f}",
        delta_std=f"{paired.B_minus_A_roc_auc.std(ddof=1):.4f}",
        delta_values=', '.join(f'{x:.4f}' for x in paired.B_minus_A_roc_auc),
        b_auc_std=f"{summary.loc['B','roc_auc_std']:.4f}",a_auc_std=f"{summary.loc['A','roc_auc_std']:.4f}",
        a_classical_auc=f"{summary.loc['A_classical','roc_auc_mean']:.4f}",
        probe_delta_values=', '.join(f'{x:.4f}' for x in paired.probe_minus_B_roc_auc),
        rho_values=', '.join(f'{x:.4f}' for x in paired.spearman_B_probe),
        b_train_losses=', '.join(f"{curves['B',seed].train_loss.iloc[-1]:.4f}" for seed in config['seeds']),
        b_validation_losses=', '.join(f"{curves['B',seed].validation_loss.iloc[-1]:.4f}" for seed in config['seeds']))
    (output/'report.md').write_text(text)
    sources=[run_path/'config.json',run_path/'comparison.csv',Path(preflight)]
    sources+=list(run_path.rglob('curves.csv'))+list(run_path.rglob('validation_predictions.csv'))
    sources+=list(run_path.rglob('model.pt'))
    sources+=list(run_path.rglob('latent_diagnostics.npz'))+list(run_path.rglob('initial_model.pt'))
    write_json(output/'analysis_provenance.json',dict(run=str(run_path),
        source_sha256={str(p):sha256_file(p) for p in sources},
        analysis_sha256=sha256_file(Path(__file__)),
        report_template_sha256=sha256_file(template_path),
        analysis_dependency_sha256={name:sha256_file(Path(__file__).parent/name)
            for name in ['analysis_views.py','analysis_artifacts.py','evaluation.py']},
        recomputation='All 15 validation metric sets independently recomputed from saved predictions',
        initialization='All three A/B initial checkpoints compared with torch.equal',
        rendered_verification='Pending visual inspection of exported figures'))
    return table,paired,summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--preflight',required=True)
    parser.add_argument('--smoke',default='runs/phase2/smoke_verified_1729')
    args=parser.parse_args()
    table,paired,summary=analyze(args.run,args.output,args.preflight,args.smoke)
    print(table.to_string(index=False))
    print(paired.to_string(index=False))
    print(summary.to_string())


if __name__=='__main__':
    main()
