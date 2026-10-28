"""Tests of CV pairing, variance summaries, rank definitions, and safe input handling."""
import numpy as np
import pandas as pd
import pytest
from src.phase2.analyze_cv import latent_summary, paired_table, summarize, analyze
from src.phase2.analysis_views import MODELS, MEASURES
from src.phase2.cv_views import load_tables


def fixture():
    return pd.DataFrame([dict(fold=f,seed=s,experiment=m,
        **{k:float(f)+.1*(s-1729)+i for k in MEASURES+['brier']})
        for f in range(5) for s in [1729,1730,1731] for i,m in enumerate(MODELS)])


def test_pairing_preserves_fold_and_seed():
    table=fixture().sample(frac=1,random_state=4)
    pairs=paired_table(table)
    assert len(pairs)==15
    np.testing.assert_allclose(pairs.B_minus_A_roc_auc,2)
    np.testing.assert_allclose(pairs.B_minus_B_frozen_probe_roc_auc,-1)
    with pytest.raises(ValueError,match='Duplicate'):
        paired_table(pd.concat([table,table.iloc[:1]]))
    with pytest.raises(ValueError,match='Incomplete'):
        paired_table(table.iloc[1:])


def test_variation_is_separated():
    summary=summarize(fixture())
    row=summary[(summary.experiment=='A')&(summary.metric=='roc_auc')].iloc[0]
    assert row['mean']==pytest.approx(2.1)
    assert row.between_fold_sd==pytest.approx(np.std(range(5),ddof=1))
    assert row.mean_within_fold_seed_sd==pytest.approx(.1)


def test_rank_and_coordinate_contraction():
    rng=np.random.default_rng(1729)
    q,_=np.linalg.qr(rng.normal(size=(30,4))-rng.normal(size=(30,4)).mean(axis=0))
    q=q-q.mean(axis=0)
    full=latent_summary(q)
    assert full['rank_relative_1e6']==4
    contracted=q.copy(); contracted[:,3]=2
    result=latent_summary(contracted)
    assert result['rank_relative_1e6']==3
    assert result['coordinate_3_std']==0
    assert 1<=result['variance_entropy_rank']<=3+1e-12
    single=np.zeros((30,4));single[:,0]=np.arange(30)
    assert latent_summary(single)['participation_ratio']==pytest.approx(1)
    assert latent_summary(np.zeros((30,4)))['variance_entropy_rank']==0


def test_missing_artifacts_are_informative(tmp_path):
    with pytest.raises(FileNotFoundError,match='No training'):
        load_tables(tmp_path)


def test_analysis_never_overwrites(tmp_path):
    with pytest.raises(FileExistsError):
        analyze('missing','missing','missing',tmp_path)


def test_metrics_plot_retains_every_fold_seed():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.phase2.cv_views import plot_metrics
    fig=plot_metrics(fixture())
    assert len(fig.axes)==5
    # Each metric retains all 75 observations plus 25 fold means.
    for ax in fig.axes:
        assert sum(len(c.get_offsets()) for c in ax.collections)==100
    plt.close(fig)


def test_saved_cv_reconstruction_never_fits_or_loads_cohort(tmp_path,monkeypatch):
    from pathlib import Path
    import src.phase2.training
    import src.phase2.data
    from sklearn.linear_model import LogisticRegression
    from src.phase2.analysis_views import read_csv
    pilot=Path('runs/phase2/pilot_fold0_50e_paired_v1')
    remaining=Path('runs/phase2/cv_folds1to4_50e_paired_v1')
    if not (remaining/'comparison.csv').exists():
        pytest.skip('Trusted local completed CV archive is not available')
    def forbidden(*args,**kwargs):
        raise AssertionError('CV analysis must not fit models or load cohort data')
    monkeypatch.setattr(src.phase2.training,'train',forbidden)
    monkeypatch.setattr(src.phase2.data,'load_development',forbidden)
    monkeypatch.setattr(src.phase2.data.Representation,'fit',forbidden)
    monkeypatch.setattr(LogisticRegression,'fit',forbidden)
    result=analyze(pilot,remaining,'data/processed/splits/phase1c_inner_cv.json',tmp_path/'cv')
    assert len(result['metrics'])==75
    assert len(result['paired'])==15
    assert len(result['curves'])==2250
    assert len(result['generalization'])==75
    assert set(result['latent'].query("experiment=='B' and checkpoint=='final'").rank_relative_1e6)=={2,3,4}
    saved=read_csv('reports/phase2/cv_5fold_50e_v1/summary.csv')
    pd.testing.assert_frame_equal(result['summary'],saved,check_exact=False,atol=1e-12,rtol=1e-12)
