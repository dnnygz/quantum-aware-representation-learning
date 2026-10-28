"""Analysis-only guards and reconciliation with the real saved pilot when available."""
from pathlib import Path
import json
import pandas as pd
import pytest
from src.phase2.analysis_artifacts import ArtifactError, _compare, load_analysis_artifacts
from src.phase2.analysis_views import generalization_table

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/phase2/pilot_fold0_50e_paired_v1'
ANALYSIS = ROOT/'reports/phase2/pilot_fold0_50e_verified'


def test_missing_inputs_are_informative_and_do_not_create_output(tmp_path):
    with pytest.raises(ArtifactError,match='QARL_RUN_DIR') as error:
        load_analysis_artifacts(tmp_path/'missing_run',tmp_path/'missing_analysis')
    assert 'config.json' in str(error.value)
    assert list(tmp_path.iterdir()) == []


def test_locked_or_unimplemented_experiment_is_rejected_before_source_reads(tmp_path):
    run,analysis = tmp_path/'run',tmp_path/'analysis'
    run.mkdir();analysis.mkdir()
    (run/'config.json').write_text(json.dumps(dict(folds=[0],epochs=50,seeds=[1729,1730,1731],experiments=['A','B'])))
    (run/'provenance.json').write_text(json.dumps(dict(stage='locked_test',locked_test_evaluated=True)))
    for name in ['comparison.csv','resources.json']:
        (run/name).write_text('{}')
    (analysis/'analysis_provenance.json').write_text('{}')
    with pytest.raises(ArtifactError,match='no locked test'):
        load_analysis_artifacts(run,analysis)


def test_changed_or_duplicate_official_table_is_rejected():
    frame = pd.DataFrame(dict(seed=[1729,1730],spearman_B_probe=[.7,.8]))
    altered = frame.copy()
    altered.loc[0,'spearman_B_probe'] = .71
    with pytest.raises(ArtifactError,match='does not match'):
        _compare('paired_comparisons',frame,altered)
    with pytest.raises(ArtifactError,match='duplicate'):
        _compare('paired_comparisons',frame,pd.concat([frame,frame]))
    _compare('paired_comparisons',frame,frame.iloc[::-1])


def test_generalization_gap_retains_online_measurement_context():
    curves = pd.DataFrame(dict(experiment=['B'],seed=[1729],epoch=[50],train_loss=[.05],validation_loss=[.25]))
    result = generalization_table(curves)
    assert result.validation_minus_online_train_bce.iloc[0] == pytest.approx(.2)
    pd.testing.assert_frame_equal(curves,result[curves.columns])


def test_notebook_executes_using_only_repository_summaries(monkeypatch):
    """Exercise the actual notebook while forbidding original/private artifacts."""
    import builtins
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import IPython.display
    from matplotlib.figure import Figure

    rendered = []
    def display(value):
        if isinstance(value, Figure):
            value.canvas.draw()
            rendered.append(value)
    monkeypatch.setattr(IPython.display, 'display', display)
    monkeypatch.setenv('QARL_PROJECT_ROOT', str(ROOT))
    monkeypatch.setenv('QARL_ANALYSIS_DIR', str(ANALYSIS))
    original_open = builtins.open
    original_path_open = Path.open
    def check(path):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            assert not resolved.is_relative_to(ROOT/'runs')
            assert not resolved.is_relative_to(ROOT/'data')
            assert resolved.suffix not in {'.npz', '.pt', '.pkl', '.png', '.svg'}
    def guarded_open(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)
    def guarded_path_open(path, *args, **kwargs):
        check(path)
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', guarded_open)
    monkeypatch.setattr(Path, 'open', guarded_path_open)
    notebook = json.loads((ROOT/'notebooks/01_representation_analysis.ipynb').read_text())
    namespace = {'__name__': '__main__'}
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
    assert len(rendered) == 6
    assert namespace['tables']['metrics_by_seed'].shape[0] == 15
    assert len(namespace['tables']['learning_curves']) == 450
    plt.close('all')


@pytest.mark.skipif(not (RUN/'comparison.csv').exists(),reason='Local unversioned pilot artifacts absent')
def test_pilot_reconciliation_never_trains_loads_checkpoints_or_reads_cohort(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('pennylane')
    import src.phase2.training
    import src.phase2.data
    def forbidden(*args,**kwargs):
        raise AssertionError('Analysis must not train, load checkpoints, or read cohort')
    monkeypatch.setattr(src.phase2.training,'train',forbidden)
    monkeypatch.setattr(src.phase2.data,'load_development',forbidden)
    monkeypatch.setattr(torch,'load',forbidden)
    original_open = Path.open
    def guarded_open(path,*args,**kwargs):
        if path.resolve().is_relative_to(ROOT/'data'):
            forbidden()
        if any(mode in str(args[:1]) for mode in ['w','a','+']):
            raise AssertionError('Analysis adapter attempted a write')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    artifacts = load_analysis_artifacts(RUN,ANALYSIS)
    assert artifacts['checks'].official_calculation_matches.all()
    assert artifacts['tables']['metrics_by_seed'].shape[0] == 15
    assert artifacts['tables']['learning_curves'].shape[0] == 450
    assert set(artifacts['tables']['latent_spectrum'].rank_relative_1e6) == {3,4}


@pytest.mark.skipif(not (RUN/'comparison.csv').exists(), reason='Local development archive absent')
def test_generator_reuses_reconciliation_and_preserves_saved_tables(tmp_path, monkeypatch):
    """Regenerate an analysis from saved development outputs, never training."""
    pytest.importorskip('torch')
    pytest.importorskip('pennylane')
    import src.phase2.training
    import src.phase2.data
    from src.phase2.analyze_pilot import analyze
    from src.phase2.analysis_artifacts import TABLE_KEYS
    from src.phase2.analysis_views import read_csv
    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-output analysis must not train or load cohort data')
    monkeypatch.setattr(src.phase2.training, 'train', forbidden)
    monkeypatch.setattr(src.phase2.data, 'load_development', forbidden)
    output = tmp_path/'analysis'
    analyze(RUN, output, ROOT/'runs/phase2/preflight_pilot_50e/diagnostics.json',
            ROOT/'runs/phase2/smoke_verified_1729')
    for name in TABLE_KEYS:
        _compare(name, read_csv(output/f'{name}.csv'), read_csv(ANALYSIS/f'{name}.csv'))
    assert len(list((output/'figures').glob('*.png'))) == 8
    assert len(list((output/'figures').glob('*.svg'))) == 8
    assert not (output/'analysis_source.py').exists()
    report = (output/'report.md').read_text()
    assert '$summary_table' not in report
    assert '0.9268' in report
