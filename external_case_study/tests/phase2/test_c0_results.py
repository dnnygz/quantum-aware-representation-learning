"""Read-only aggregation and notebook safeguards on synthetic aggregate inputs."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.phase2.analyze_c0 import summarize
from src.phase2.c0_views import load_results


def test_fold_summary_does_not_treat_seeds_as_independent_folds():
    data = pd.DataFrame({'arm': ['Q'] * 6, 'fold': [0, 0, 0, 1, 1, 1],
                         'score': [0., 1., 2., 4., 4., 4.]})
    folds, result = summarize(data, ['arm'], ['score'])
    np.testing.assert_allclose(folds.score, [1., 4.])
    assert result.score_mean.iloc[0] == 2.5
    assert result.score_std.iloc[0] == pytest.approx(np.std([1., 4.], ddof=1))


def test_missing_aggregates_fail_without_training(tmp_path):
    with pytest.raises(FileNotFoundError, match='No training will be performed'):
        load_results(tmp_path)
    assert not list(tmp_path.iterdir())


def test_notebook_is_clean_and_uses_aggregate_loader_only():
    notebook = json.loads(Path('notebooks/05_c0_development_results.ipynb').read_text())
    source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
    ast.parse(source)
    assert 'load_results' in source
    for forbidden in ['.fit(', 'torch.load', 'np.load', 'read_parquet', 'pickle.load', 'run_c0']:
        assert forbidden not in source
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['outputs'] == []
            assert cell['execution_count'] is None


def test_plot_loader_has_no_model_or_cohort_dependencies():
    module = ast.parse(Path('src/phase2/c0_views.py').read_text())
    imports = [node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
    imports += [alias.name for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and any(token in name for token in ['torch', 'sklearn', 'c0_data', 'training']) for name in imports)
