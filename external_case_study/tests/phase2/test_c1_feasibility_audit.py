"""Checks for descriptive score accounting, without model fitting or research data."""
import numpy as np
import pytest

from src.phase2.c1_feasibility_audit import audit, paired_diagnostics, score_diagnostics


def test_pairwise_auc_counts_ties_and_reversals():
    y = np.array([0, 0, 1, 1])
    r = score_diagnostics(y, np.array([.2, .6, .6, .8]))
    assert r['roc_auc'] == .875
    assert r['tied_pair_fraction'] == .25
    assert r['reversed_pair_fraction'] == 0


def test_monotone_link_changes_loss_but_not_ranking():
    y = np.array([0, 0, 1, 1])
    p = np.array([.1, .7, .6, .9])
    transformed = 1/(1+np.exp(-.5*np.log(p/(1-p))))
    a, b = score_diagnostics(y, p), score_diagnostics(y, transformed)
    assert a['roc_auc'] == b['roc_auc']
    assert a['log_loss'] != b['log_loss']
    assert a['balanced_accuracy'] == b['balanced_accuracy']


def test_paired_accounting_includes_four_error_groups():
    y = np.array([0, 0, 1, 1])
    v = np.array([.2, .8, .3, .7])
    p = np.array([.1, .7, .8, .4])
    r = paired_diagnostics(y, v, p)
    assert all(r[k+'_fraction'] == .25 for k in
               ['both_correct', 'both_wrong', 'lr_only_correct', 'vqc_only_correct'])
    assert r['delta_loss'] == pytest.approx(score_diagnostics(y, p)['log_loss']-score_diagnostics(y, v)['log_loss'])
    assert r['delta_auc'] == pytest.approx(score_diagnostics(y, p)['roc_auc']-score_diagnostics(y, v)['roc_auc'])


@pytest.mark.parametrize('p', [[0., .9], [.1, np.nan], [.1, 1.1]])
def test_invalid_probability_inputs_fail(p):
    with pytest.raises(ValueError):
        score_diagnostics(np.array([0, 1]), np.array(p))


def test_refuses_to_overwrite_report(tmp_path):
    with pytest.raises(FileExistsError):
        audit(tmp_path)


def test_rejects_locked_overlap_before_prediction_access(tmp_path, monkeypatch):
    import hashlib
    import json
    monkeypatch.chdir(tmp_path)
    split_path = tmp_path/'data/processed/splits/phase1c_inner_cv.json'
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(dict(development_case_ids=list(range(835)),
                                         locked_test_case_ids=list(range(209)))))
    manifest = tmp_path/'reports/phase2/cv_5fold_50e_v1/provenance.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(dict(protocol_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest())))
    with pytest.raises(ValueError, match='development/locked'):
        audit(tmp_path/'new_output')
    assert not (tmp_path/'new_output').exists()
