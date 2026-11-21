"""Leakage-safe, fold-local RNA-seq preprocessing for Phase 1B.

The canonical cohort remains raw counts.  Every component in this module uses
the estimator contract ``fit(training)`` / ``transform(validation_or_test)``.
No target labels are accepted or inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from .utils import (
    canonical_json_bytes,
    configure_logging,
    dependency_versions,
    git_state,
    load_config,
    resolve_path,
    sha256_file,
    utc_now,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)


class PreprocessingValidationError(ValueError):
    """Raised when counts or a fitted preprocessing state are invalid."""


def _as_frame(X: Any, *, raw_counts: bool, context: str) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{context} requires a pandas DataFrame with named genes")
    if X.index.has_duplicates:
        raise PreprocessingValidationError(f"{context} contains duplicate patient IDs")
    if X.columns.has_duplicates:
        raise PreprocessingValidationError(f"{context} contains duplicate gene IDs")
    if X.empty or X.shape[1] == 0:
        raise PreprocessingValidationError(f"{context} must contain patients and genes")
    values = X.to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise PreprocessingValidationError(f"{context} contains NaN or infinite values")
    if raw_counts:
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise PreprocessingValidationError(
                f"{context} raw counts must be non-negative integers"
            )
        if (values.sum(axis=1) <= 0).any():
            raise PreprocessingValidationError(f"{context} contains an empty library")
    return X


def _align_features(X: pd.DataFrame, expected: Sequence[str], context: str) -> pd.DataFrame:
    _as_frame(X, raw_counts=False, context=context)
    expected_index = pd.Index(expected, dtype=object)
    observed = pd.Index(X.columns.astype(str), dtype=object)
    missing = expected_index.difference(observed)
    extra = observed.difference(expected_index)
    if len(missing) or len(extra):
        raise PreprocessingValidationError(
            f"{context} gene schema differs from fit; "
            f"missing={missing[:10].tolist()}, extra={extra[:10].tolist()}"
        )
    # Feature order at transform time is never trusted.
    return X.loc[:, expected_index]


def _sha256_strings(values: Iterable[str]) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "q1": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q3": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
    }


class TrainingExpressionFilter(BaseEstimator, TransformerMixin):
    """Keep genes with CPM >= threshold in a fixed prevalence of training patients."""

    def __init__(self, cpm_threshold: float = 1.0, min_training_prevalence: float = 0.10):
        self.cpm_threshold = cpm_threshold
        self.min_training_prevalence = min_training_prevalence

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainingExpressionFilter":
        del y
        frame = _as_frame(X, raw_counts=True, context="expression-filter fit")
        if float(self.cpm_threshold) <= 0:
            raise ValueError("cpm_threshold must be positive")
        prevalence = float(self.min_training_prevalence)
        if not 0 < prevalence <= 1:
            raise ValueError("min_training_prevalence must be in (0, 1]")

        counts = frame.to_numpy(dtype=float, copy=False)
        library_sizes = counts.sum(axis=1)
        minimum_samples = math.ceil(prevalence * len(frame))
        expressed_samples = np.zeros(frame.shape[1], dtype=np.int64)
        for start in range(0, frame.shape[1], 2048):
            block = counts[:, start : start + 2048]
            expressed_samples[start : start + block.shape[1]] = (
                block * 1_000_000.0 / library_sizes[:, None] >= float(self.cpm_threshold)
            ).sum(axis=0)

        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.minimum_samples_ = int(minimum_samples)
        self.expressed_samples_ = expressed_samples
        self.keep_mask_ = expressed_samples >= minimum_samples
        if not self.keep_mask_.any():
            raise PreprocessingValidationError("Expression filter removed every gene")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _align_features(X, self.feature_names_in_, "expression-filter transform")
        _as_frame(frame, raw_counts=True, context="expression-filter transform")
        return frame.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_out_.copy()


class FrozenTMMLogCPM(BaseEstimator, TransformerMixin):
    """Training-fitted TMM followed by edgeR-style log2-CPM.

    A single training library is frozen as reference.  A new patient's raw TMM
    factor is calculated independently against that reference and divided by
    the geometric-mean calibration learned on training.  Consequently, one
    test patient cannot change another test patient's representation.
    """

    def __init__(
        self,
        logratio_trim: float = 0.30,
        sum_trim: float = 0.05,
        weighted: bool = True,
        prior_count: float = 2.0,
    ):
        self.logratio_trim = logratio_trim
        self.sum_trim = sum_trim
        self.weighted = weighted
        self.prior_count = prior_count

    @staticmethod
    def _upper_quartile(counts: np.ndarray, library_sizes: np.ndarray) -> np.ndarray:
        return np.quantile(counts, 0.75, axis=1) / library_sizes

    @staticmethod
    def _average_ranks(values: np.ndarray) -> np.ndarray:
        """One-based average ranks, matching R's default ties.method."""

        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=float)
        starts = np.flatnonzero(
            np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
        )
        stops = np.concatenate((starts[1:], [len(values)]))
        average_ranks = 0.5 * ((starts + 1) + stops)
        ranks[order] = np.repeat(average_ranks, stops - starts)
        return ranks

    def _raw_factor(self, observed: np.ndarray, library_size: float) -> float:
        reference = self.reference_counts_
        ref_library = self.reference_library_size_
        valid = (observed > 0) & (reference > 0)
        if valid.sum() < 2:
            raise PreprocessingValidationError(
                "Too few shared positive genes to calculate a TMM factor"
            )
        obs = observed[valid].astype(float, copy=False)
        ref = reference[valid].astype(float, copy=False)
        log_r = np.log2((obs / library_size) / (ref / ref_library))
        abs_e = 0.5 * np.log2((obs / library_size) * (ref / ref_library))
        variance = (library_size - obs) / library_size / obs
        variance += (ref_library - ref) / ref_library / ref
        finite = np.isfinite(log_r) & np.isfinite(abs_e) & np.isfinite(variance) & (variance > 0)
        log_r, abs_e, variance = log_r[finite], abs_e[finite], variance[finite]
        if log_r.size < 2:
            raise PreprocessingValidationError("No finite genes remain for TMM")

        n_values = log_r.size
        lower_m = math.floor(n_values * float(self.logratio_trim)) + 1
        upper_m = n_values + 1 - lower_m
        lower_a = math.floor(n_values * float(self.sum_trim)) + 1
        upper_a = n_values + 1 - lower_a
        m_ranks = self._average_ranks(log_r)
        a_ranks = self._average_ranks(abs_e)
        keep = (
            (m_ranks >= lower_m)
            & (m_ranks <= upper_m)
            & (a_ranks >= lower_a)
            & (a_ranks <= upper_a)
        )
        if not keep.any():
            raise PreprocessingValidationError("TMM trimming removed every comparable gene")
        if bool(self.weighted):
            weights = 1.0 / variance[keep]
            mean_log_ratio = float(np.sum(weights * log_r[keep]) / np.sum(weights))
        else:
            mean_log_ratio = float(np.mean(log_r[keep]))
        return float(2.0**mean_log_ratio)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrozenTMMLogCPM":
        del y
        frame = _as_frame(X, raw_counts=True, context="TMM fit")
        for name, value in {
            "logratio_trim": self.logratio_trim,
            "sum_trim": self.sum_trim,
        }.items():
            if not 0 <= float(value) < 0.5:
                raise ValueError(f"{name} must be in [0, 0.5)")
        if float(self.prior_count) <= 0:
            raise ValueError("prior_count must be positive")

        counts = frame.to_numpy(dtype=float, copy=False)
        library_sizes = counts.sum(axis=1)
        upper_quartiles = self._upper_quartile(counts, library_sizes)
        if float(np.median(upper_quartiles)) < 1e-20:
            reference_position = int(np.argmin(np.sum(np.sqrt(counts), axis=1)))
        else:
            reference_position = int(
                np.argmin(np.abs(upper_quartiles - float(np.mean(upper_quartiles))))
            )
        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.reference_position_ = reference_position
        self.reference_sample_id_ = str(frame.index[reference_position])
        self.reference_counts_ = counts[reference_position].astype(np.int64, copy=True)
        self.reference_library_size_ = float(library_sizes[reference_position])

        raw_factors = np.asarray(
            [self._raw_factor(row, lib) for row, lib in zip(counts, library_sizes)], dtype=float
        )
        if not np.isfinite(raw_factors).all() or (raw_factors <= 0).any():
            raise PreprocessingValidationError("TMM produced invalid training factors")
        self.calibration_ = float(np.exp(np.mean(np.log(raw_factors))))
        self.training_raw_factors_ = raw_factors
        self.training_factors_ = raw_factors / self.calibration_
        self.training_library_sizes_ = library_sizes
        self.training_effective_library_sizes_ = library_sizes * self.training_factors_
        self.mean_training_effective_library_size_ = float(
            np.mean(self.training_effective_library_sizes_)
        )
        return self

    def _factors(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        counts = frame.to_numpy(dtype=float, copy=False)
        library_sizes = counts.sum(axis=1)
        if (library_sizes <= 0).any():
            raise PreprocessingValidationError("TMM transform contains an empty library")
        raw = np.asarray(
            [self._raw_factor(row, lib) for row, lib in zip(counts, library_sizes)], dtype=float
        )
        factors = raw / self.calibration_
        effective = library_sizes * factors
        if not np.isfinite(factors).all() or (factors <= 0).any():
            raise PreprocessingValidationError("TMM transform produced invalid factors")
        return library_sizes, factors, effective

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _align_features(X, self.feature_names_in_, "TMM transform")
        _as_frame(frame, raw_counts=True, context="TMM transform")
        _, _, effective = self._factors(frame)
        counts = frame.to_numpy(dtype=float, copy=False)
        adjusted_prior = (
            float(self.prior_count)
            * effective
            / self.mean_training_effective_library_size_
        )
        cpm = (
            (counts + adjusted_prior[:, None])
            / (effective[:, None] + 2.0 * adjusted_prior[:, None])
            * 1_000_000.0
        )
        result = np.log2(cpm)
        if not np.isfinite(result).all():
            raise PreprocessingValidationError("log2-CPM produced NaN or infinite values")
        return pd.DataFrame(result, index=frame.index.copy(), columns=self.feature_names_in_)

    def diagnostics(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _align_features(X, self.feature_names_in_, "TMM diagnostics")
        library, factors, effective = self._factors(frame)
        return pd.DataFrame(
            {
                "patient_id": frame.index.astype(str),
                "filtered_library_size": library,
                "tmm_factor": factors,
                "effective_library_size": effective,
            }
        )

    def ma_values(self, sample: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        aligned = sample.reindex(self.feature_names_in_).to_numpy(dtype=float)
        valid = (aligned > 0) & (self.reference_counts_ > 0)
        obs = aligned[valid]
        ref = self.reference_counts_[valid].astype(float)
        library = float(aligned.sum())
        m_values = np.log2((obs / library) / (ref / self.reference_library_size_))
        a_values = 0.5 * np.log2(
            (obs / library) * (ref / self.reference_library_size_)
        )
        finite = np.isfinite(m_values) & np.isfinite(a_values)
        return a_values[finite], m_values[finite]


class ExactZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """Remove only features that are exactly constant in training."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ExactZeroVarianceFilter":
        del y
        frame = _as_frame(X, raw_counts=False, context="zero-variance fit")
        values = frame.to_numpy(dtype=float, copy=False)
        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.keep_mask_ = np.ptp(values, axis=0) != 0.0
        if not self.keep_mask_.any():
            raise PreprocessingValidationError("Every transformed gene has zero training variance")
        self.feature_names_out_ = self.feature_names_in_[self.keep_mask_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _align_features(X, self.feature_names_in_, "zero-variance transform")
        return frame.loc[:, self.feature_names_out_].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_out_.copy()


class TrainingStandardScaler(BaseEstimator, TransformerMixin):
    """DataFrame-preserving StandardScaler that learns from training only."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TrainingStandardScaler":
        del y
        frame = _as_frame(X, raw_counts=False, context="scaler fit")
        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.scaler_ = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.scaler_.fit(frame.to_numpy(dtype=float, copy=False))
        return self

    @property
    def mean_(self) -> np.ndarray:
        return self.scaler_.mean_

    @property
    def scale_(self) -> np.ndarray:
        return self.scaler_.scale_

    @property
    def var_(self) -> np.ndarray:
        return self.scaler_.var_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _align_features(X, self.feature_names_in_, "scaler transform")
        values = self.scaler_.transform(frame.to_numpy(dtype=float, copy=False))
        return pd.DataFrame(values, index=frame.index.copy(), columns=self.feature_names_in_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_in_.copy()


class RNASeqPreprocessor(BaseEstimator, TransformerMixin):
    """Composable primary Phase-1B preprocessing specification."""

    def __init__(
        self,
        cpm_threshold: float = 1.0,
        min_training_prevalence: float = 0.10,
        logratio_trim: float = 0.30,
        sum_trim: float = 0.05,
        weighted_tmm: bool = True,
        prior_count: float = 2.0,
        scale: bool = False,
    ):
        self.cpm_threshold = cpm_threshold
        self.min_training_prevalence = min_training_prevalence
        self.logratio_trim = logratio_trim
        self.sum_trim = sum_trim
        self.weighted_tmm = weighted_tmm
        self.prior_count = prior_count
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RNASeqPreprocessor":
        del y
        started = time.perf_counter()
        frame = _as_frame(X, raw_counts=True, context="RNA-seq preprocessor fit")
        self.feature_names_in_ = np.asarray(frame.columns.astype(str), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.expression_filter_ = TrainingExpressionFilter(
            cpm_threshold=self.cpm_threshold,
            min_training_prevalence=self.min_training_prevalence,
        ).fit(frame)
        filtered = self.expression_filter_.transform(frame)
        self.tmm_ = FrozenTMMLogCPM(
            logratio_trim=self.logratio_trim,
            sum_trim=self.sum_trim,
            weighted=self.weighted_tmm,
            prior_count=self.prior_count,
        ).fit(filtered)
        normalized = self.tmm_.transform(filtered)
        self.zero_variance_filter_ = ExactZeroVarianceFilter().fit(normalized)
        reduced = self.zero_variance_filter_.transform(normalized)
        self.scaler_ = TrainingStandardScaler().fit(reduced) if bool(self.scale) else None
        self.feature_names_out_ = np.asarray(reduced.columns.astype(str), dtype=object)
        self.fit_runtime_seconds_ = float(time.perf_counter() - started)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        filtered = self.expression_filter_.transform(X)
        normalized = self.tmm_.transform(filtered)
        reduced = self.zero_variance_filter_.transform(normalized)
        if self.scaler_ is not None:
            reduced = self.scaler_.transform(reduced)
        if list(reduced.columns) != list(self.feature_names_out_):
            raise AssertionError("Preprocessor changed the fitted feature order")
        return reduced

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        del input_features
        return self.feature_names_out_.copy()

    def parameter_fingerprint(self) -> str:
        """Hash every learned parameter; useful for leakage regression tests."""

        state: dict[str, Any] = {
            "input": self.feature_names_in_.tolist(),
            "expression_mask": self.expression_filter_.keep_mask_.astype(int).tolist(),
            "tmm_reference_sample": self.tmm_.reference_sample_id_,
            "tmm_reference_counts": self.tmm_.reference_counts_.astype(int).tolist(),
            "tmm_reference_library_size": self.tmm_.reference_library_size_,
            "tmm_calibration": self.tmm_.calibration_,
            "tmm_mean_effective_library": self.tmm_.mean_training_effective_library_size_,
            "zero_variance_mask": self.zero_variance_filter_.keep_mask_.astype(int).tolist(),
            "output": self.feature_names_out_.tolist(),
        }
        if self.scaler_ is not None:
            state["scaler"] = {
                "mean": self.scaler_.mean_.tolist(),
                "scale": self.scaler_.scale_.tolist(),
                "var": self.scaler_.var_.tolist(),
            }
        return hashlib.sha256(canonical_json_bytes(state)).hexdigest()


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def persist_fitted_preprocessor(
    preprocessor: RNASeqPreprocessor,
    output_dir: Path,
    training_diagnostics: pd.DataFrame,
) -> dict[str, str]:
    """Persist only fitted parameters, never a transformed patient matrix."""

    output_dir.mkdir(parents=True, exist_ok=True)
    input_genes = pd.Index(preprocessor.feature_names_in_)
    filtered_genes = pd.Index(preprocessor.expression_filter_.feature_names_out_)
    final_genes = pd.Index(preprocessor.feature_names_out_)
    feature_state = pd.DataFrame(
        {
            "input_order": np.arange(len(input_genes), dtype=np.int64),
            "gene_id": input_genes,
            "expression_filter_kept": input_genes.isin(filtered_genes),
        }
    )
    reference_by_gene = pd.Series(
        preprocessor.tmm_.reference_counts_, index=filtered_genes, dtype="Int64"
    )
    zero_keep_by_gene = pd.Series(
        preprocessor.zero_variance_filter_.keep_mask_, index=filtered_genes, dtype="boolean"
    )
    final_order = pd.Series(np.arange(len(final_genes)), index=final_genes, dtype="Int64")
    feature_state["tmm_reference_count"] = feature_state["gene_id"].map(reference_by_gene).astype("Int64")
    feature_state["zero_variance_kept"] = feature_state["gene_id"].map(zero_keep_by_gene).astype("boolean")
    feature_state["final_feature_order"] = feature_state["gene_id"].map(final_order).astype("Int64")
    if preprocessor.scaler_ is not None:
        feature_state["scaler_mean"] = feature_state["gene_id"].map(
            pd.Series(preprocessor.scaler_.mean_, index=final_genes)
        )
        feature_state["scaler_scale"] = feature_state["gene_id"].map(
            pd.Series(preprocessor.scaler_.scale_, index=final_genes)
        )

    feature_path = output_dir / "feature_state.parquet"
    training_path = output_dir / "training_sample_qc.parquet"
    _write_parquet_atomic(feature_state, feature_path)
    _write_parquet_atomic(training_diagnostics, training_path)
    state = {
        "created_at": utc_now(),
        "specification_id": "phase1b_primary_v1",
        "fit_semantics": "fit(training_only)/transform(validation_or_test)",
        "target_used": False,
        "parameters": {
            "cpm_threshold": float(preprocessor.cpm_threshold),
            "min_training_prevalence": float(preprocessor.min_training_prevalence),
            "minimum_training_samples": preprocessor.expression_filter_.minimum_samples_,
            "logratio_trim": float(preprocessor.logratio_trim),
            "sum_trim": float(preprocessor.sum_trim),
            "weighted_tmm": bool(preprocessor.weighted_tmm),
            "prior_count": float(preprocessor.prior_count),
            "scale": bool(preprocessor.scale),
        },
        "dimensions": {
            "genes_initial": int(preprocessor.n_features_in_),
            "genes_after_expression_filter": int(len(filtered_genes)),
            "genes_removed_zero_variance": int(len(filtered_genes) - len(final_genes)),
            "genes_final": int(len(final_genes)),
        },
        "tmm": {
            "reference_sample_id": preprocessor.tmm_.reference_sample_id_,
            "reference_library_size": preprocessor.tmm_.reference_library_size_,
            "training_geometric_mean_calibration": preprocessor.tmm_.calibration_,
            "mean_training_effective_library_size": preprocessor.tmm_.mean_training_effective_library_size_,
        },
        "feature_order_sha256": {
            "input": _sha256_strings(input_genes),
            "after_expression_filter": _sha256_strings(filtered_genes),
            "final": _sha256_strings(final_genes),
        },
        "parameter_fingerprint": preprocessor.parameter_fingerprint(),
        "files": {
            "feature_state": feature_path.name,
            "training_sample_qc": training_path.name,
        },
    }
    state_path = output_dir / "state.json"
    write_json_atomic(state_path, state)
    return {
        "state": sha256_file(state_path),
        "feature_state": sha256_file(feature_path),
        "training_sample_qc": sha256_file(training_path),
    }


def _factor_outliers(
    diagnostics: pd.DataFrame,
    *,
    lower: float,
    upper: float,
    iqr_multiplier: float,
) -> dict[str, Any]:
    factors = diagnostics["tmm_factor"].to_numpy(dtype=float)
    patient_ids = diagnostics["patient_id"].astype(str).to_numpy()
    log_factors = np.log2(factors)
    q1, q3 = np.quantile(log_factors, [0.25, 0.75])
    iqr = q3 - q1
    robust_low = q1 - iqr_multiplier * iqr
    robust_high = q3 + iqr_multiplier * iqr
    extreme_mask = (factors < lower) | (factors > upper)
    robust_mask = (log_factors < robust_low) | (log_factors > robust_high)
    minimum_position = int(np.argmin(factors))
    maximum_position = int(np.argmax(factors))
    return {
        "prespecified_extreme_bounds": [float(lower), float(upper)],
        "prespecified_extreme_count": int(extreme_mask.sum()),
        "prespecified_extreme_patient_ids": patient_ids[extreme_mask].tolist(),
        "robust_log2_iqr_bounds": [float(robust_low), float(robust_high)],
        "robust_outlier_count": int(robust_mask.sum()),
        "robust_outlier_patient_ids": patient_ids[robust_mask].tolist(),
        "minimum_factor_patient": {
            "patient_id": str(patient_ids[minimum_position]),
            "factor": float(factors[minimum_position]),
        },
        "maximum_factor_patient": {
            "patient_id": str(patient_ids[maximum_position]),
            "factor": float(factors[maximum_position]),
        },
    }


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_ma_plot_svg(
    path: Path,
    a_values: np.ndarray,
    m_values: np.ndarray,
    *,
    title: str,
    max_points: int,
) -> None:
    """Write a deterministic, dependency-free MA diagnostic plot."""

    if len(a_values) > max_points:
        positions = np.linspace(0, len(a_values) - 1, max_points, dtype=int)
        a_values, m_values = a_values[positions], m_values[positions]
    width, height = 900, 560
    left, right, top, bottom = 75, 25, 55, 65
    xmin, xmax = np.quantile(a_values, [0.005, 0.995])
    ymax = max(1.0, float(np.quantile(np.abs(m_values), 0.995)))
    ymin = -ymax

    def sx(value: float) -> float:
        return left + (value - xmin) / max(xmax - xmin, 1e-12) * (width - left - right)

    def sy(value: float) -> float:
        return top + (ymax - value) / max(ymax - ymin, 1e-12) * (height - top - bottom)

    circles = "".join(
        f'<circle cx="{sx(float(a)):.2f}" cy="{sy(float(m)):.2f}" r="1.15" />'
        for a, m in zip(a_values, m_values)
        if xmin <= a <= xmax and ymin <= m <= ymax
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="16">{_svg_escape(title)}</text>
<line x1="{left}" y1="{sy(0):.2f}" x2="{width-right}" y2="{sy(0):.2f}" stroke="#b42318" stroke-width="1"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#222"/>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#222"/>
<g fill="#2463a5" fill-opacity="0.28">{circles}</g>
<text x="{width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">A: mean normalized log abundance</text>
<text x="18" y="{height/2}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 {height/2})">M: log2 ratio to frozen training reference</text>
<text x="{left}" y="{height-bottom+20}" font-family="sans-serif" font-size="11">{xmin:.2f}</text>
<text x="{width-right}" y="{height-bottom+20}" text-anchor="end" font-family="sans-serif" font-size="11">{xmax:.2f}</text>
<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="sans-serif" font-size="11">{ymax:.2f}</text>
<text x="{left-8}" y="{height-bottom}" text-anchor="end" font-family="sans-serif" font-size="11">{ymin:.2f}</text>
</svg>\n'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(svg, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Partition:
    name: str
    train_ids: list[str]
    evaluation_ids: list[str]
    evaluation_role: str


def _partitions(splits: Mapping[str, Any]) -> list[Partition]:
    outer = splits["stratified_train_test"]
    result = [
        Partition(
            name="outer_train_test",
            train_ids=list(outer["train_case_ids"]),
            evaluation_ids=list(outer["test_case_ids"]),
            evaluation_role="test",
        )
    ]
    for fold in splits["stratified_kfold"]["folds"]:
        result.append(
            Partition(
                name=f"stratified_kfold_{fold['fold']}",
                train_ids=list(fold["train_case_ids"]),
                evaluation_ids=list(fold["validation_case_ids"]),
                evaluation_role="validation",
            )
        )
    return result


def _build_preprocessor(config: Mapping[str, Any]) -> RNASeqPreprocessor:
    expression_filter = config["expression_filter"]
    tmm = config["tmm"]
    log_cpm = config["log_cpm"]
    return RNASeqPreprocessor(
        cpm_threshold=float(expression_filter["cpm_threshold"]),
        min_training_prevalence=float(expression_filter["min_training_prevalence"]),
        logratio_trim=float(tmm["logratio_trim"]),
        sum_trim=float(tmm["sum_trim"]),
        weighted_tmm=bool(tmm["weighted"]),
        prior_count=float(log_cpm["prior_count"]),
        # Scaling is deliberately estimator-specific and is tested separately.
        scale=bool(config["scaling"].get("default_enabled", False)),
    )


def _dataset_summary(frame: pd.DataFrame, tmm_diagnostics: pd.DataFrame) -> dict[str, Any]:
    raw_library = frame.to_numpy(dtype=np.int64, copy=False).sum(axis=1, dtype=np.int64)
    return {
        "patients": int(len(frame)),
        "raw_library_size": _quantiles(raw_library),
        "filtered_library_size": _quantiles(tmm_diagnostics["filtered_library_size"].to_numpy()),
        "effective_library_size": _quantiles(tmm_diagnostics["effective_library_size"].to_numpy()),
        "tmm_factor": _quantiles(tmm_diagnostics["tmm_factor"].to_numpy()),
    }


def run_phase1b(config_path: str | Path) -> dict[str, Any]:
    """Fit every frozen partition and write state/QC without normalized matrices."""

    config, repo_root = load_config(config_path)
    pre_config = config.get("preprocessing")
    if not isinstance(pre_config, dict):
        raise ValueError("Configuration is missing the preprocessing section")
    processed_dir = resolve_path(repo_root, config["paths"]["processed_dir"])
    expression_path = processed_dir / "expression.parquet"
    genes_path = processed_dir / "genes.parquet"
    labels_path = processed_dir / "labels.parquet"
    metadata_path = processed_dir / "metadata.parquet"
    splits_path = processed_dir / "splits.json"
    canonical_paths = [expression_path, genes_path, labels_path, metadata_path, splits_path]
    for path in canonical_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical-v1 artifact: {path}")
    before_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in canonical_paths}

    expression = pd.read_parquet(expression_path)
    expression.index = expression.index.astype(str)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    artifacts_root = resolve_path(repo_root, pre_config["artifacts_dir"])
    diagnostics_config = pre_config["diagnostics"]
    plots_root = repo_root / "reports" / "data" / "preprocessing" / "ma_plots"
    run_started = time.perf_counter()
    partition_reports: list[dict[str, Any]] = []

    for partition in _partitions(splits):
        LOGGER.info("Fitting Phase 1B preprocessing for %s", partition.name)
        train = expression.loc[partition.train_ids]
        evaluation = expression.loc[partition.evaluation_ids]
        fit_started = time.perf_counter()
        preprocessor = _build_preprocessor(pre_config).fit(train)
        fit_seconds = time.perf_counter() - fit_started

        transform_started = time.perf_counter()
        transformed_train = preprocessor.transform(train)
        transformed_evaluation = preprocessor.transform(evaluation)
        transform_seconds = time.perf_counter() - transform_started
        filtered_train = preprocessor.expression_filter_.transform(train)
        filtered_evaluation = preprocessor.expression_filter_.transform(evaluation)
        train_tmm = preprocessor.tmm_.diagnostics(filtered_train)
        evaluation_tmm = preprocessor.tmm_.diagnostics(filtered_evaluation)

        diag_lower = float(diagnostics_config["extreme_factor_lower"])
        diag_upper = float(diagnostics_config["extreme_factor_upper"])
        diag_iqr = float(diagnostics_config["robust_outlier_iqr_multiplier"])
        train_outliers = _factor_outliers(
            train_tmm,
            lower=diag_lower,
            upper=diag_upper,
            iqr_multiplier=diag_iqr,
        )
        evaluation_outliers = _factor_outliers(
            evaluation_tmm,
            lower=diag_lower,
            upper=diag_upper,
            iqr_multiplier=diag_iqr,
        )
        state_hashes = persist_fitted_preprocessor(
            preprocessor, artifacts_root / partition.name, train_tmm
        )

        factor_distance = np.abs(np.log2(train_tmm["tmm_factor"].to_numpy()))
        selected_positions = {
            "median_factor": int(
                np.argmin(
                    np.abs(
                        train_tmm["tmm_factor"].to_numpy()
                        - np.median(train_tmm["tmm_factor"].to_numpy())
                    )
                )
            ),
            "most_extreme_factor": int(np.argmax(factor_distance)),
        }
        plot_paths: list[str] = []
        for role, position in selected_positions.items():
            patient_id = str(train_tmm.iloc[position]["patient_id"])
            a_values, m_values = preprocessor.tmm_.ma_values(filtered_train.loc[patient_id])
            plot_path = plots_root / f"{partition.name}__{role}.svg"
            write_ma_plot_svg(
                plot_path,
                a_values,
                m_values,
                title=f"{partition.name}: {role} ({patient_id})",
                max_points=int(diagnostics_config["ma_plot_max_points"]),
            )
            plot_paths.append(str(plot_path.relative_to(repo_root)))

        partition_reports.append(
            {
                "partition": partition.name,
                "evaluation_role": partition.evaluation_role,
                "fit_scope": "training_case_ids_only",
                "target_used": False,
                "patients": {
                    "training": int(len(train)),
                    partition.evaluation_role: int(len(evaluation)),
                },
                "genes": {
                    "initial": int(train.shape[1]),
                    "after_expression_filter": int(
                        len(preprocessor.expression_filter_.feature_names_out_)
                    ),
                    "removed_by_expression_filter": int(
                        train.shape[1] - len(preprocessor.expression_filter_.feature_names_out_)
                    ),
                    "removed_by_exact_zero_variance": int(
                        len(preprocessor.expression_filter_.feature_names_out_)
                        - len(preprocessor.feature_names_out_)
                    ),
                    "final": int(len(preprocessor.feature_names_out_)),
                },
                "training": {
                    **_dataset_summary(train, train_tmm),
                    "factor_outliers": train_outliers,
                },
                partition.evaluation_role: {
                    **_dataset_summary(evaluation, evaluation_tmm),
                    "factor_outliers": evaluation_outliers,
                },
                "tmm_fit": {
                    "reference_sample_id": preprocessor.tmm_.reference_sample_id_,
                    "reference_library_size": preprocessor.tmm_.reference_library_size_,
                    "training_geometric_mean_calibration": preprocessor.tmm_.calibration_,
                    "geometric_mean_training_factor": float(
                        np.exp(np.mean(np.log(preprocessor.tmm_.training_factors_)))
                    ),
                },
                "output_checks": {
                    "training_shape": list(transformed_train.shape),
                    f"{partition.evaluation_role}_shape": list(transformed_evaluation.shape),
                    "feature_order_identical": list(transformed_train.columns)
                    == list(transformed_evaluation.columns),
                    "training_nan": int(transformed_train.isna().sum().sum()),
                    f"{partition.evaluation_role}_nan": int(
                        transformed_evaluation.isna().sum().sum()
                    ),
                    "training_infinite": int(
                        np.isinf(transformed_train.to_numpy(dtype=float)).sum()
                    ),
                    f"{partition.evaluation_role}_infinite": int(
                        np.isinf(transformed_evaluation.to_numpy(dtype=float)).sum()
                    ),
                },
                "runtime_seconds": {
                    "fit": float(fit_seconds),
                    "transform_train_and_evaluation": float(transform_seconds),
                },
                "artifacts": {
                    "state_dir": str((artifacts_root / partition.name).relative_to(repo_root)),
                    "sha256": state_hashes,
                    "ma_plots": plot_paths,
                },
            }
        )
        del transformed_train, transformed_evaluation, filtered_train, filtered_evaluation

    after_hashes = {str(path.relative_to(repo_root)): sha256_file(path) for path in canonical_paths}
    if before_hashes != after_hashes:
        raise RuntimeError("Canonical v1 artifacts changed during Phase 1B; aborting")
    total_runtime = float(time.perf_counter() - run_started)
    report = {
        "generated_at": utc_now(),
        "status": "pass",
        "specification_id": pre_config["specification_id"],
        "primary_methodological_decision": {
            "pipeline": "training CPM filter -> frozen-reference TMM -> log2-CPM prior -> exact-zero-variance -> estimator-specific scaling",
            "cpm_threshold": float(pre_config["expression_filter"]["cpm_threshold"]),
            "minimum_training_prevalence": float(
                pre_config["expression_filter"]["min_training_prevalence"]
            ),
            "selection_basis": pre_config["expression_filter"]["selection_basis"],
            "target_used": False,
            "global_normalized_modeling_matrix_created": False,
        },
        "canonical_v1_immutable": before_hashes == after_hashes,
        "canonical_v1_sha256_before": before_hashes,
        "canonical_v1_sha256_after": after_hashes,
        "partitions": partition_reports,
        "total_runtime_seconds": total_runtime,
    }
    qc_path = resolve_path(repo_root, pre_config["qc_report"])
    write_json_atomic(qc_path, report)

    config_resolved = Path(config_path).resolve()
    source_paths = sorted((repo_root / "src" / "data").glob("*.py"))
    provenance = {
        "generated_at": utc_now(),
        "project": config["project_id"],
        "cohort_version": config["cohort_version"],
        "cohort_status": config["cohort_status"],
        "phase": "1B RNA-seq preprocessing",
        "specification": pre_config,
        "fit_semantics": "all patient-distribution-dependent operations fit on training only",
        "target_used": False,
        "global_normalized_modeling_matrix_created": False,
        "configuration_sha256": sha256_file(config_resolved),
        "canonical_v1_sha256": after_hashes,
        "qc_report": str(qc_path.relative_to(repo_root)),
        "qc_report_sha256": sha256_file(qc_path),
        "code": {
            **git_state(repo_root),
            "source_files_sha256": {
                str(path.relative_to(repo_root)): sha256_file(path) for path in source_paths
            },
        },
        "dependencies": dependency_versions(),
        "references": {
            "gdc_mrna": "https://docs.gdc.cancer.gov/Data/Bioinformatics_Pipelines/Expression_mRNA_Pipeline/",
            "edgeR_user_guide": "https://bioconductor.org/packages/release/bioc/vignettes/edgeR/inst/doc/edgeRUsersGuide.pdf",
            "tmm": "https://doi.org/10.1186/gb-2010-11-3-r25",
            "sklearn_leakage": "https://scikit-learn.org/stable/common_pitfalls.html#data-leakage",
        },
    }
    provenance_path = resolve_path(repo_root, pre_config["provenance_report"])
    write_json_atomic(provenance_path, provenance)
    report["provenance_report"] = str(provenance_path.relative_to(repo_root))
    report["qc_report"] = str(qc_path.relative_to(repo_root))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    report = run_phase1b(args.config)
    LOGGER.info(
        "Phase 1B complete: %s partitions in %.2f seconds",
        len(report["partitions"]),
        report["total_runtime_seconds"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
