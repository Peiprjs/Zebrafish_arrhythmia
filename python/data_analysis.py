"""
Heart Rate Variability (HRV) Analysis for Zebrafish MUSCLEMOTION Data
=====================================================================

Reads MUSCLEMOTION output folders from MM_Results/, extracts contraction
peaks, computes inter-beat intervals, instantaneous and global HRV metrics,
and estimates a heuristic arrhythmia risk score for each sample.

Usage:
    python data_analysis.py [--results_dir PATH] [--output_dir PATH]

GUI:
    streamlit run gui_app.py
"""

import argparse
import os
import re
import sys

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.cluster.vq import kmeans2
from scipy.optimize import minimize
from scipy.signal import find_peaks
from scipy.stats import linregress, norm, t as student_t, ttest_rel


SUMMARY_COLUMNS = [
    "sample", "exposure", "concentration", "well", "fish",
    "n_peaks", "mean_ibi_ms", "sdnn_ms", "rmssd_ms",
    "cv_ibi", "pnn50", "mean_hr_bpm",
    "arrhythmia_risk_score", "arrhythmia_probability",
    "arrhythmia_score_cv", "arrhythmia_score_rmssd", "arrhythmia_score_outlier",
    "arrhythmia_outlier_fraction",
    "arrhythmia_data_sufficient", "arrhythmia_quality_flag",
    "arrhythmia_ibi_count", "arrhythmia_threshold",
    "paired_ttest_pvalue_vs_control0_mean_ibi",
    "Arrhymia",
    "snr", "baseline_drift",
]
ROLLING_RMSSD_WINDOW = 5
ARRHYTHMIA_DEFAULT_THRESHOLD = 0.5
ARRHYTHMIA_MIN_IBI_FOR_SCORE = 3
ARRHYTHMIA_MIN_IBI_FOR_CONFIDENT_DECISION = 6
CONTROL_CONCENTRATION = "0"
PEAK_PROMINENCE_OUTLIER_RATIO_THRESHOLD = 2.0
PEAK_PROMINENCE_TRIM_PERCENTILES = (1.0, 99.0)
UNSUPERVISED_FEATURE_COLUMNS = [
    "n_peaks",
    "mean_ibi_ms",
    "sdnn_ms",
    "rmssd_ms",
    "cv_ibi",
    "pnn50",
    "mean_hr_bpm",
    "arrhythmia_risk_score",
    "snr",
    "baseline_drift",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_folder_name(folder_name):
    """Extract metadata from a MUSCLEMOTION results folder name.

    Expected pattern: {Exposure}_{Concentration}_{Well}.{Fish}-Contr-Results
    Example: Phe_50_1.1-Contr-Results -> ('Phe', '50', '1', '1')

    Returns a dict with keys: exposure, concentration, well, fish.
    """
    match = re.match(
        r"^(.+?)_(\d+)_(\d+)\.(\d+)-Contr-Results$", folder_name
    )
    if not match:
        return None
    return {
        "exposure": match.group(1),
        "concentration": match.group(2),
        "well": match.group(3),
        "fish": match.group(4),
    }


def load_tsv(filepath):
    """Load a two-column, tab-separated MUSCLEMOTION output file.

    Returns (time_ms, values) as NumPy arrays.
    """
    data = np.loadtxt(filepath, delimiter="\t")
    return data[:, 0], data[:, 1]


def list_result_folders(results_dir):
    """List MUSCLEMOTION result folders in sorted order."""
    if not os.path.isdir(results_dir):
        return []

    return sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.endswith("-Contr-Results")
    )


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def _rolling_median(values, window_samples):
    """Return centered rolling median with edge support."""
    series = pd.Series(np.asarray(values, dtype=float))
    return (
        series
        .rolling(window=max(1, int(window_samples)), center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def compute_signal_quality(time_ms, signal, baseline_window_ms=800.0, smoothing_window_ms=50.0):
    """Compute Signal-to-Noise Ratio (SNR) and baseline drift.

    Parameters
    ----------
    time_ms : array-like
        Time axis in milliseconds.
    signal : array-like
        Contraction signal values.
    baseline_window_ms : float
        Window size for rolling-median baseline estimation.
    smoothing_window_ms : float
        Window size for rolling-mean signal smoothing to estimate noise.

    Returns
    -------
    dict with keys:
        snr : float
            Estimated Signal-to-Noise Ratio in dB.
        baseline_drift : float
            Maximum variation of the estimated baseline.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)

    if len(time_ms) < 3 or len(signal) < 3:
        return {"snr": np.nan, "baseline_drift": np.nan}

    dt = np.median(np.diff(time_ms))
    if not np.isfinite(dt) or dt <= 0:
        return {"snr": np.nan, "baseline_drift": np.nan}

    # Baseline drift
    baseline_samples = max(3, int(baseline_window_ms / dt))
    if baseline_samples % 2 == 0:
        baseline_samples += 1

    baseline = _rolling_median(signal, baseline_samples)
    baseline_drift = float(np.max(baseline) - np.min(baseline))

    # SNR calculation
    # Signal power is variance of original signal
    signal_power = np.var(signal)

    # Noise is original signal minus smoothed version
    smooth_samples = max(3, int(smoothing_window_ms / dt))
    if smooth_samples % 2 == 0:
        smooth_samples += 1

    smoothed_signal = (
        pd.Series(signal)
        .rolling(window=smooth_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    noise = signal - smoothed_signal
    noise_power = np.var(noise)

    if noise_power > 0:
        snr = float(10 * np.log10(signal_power / noise_power))
    else:
        snr = np.nan

    return {"snr": snr, "baseline_drift": baseline_drift}


def detect_peaks(
    time_ms,
    signal,
    prominence_factor=0.3,
    distance_ms=50.0,
    baseline_window_ms=800.0,
    sawtooth_group_distance_ms=500.0,
):
    """Detect contraction peaks in the signal and handle sawtooth profiles.

    Parameters
    ----------
    time_ms : array-like
        Time axis in milliseconds.
    signal : array-like
        Contraction signal values.
    prominence_factor : float
        Minimum peak prominence as a fraction of the detrended signal range.
    distance_ms : float
        Minimum distance between peaks in milliseconds (used for initial peak picking).
    baseline_window_ms : float
        Window size for rolling-median baseline estimation used to remove slow
        baseline drift before peak picking.
    sawtooth_group_distance_ms : float
        Maximum distance between consecutive initial peaks to group them into a sawtooth event.

    Returns
    -------
    peak_indices : ndarray
        Indices of detected final peaks.
    peak_times : ndarray
        Times (ms) of detected final peaks.
    sawtooth_peak : ndarray
        Boolean array indicating whether each final peak was part of a sawtooth event.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if len(time_ms) < 3 or len(signal) < 3:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    dt = np.median(np.diff(time_ms))
    if not np.isfinite(dt) or dt <= 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    distance_samples = max(1, int(distance_ms / dt))
    baseline_samples = max(3, int(baseline_window_ms / dt))
    if baseline_samples % 2 == 0:
        baseline_samples += 1

    baseline = _rolling_median(signal, baseline_samples)
    detrended_signal = signal - baseline
    detrended_range = np.max(detrended_signal) - np.min(detrended_signal)
    if not np.isfinite(detrended_range) or detrended_range <= 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    lower_q, upper_q = PEAK_PROMINENCE_TRIM_PERCENTILES
    detrended_trimmed_range = np.percentile(detrended_signal, upper_q) - np.percentile(detrended_signal, lower_q)
    if not np.isfinite(detrended_trimmed_range) or detrended_trimmed_range <= 0:
        detrended_trimmed_range = detrended_range

    outlier_ratio = detrended_range / max(detrended_trimmed_range, np.finfo(float).eps)
    if outlier_ratio > PEAK_PROMINENCE_OUTLIER_RATIO_THRESHOLD:
        spread_for_prominence = detrended_trimmed_range
    else:
        spread_for_prominence = detrended_range

    prominence = max(np.finfo(float).eps, prominence_factor * spread_for_prominence)

    peak_indices, _ = find_peaks(
        detrended_signal, prominence=prominence, distance=distance_samples
    )

    # Fallback to raw-signal detection when detrending is overly conservative.
    if len(peak_indices) < 2:
        raw_range = np.max(signal) - np.min(signal)
        if np.isfinite(raw_range) and raw_range > 0:
            raw_prominence = max(np.finfo(float).eps, prominence_factor * raw_range)
            raw_peak_indices, _ = find_peaks(
                signal,
                prominence=raw_prominence,
                distance=distance_samples,
            )
            if len(raw_peak_indices) > len(peak_indices):
                peak_indices = raw_peak_indices

    if len(peak_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=bool)

    # Group sawtooth peaks
    # Use a data-driven constraint on the grouping window so we do not
    # merge distinct heartbeats when their true inter-beat interval is
    # shorter than the default sawtooth_group_distance_ms.
    effective_group_distance_ms = sawtooth_group_distance_ms
    max_group_span_ms = None
    if len(peak_indices) > 1:
        peak_times_ms = time_ms[peak_indices]
        ibis = np.diff(peak_times_ms)
        finite_ibis = ibis[np.isfinite(ibis) & (ibis > 0)]
        if finite_ibis.size > 0:
            median_ibi = np.median(finite_ibis)
            # Limit the grouping distance so it cannot exceed a fraction of
            # the observed median inter-peak interval.
            dynamic_window = 0.5 * median_ibi
            if dynamic_window > 0:
                effective_group_distance_ms = min(sawtooth_group_distance_ms, dynamic_window)
            # Also prevent any single group from spanning much more than a
            # typical beat interval.
            max_group_span_ms = median_ibi

    final_peak_indices = []
    sawtooth_flags = []

    current_group = [peak_indices[0]]

    for i in range(1, len(peak_indices)):
        prev_idx = current_group[-1]
        curr_idx = peak_indices[i]

        time_diff_prev = time_ms[curr_idx] - time_ms[prev_idx]
        time_diff_group_start = time_ms[curr_idx] - time_ms[current_group[0]]

        # If the time difference between the current peak and the last peak in the group
        # is within the (possibly tightened) threshold, and the overall group span is
        # not excessive, add it to the group.
        if (
            time_diff_prev <= effective_group_distance_ms
            and (max_group_span_ms is None or time_diff_group_start <= max_group_span_ms)
        ):
            current_group.append(curr_idx)
        else:
            # Process the completed group
            if len(current_group) > 1:
                # Find the index of the peak with maximum amplitude in the original signal
                amplitudes = signal[current_group]
                max_amp_index_in_group = np.argmax(amplitudes)
                final_peak_indices.append(current_group[max_amp_index_in_group])
                sawtooth_flags.append(True)
            else:
                final_peak_indices.append(current_group[0])
                sawtooth_flags.append(False)

            # Start a new group
            current_group = [curr_idx]

    # Process the last group
    if current_group:
        if len(current_group) > 1:
            amplitudes = signal[current_group]
            max_amp_index_in_group = np.argmax(amplitudes)
            final_peak_indices.append(current_group[max_amp_index_in_group])
            sawtooth_flags.append(True)
        else:
            final_peak_indices.append(current_group[0])
            sawtooth_flags.append(False)

    final_peak_indices = np.array(final_peak_indices, dtype=int)
    return final_peak_indices, time_ms[final_peak_indices], np.array(sawtooth_flags, dtype=bool)


# ---------------------------------------------------------------------------
# HRV computation
# ---------------------------------------------------------------------------

def compute_ibi(peak_times_ms):
    """Compute inter-beat intervals from peak times.

    Returns IBI array in milliseconds.
    """
    return np.diff(peak_times_ms)


def compute_hrv_metrics(ibi_ms):
    """Compute heart-rate variability metrics from inter-beat intervals.

    Parameters
    ----------
    ibi_ms : array-like
        Inter-beat intervals in milliseconds.

    Returns
    -------
    dict with keys:
        mean_ibi_ms    – mean IBI (ms)
        sdnn_ms        – standard deviation of IBI (global HRV, ms)
        rmssd_ms       – root mean square of successive differences (instantaneous HRV, ms)
        cv_ibi         – coefficient of variation of IBI (unitless)
        pnn50          – proportion of successive IBI differences > 50 ms (%)
        mean_hr_bpm    – mean heart rate (beats per minute)
    """
    ibi = np.asarray(ibi_ms, dtype=float)
    successive_diff = np.diff(ibi)

    mean_ibi = np.mean(ibi)
    sdnn = np.std(ibi, ddof=1) if len(ibi) > 1 else 0.0
    rmssd = np.sqrt(np.mean(successive_diff ** 2)) if len(successive_diff) > 0 else 0.0
    cv = (sdnn / mean_ibi) if mean_ibi > 0 else 0.0
    pnn50 = (
        100.0 * np.sum(np.abs(successive_diff) > 50) / len(successive_diff)
        if len(successive_diff) > 0
        else 0.0
    )
    mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0

    return {
        "mean_ibi_ms": mean_ibi,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "cv_ibi": cv,
        "pnn50": pnn50,
        "mean_hr_bpm": mean_hr,
    }


def compute_rolling_rmssd(ibi_ms, window=ROLLING_RMSSD_WINDOW):
    """Compute rolling RMSSD values over IBI windows."""
    if window < 2:
        raise ValueError("window must be >= 2")

    ibi = np.asarray(ibi_ms, dtype=float)
    if len(ibi) < window:
        return np.array([], dtype=float)

    rolling = np.empty(len(ibi) - window + 1, dtype=float)
    for i in range(window - 1, len(ibi)):
        segment = ibi[i - window + 1: i + 1]
        diffs = np.diff(segment)
        rolling[i - window + 1] = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 0 else 0.0

    return rolling


# ---------------------------------------------------------------------------
# Arrhythmia probability
# ---------------------------------------------------------------------------

def _logistic_score(value, midpoint, steepness):
    """Map an indicator to [0, 1] using a logistic transform."""
    return 1.0 / (1.0 + np.exp(-steepness * (value - midpoint)))


def compute_arrhythmia_risk(ibi_ms):
    """Compute a heuristic arrhythmia risk score and supporting diagnostics.

    This is a deterministic irregularity score derived from IBI statistics.
    It is not a supervised, calibrated probability model.
    """
    ibi = np.asarray(ibi_ms, dtype=float)
    n_ibi = int(len(ibi))
    empty_result = {
        "arrhythmia_risk_score": np.nan,
        "arrhythmia_probability": np.nan,  # compatibility alias
        "arrhythmia_score_cv": np.nan,
        "arrhythmia_score_rmssd": np.nan,
        "arrhythmia_score_outlier": np.nan,
        "arrhythmia_outlier_fraction": np.nan,
        "arrhythmia_data_sufficient": False,
        "arrhythmia_quality_flag": "insufficient_ibi",
        "arrhythmia_ibi_count": n_ibi,
    }
    if n_ibi < ARRHYTHMIA_MIN_IBI_FOR_SCORE:
        return empty_result

    mean_ibi = np.mean(ibi)
    if mean_ibi <= 0:
        invalid_result = dict(empty_result)
        invalid_result["arrhythmia_quality_flag"] = "invalid_ibi"
        return invalid_result

    # Indicator 1: coefficient of variation
    cv = np.std(ibi, ddof=1) / mean_ibi
    score_cv = _logistic_score(cv, 0.15, 30)

    # Indicator 2: RMSSD relative to mean IBI
    successive_diff = np.diff(ibi)
    rmssd_rel = np.sqrt(np.mean(successive_diff ** 2)) / mean_ibi
    score_rmssd = _logistic_score(rmssd_rel, 0.15, 30)

    # Indicator 3: fraction of outlier IBIs (>30 % from median)
    median_ibi = np.median(ibi)
    outlier_frac = np.mean(
        np.abs(ibi - median_ibi) / (median_ibi if median_ibi > 0 else 1.0) > 0.30
    )
    score_outlier = _logistic_score(outlier_frac, 0.15, 20)
    risk_score = float(np.mean([score_cv, score_rmssd, score_outlier]))

    quality_flag = (
        "low_ibi_count"
        if n_ibi < ARRHYTHMIA_MIN_IBI_FOR_CONFIDENT_DECISION
        else "ok"
    )
    return {
        "arrhythmia_risk_score": risk_score,
        "arrhythmia_probability": risk_score,  # compatibility alias
        "arrhythmia_score_cv": float(score_cv),
        "arrhythmia_score_rmssd": float(score_rmssd),
        "arrhythmia_score_outlier": float(score_outlier),
        "arrhythmia_outlier_fraction": float(outlier_frac),
        "arrhythmia_data_sufficient": True,
        "arrhythmia_quality_flag": quality_flag,
        "arrhythmia_ibi_count": n_ibi,
    }


def arrhythmia_probability(ibi_ms):
    """Compatibility wrapper for the heuristic arrhythmia risk score."""
    return compute_arrhythmia_risk(ibi_ms)["arrhythmia_probability"]


def arrhythmia_decision(risk_score, threshold, data_sufficient):
    """Return a conservative arrhythmia decision from risk score + threshold."""
    if not data_sufficient or np.isnan(risk_score):
        return False
    return bool(risk_score > threshold)


# ---------------------------------------------------------------------------
# Control comparison statistics
# ---------------------------------------------------------------------------

def _resample_series(values, target_len):
    """Resample a 1D series to *target_len* points on a normalized index."""
    series = np.asarray(values, dtype=float)
    if target_len < 2 or len(series) < 2:
        return np.array([], dtype=float)
    if len(series) == target_len:
        return series

    x_old = np.linspace(0.0, 1.0, len(series))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.interp(x_new, x_old, series)


def paired_ttest_pvalue(values, reference_values, paired=True):
    """Return paired t-test p-value for two aligned vectors or vs a reference mean.

    If paired=True, values and reference_values are aligned and must have enough overlap.
    If paired=False, reference_values' mean is used as a constant reference for values.
    """
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    if paired:
        if len(values) < 2 or len(reference_values) < 2:
            return np.nan
        n = min(len(values), len(reference_values))
        values = values[:n]
        reference = reference_values[:n]
    else:
        if len(values) < 2 or len(reference_values) < 1:
            return np.nan
        reference_mean = float(np.mean(reference_values))
        reference = np.full(values.shape, reference_mean, dtype=float)

    differences = values - reference

    if np.allclose(differences, 0.0):
        return 1.0
    if np.allclose(differences, differences[0]):
        return 0.0

    _, p_value = ttest_rel(values, reference, nan_policy="omit")
    return float(p_value) if np.isfinite(p_value) else np.nan


def add_control_pvalues(full_df):
    """Add per-sample paired t-test p-value vs mean concentration-0 controls.

    Controls are matched within each exposure group.
    """
    df = full_df.copy()
    pvalues = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Adding control p-values"):
        sample_ibi = np.asarray(row["ibi_values_ms"], dtype=float)
        if len(sample_ibi) < 2:
            pvalues.append(np.nan)
            continue

        same_exposure = df[df["exposure"] == row["exposure"]]
        controls = same_exposure[
            same_exposure["concentration"].astype(str) == CONTROL_CONCENTRATION
        ]
        controls = controls[controls["sample"] != row["sample"]]
        control_ibi_list = [
            np.asarray(values, dtype=float)
            for values in controls["ibi_values_ms"].tolist()
            if len(values) >= 2
        ]
        if not control_ibi_list:
            pvalues.append(np.nan)
            continue

        aligned_controls = [
            _resample_series(control_ibi, len(sample_ibi))
            for control_ibi in control_ibi_list
        ]
        aligned_controls = [arr for arr in aligned_controls if len(arr) == len(sample_ibi)]
        if not aligned_controls:
            pvalues.append(np.nan)
            continue

        control_mean = np.mean(np.vstack(aligned_controls), axis=0)
        pvalues.append(paired_ttest_pvalue(sample_ibi, control_mean, paired=True))

    df["paired_ttest_pvalue_vs_control0_mean_ibi"] = pvalues
    return df


# ---------------------------------------------------------------------------
# Additional statistical tests
# ---------------------------------------------------------------------------

def _prepare_regression_predictors(summary_df):
    """Prepare regression-ready table and design matrix from summary outputs."""
    df = summary_df.copy()
    df["concentration_numeric"] = pd.to_numeric(df["concentration"], errors="coerce")
    df = df.dropna(subset=["concentration_numeric", "arrhythmia_risk_score", "Arrhymia"])
    if df.empty:
        return df, pd.DataFrame()

    predictors = pd.DataFrame(
        {"concentration_numeric": df["concentration_numeric"].astype(float).to_numpy()},
        index=df.index,
    )
    exposure_dummies = pd.get_dummies(
        df["exposure"].astype(str),
        prefix="exposure",
        drop_first=True,
        dtype=float,
    )
    predictors = pd.concat([predictors, exposure_dummies], axis=1)
    return df, predictors


def _fit_linear_regression(y, predictors):
    """Fit OLS and return coefficient table + model summary."""
    X_no_intercept = np.asarray(predictors, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), X_no_intercept])
    terms = ["intercept"] + list(predictors.columns)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat
    n_samples, n_params = X.shape
    dof = n_samples - n_params

    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = np.nan if tss <= 0 else 1.0 - (rss / tss)

    if dof > 0:
        xtx_inv = np.linalg.pinv(X.T @ X)
        sigma2 = rss / dof
        covariance = sigma2 * xtx_inv
        std_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(std_error > 0, beta / std_error, np.nan)
        p_value = 2.0 * student_t.sf(np.abs(t_stat), df=dof)
    else:
        std_error = np.full(len(beta), np.nan)
        t_stat = np.full(len(beta), np.nan)
        p_value = np.full(len(beta), np.nan)

    coef_df = pd.DataFrame(
        {
            "term": terms,
            "coefficient": beta,
            "std_error": std_error,
            "t_statistic": t_stat,
            "p_value": p_value,
        }
    )
    summary_df = pd.DataFrame(
        [
            {
                "model": "linear_regression",
                "n_samples": n_samples,
                "n_parameters": n_params,
                "dof": dof,
                "r_squared": r_squared,
                "rss": rss,
            }
        ]
    )
    return coef_df, summary_df


def _fit_logistic_regression(y, predictors):
    """Fit logistic regression via MLE and return coefficients + summary."""
    y = np.asarray(y, dtype=float)
    X_no_intercept = np.asarray(predictors, dtype=float)
    X = np.column_stack([np.ones(len(y)), X_no_intercept])
    terms = ["intercept"] + list(predictors.columns)
    n_samples, n_params = X.shape

    if len(np.unique(y)) < 2:
        empty_coef = pd.DataFrame(
            {
                "term": terms,
                "coefficient": np.nan,
                "std_error": np.nan,
                "z_statistic": np.nan,
                "p_value": np.nan,
                "odds_ratio": np.nan,
            }
        )
        summary = pd.DataFrame(
            [
                {
                    "model": "logistic_regression",
                    "n_samples": n_samples,
                    "n_parameters": n_params,
                    "converged": False,
                    "status": "single_class_outcome",
                    "accuracy_at_0_5": np.nan,
                    "mcfadden_pseudo_r2": np.nan,
                    "log_likelihood": np.nan,
                }
            ]
        )
        return empty_coef, summary

    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _negative_log_likelihood(beta):
        p = np.clip(_sigmoid(X @ beta), 1e-9, 1.0 - 1e-9)
        return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    result = minimize(
        _negative_log_likelihood,
        x0=np.zeros(n_params, dtype=float),
        method="BFGS",
    )

    beta = result.x
    covariance = np.asarray(result.hess_inv) if hasattr(result, "hess_inv") else np.full((n_params, n_params), np.nan)
    if covariance.shape != (n_params, n_params):
        covariance = np.full((n_params, n_params), np.nan)

    std_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z_stat = np.where(std_error > 0, beta / std_error, np.nan)
    p_value = 2.0 * norm.sf(np.abs(z_stat))
    odds_ratio = np.exp(beta)

    p_hat = _sigmoid(X @ beta)
    accuracy = float(np.mean((p_hat >= 0.5).astype(float) == y))
    ll_model = -float(_negative_log_likelihood(beta))
    p_null = float(np.clip(np.mean(y), 1e-9, 1.0 - 1e-9))
    ll_null = float(np.sum(y * np.log(p_null) + (1.0 - y) * np.log(1.0 - p_null)))
    pseudo_r2 = np.nan if ll_null == 0 else 1.0 - (ll_model / ll_null)

    coef_df = pd.DataFrame(
        {
            "term": terms,
            "coefficient": beta,
            "std_error": std_error,
            "z_statistic": z_stat,
            "p_value": p_value,
            "odds_ratio": odds_ratio,
        }
    )
    summary_df = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "n_samples": n_samples,
                "n_parameters": n_params,
                "converged": bool(result.success),
                "status": result.message,
                "accuracy_at_0_5": accuracy,
                "mcfadden_pseudo_r2": pseudo_r2,
                "log_likelihood": ll_model,
            }
        ]
    )
    return coef_df, summary_df


def run_additional_statistical_tests(summary_df, output_dir):
    """Run additional statistical tests and save regression outputs."""
    prepared_df, predictors = _prepare_regression_predictors(summary_df)
    if prepared_df.empty or predictors.empty:
        return {}

    linear_coef_df, linear_summary_df = _fit_linear_regression(
        prepared_df["arrhythmia_risk_score"].to_numpy(dtype=float),
        predictors,
    )
    logistic_coef_df, logistic_summary_df = _fit_logistic_regression(
        prepared_df["Arrhymia"].astype(float).to_numpy(),
        predictors,
    )

    trend = linregress(
        prepared_df["concentration_numeric"].to_numpy(dtype=float),
        prepared_df["arrhythmia_risk_score"].to_numpy(dtype=float),
    )
    trend_df = pd.DataFrame(
        [
            {
                "model": "simple_linear_trend",
                "slope": trend.slope,
                "intercept": trend.intercept,
                "r_value": trend.rvalue,
                "p_value": trend.pvalue,
                "std_err": trend.stderr,
            }
        ]
    )

    linear_coef_path = os.path.join(output_dir, "linear_regression_coefficients.csv")
    linear_summary_path = os.path.join(output_dir, "linear_regression_summary.csv")
    logistic_coef_path = os.path.join(output_dir, "logistic_regression_coefficients.csv")
    logistic_summary_path = os.path.join(output_dir, "logistic_regression_summary.csv")
    trend_path = os.path.join(output_dir, "linear_trend_summary.csv")

    linear_coef_df.to_csv(linear_coef_path, index=False)
    linear_summary_df.to_csv(linear_summary_path, index=False)
    logistic_coef_df.to_csv(logistic_coef_path, index=False)
    logistic_summary_df.to_csv(logistic_summary_path, index=False)
    trend_df.to_csv(trend_path, index=False)

    return {
        "linear_regression_coefficients": linear_coef_path,
        "linear_regression_summary": linear_summary_path,
        "logistic_regression_coefficients": logistic_coef_path,
        "logistic_regression_summary": logistic_summary_path,
        "linear_trend_summary": trend_path,
    }


# ---------------------------------------------------------------------------
# Unsupervised learning models
# ---------------------------------------------------------------------------

def _standardize_with_median_imputation(feature_df):
    """Median-impute then standardize numeric features."""
    medians = feature_df.median(axis=0, numeric_only=True)
    filled = feature_df.fillna(medians)
    means = filled.mean(axis=0, numeric_only=True)
    stds = filled.std(axis=0, ddof=0, numeric_only=True).replace(0.0, 1.0)
    standardized = (filled - means) / stds
    return standardized, medians, means, stds


def _compute_pca_from_standardized(X, n_components=2):
    """Compute PCA scores/loadings from standardized feature matrix."""
    if X.shape[0] < 2 or X.shape[1] < 1:
        return np.empty((X.shape[0], 0)), np.empty((0, X.shape[1])), np.array([])

    max_components = min(n_components, X.shape[1], X.shape[0])
    centered = X - np.mean(X, axis=0)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:max_components]
    scores = centered @ components.T

    if X.shape[0] > 1:
        variance = (singular_values ** 2) / (X.shape[0] - 1)
    else:
        variance = np.zeros_like(singular_values)
    total_variance = np.sum(variance)
    explained_ratio = (
        variance[:max_components] / total_variance
        if total_variance > 0
        else np.zeros(max_components, dtype=float)
    )
    return scores, components, explained_ratio


def run_unsupervised_models(summary_df, output_dir):
    """Run unsupervised models (KMeans, hierarchical clustering, PCA)."""
    available_features = [
        col for col in UNSUPERVISED_FEATURE_COLUMNS
        if col in summary_df.columns
    ]
    if not available_features:
        return {}

    feature_df = summary_df[available_features].apply(pd.to_numeric, errors="coerce")
    standardized_df, _, _, _ = _standardize_with_median_imputation(feature_df)
    X = standardized_df.to_numpy(dtype=float)
    n_samples = X.shape[0]
    if n_samples < 2:
        return {}

    assignments_df = summary_df[
        ["sample", "exposure", "concentration", "Arrhymia", "arrhythmia_risk_score"]
    ].copy()
    assignments_df["concentration"] = assignments_df["concentration"].astype(str)

    cluster_columns = []
    centers_rows = []
    np.random.seed(42)
    for k in (2, 3):
        if n_samples < k:
            continue
        try:
            centers, labels = kmeans2(X, k, minit="++", iter=100)
        except Exception:
            continue

        col = f"cluster_kmeans_k{k}"
        assignments_df[col] = labels + 1
        cluster_columns.append(col)

        for cluster_idx, center in enumerate(centers, start=1):
            centers_rows.append(
                {
                    "model": col,
                    "cluster_id": cluster_idx,
                    **{
                        f"z_{feature}": float(center[i])
                        for i, feature in enumerate(available_features)
                    },
                }
            )

    try:
        hierarchical = linkage(X, method="ward")
        hierarchical_labels = fcluster(hierarchical, t=2, criterion="maxclust")
        hierarchical_col = "cluster_hierarchical_k2"
        assignments_df[hierarchical_col] = hierarchical_labels
        cluster_columns.append(hierarchical_col)
    except Exception:
        pass

    scores, components, explained_ratio = _compute_pca_from_standardized(X, n_components=2)
    if scores.shape[1] >= 1:
        assignments_df["pca_pc1"] = scores[:, 0]
    if scores.shape[1] >= 2:
        assignments_df["pca_pc2"] = scores[:, 1]

    pca_loadings_df = pd.DataFrame({"feature": available_features})
    if components.shape[0] >= 1:
        pca_loadings_df["pc1_loading"] = components[0, :]
    if components.shape[0] >= 2:
        pca_loadings_df["pc2_loading"] = components[1, :]

    pca_variance_df = pd.DataFrame(
        [
            {
                "component": f"PC{i + 1}",
                "explained_variance_ratio": explained_ratio[i] if i < len(explained_ratio) else np.nan,
            }
            for i in range(max(2, len(explained_ratio)))
        ]
    )

    cluster_summary_rows = []
    for cluster_col in cluster_columns:
        for cluster_id, cluster_group in assignments_df.groupby(cluster_col):
            cluster_summary_rows.append(
                {
                    "model": cluster_col,
                    "cluster_id": int(cluster_id),
                    "n_samples": int(len(cluster_group)),
                    "arrhythmia_rate": float(cluster_group["Arrhymia"].astype(float).mean()),
                    "mean_arrhythmia_risk_score": float(cluster_group["arrhythmia_risk_score"].mean()),
                }
            )
    cluster_summary_df = pd.DataFrame(cluster_summary_rows)
    centers_df = pd.DataFrame(centers_rows)

    assignments_path = os.path.join(output_dir, "unsupervised_assignments.csv")
    cluster_summary_path = os.path.join(output_dir, "unsupervised_cluster_summary.csv")
    pca_loadings_path = os.path.join(output_dir, "unsupervised_pca_loadings.csv")
    pca_variance_path = os.path.join(output_dir, "unsupervised_pca_variance.csv")
    cluster_centers_path = os.path.join(output_dir, "unsupervised_kmeans_centers.csv")

    assignments_df.to_csv(assignments_path, index=False)
    cluster_summary_df.to_csv(cluster_summary_path, index=False)
    pca_loadings_df.to_csv(pca_loadings_path, index=False)
    pca_variance_df.to_csv(pca_variance_path, index=False)
    centers_df.to_csv(cluster_centers_path, index=False)

    return {
        "unsupervised_assignments": assignments_path,
        "unsupervised_cluster_summary": cluster_summary_path,
        "unsupervised_pca_loadings": pca_loadings_path,
        "unsupervised_pca_variance": pca_variance_path,
        "unsupervised_kmeans_centers": cluster_centers_path,
    }


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_contraction_with_peaks(time_ms, signal, peak_indices, peak_times,
                                sawtooth_peak,
                                ibi_ms, sample_label, output_path):
    """Save a figure showing the contraction signal, detected peaks, and intervals."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Top: contraction + peaks
    ax = axes[0]
    ax.plot(time_ms / 1000, signal, linewidth=0.7, label="Contraction")

    # Separate normal peaks and sawtooth peaks for plotting
    if sawtooth_peak is not None:
        # Robustly coerce to a 1D boolean NumPy array for elementwise operations
        sawtooth_arr = np.asarray(sawtooth_peak, dtype=bool)
        if sawtooth_arr.ndim != 1:
            sawtooth_arr = sawtooth_arr.ravel()

        if sawtooth_arr.size == len(peak_indices):
            normal_mask = ~sawtooth_arr
            sawtooth_mask = sawtooth_arr

            ax.plot(peak_times[normal_mask] / 1000, signal[peak_indices[normal_mask]], "rv", markersize=8, label="Peaks")
            if np.any(sawtooth_mask):
                ax.plot(peak_times[sawtooth_mask] / 1000, signal[peak_indices[sawtooth_mask]], "bv", markersize=8, label="Sawtooth Peaks")
        else:
            # Fallback: length mismatch, plot all peaks as normal
            ax.plot(peak_times / 1000, signal[peak_indices], "rv", markersize=8, label="Peaks")
    else:
        ax.plot(peak_times / 1000, signal[peak_indices], "rv", markersize=8, label="Peaks")
    ax.set_ylabel("Contraction amplitude (a.u.)")
    ax.set_title(f"Contraction with detected peaks - {sample_label}")
    ax.legend(loc="upper right")

    # Bottom: inter-beat intervals
    ax2 = axes[1]
    if len(ibi_ms) > 0:
        ax2.plot(peak_times[1:] / 1000, ibi_ms, "o-", markersize=4)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Inter-beat interval (ms)")
    ax2.set_title("Inter-beat intervals over recording time")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyse_sample_timeseries(
    folder_path,
    arrhythmia_threshold=ARRHYTHMIA_DEFAULT_THRESHOLD,
    verbose=True,
):
    """Analyse a sample folder and return summary + time-series fields.

    Returns a dict or None if folder is invalid.
    """
    folder_name = os.path.basename(folder_path)
    meta = parse_folder_name(folder_name)
    if meta is None:
        return None

    contraction_file = os.path.join(folder_path, "contraction.txt")
    if not os.path.isfile(contraction_file):
        if verbose:
            tqdm.write(f"  [SKIP] No contraction.txt in {folder_name}")
        return None

    time_ms, signal = load_tsv(contraction_file)

    peak_indices, peak_times, sawtooth_peak = detect_peaks(time_ms, signal)
    if len(peak_times) < 5:
        if verbose:
            tqdm.write(f"  [WARN] Fewer than 5 peaks detected in {folder_name}")
        return None

    ibi_ms = compute_ibi(peak_times)
    ibi_time_ms = peak_times[1:]

    hrv = compute_hrv_metrics(ibi_ms)
    arrhythmia = compute_arrhythmia_risk(ibi_ms)
    arr_decision = arrhythmia_decision(
        arrhythmia["arrhythmia_risk_score"],
        arrhythmia_threshold,
        arrhythmia["arrhythmia_data_sufficient"],
    )
    quality = compute_signal_quality(time_ms, signal)
    sample_label = f"{meta['exposure']}_{meta['concentration']}_{meta['well']}.{meta['fish']}"

    instantaneous_hr = np.where(ibi_ms > 0, 60000.0 / ibi_ms, np.nan)
    rolling_rmssd = compute_rolling_rmssd(ibi_ms, window=ROLLING_RMSSD_WINDOW)
    if len(rolling_rmssd) > 0:
        rolling_rmssd_time_ms = ibi_time_ms[ROLLING_RMSSD_WINDOW - 1:]
    else:
        rolling_rmssd_time_ms = np.array([], dtype=float)

    speed_file = os.path.join(folder_path, "speed-of-contraction.txt")
    if os.path.isfile(speed_file):
        speed_time_ms, speed_values = load_tsv(speed_file)
    else:
        speed_time_ms = np.array([], dtype=float)
        speed_values = np.array([], dtype=float)

    return {
        "sample": sample_label,
        **meta,
        "n_peaks": len(peak_times),
        "time_ms": time_ms.tolist(),
        "contraction_values": signal.tolist(),
        "peak_indices": peak_indices.tolist(),
        "peak_times_ms": peak_times.tolist(),
        "sawtooth_peak": sawtooth_peak.tolist(),
        "ibi_time_ms": ibi_time_ms.tolist(),
        "ibi_values_ms": ibi_ms.tolist(),
        "instantaneous_hr_bpm": instantaneous_hr.tolist(),
        "rolling_rmssd_time_ms": rolling_rmssd_time_ms.tolist(),
        "rolling_rmssd_ms": rolling_rmssd.tolist(),
        "speed_time_ms": speed_time_ms.tolist(),
        "speed_values": speed_values.tolist(),
        "has_speed_profile": bool(len(speed_time_ms) > 0),
        **hrv,
        **arrhythmia,
        "arrhythmia_threshold": float(arrhythmia_threshold),
        "paired_ttest_pvalue_vs_control0_mean_ibi": np.nan,
        "Arrhymia": arr_decision,
        **quality,
    }


def analyse_sample(
    folder_path,
    output_dir=None,
    arrhythmia_threshold=ARRHYTHMIA_DEFAULT_THRESHOLD,
):
    """Run full HRV analysis on one MUSCLEMOTION results folder.

    Returns a summary dict or None if folder is not valid.
    """
    sample_data = analyse_sample_timeseries(
        folder_path,
        arrhythmia_threshold=arrhythmia_threshold,
        verbose=True,
    )
    if sample_data is None:
        # Keep summary behavior unchanged, but still export a peak-detection image
        # when a valid sample has too few detected peaks for metric computation.
        if output_dir is not None:
            folder_name = os.path.basename(folder_path)
            meta = parse_folder_name(folder_name)
            contraction_file = os.path.join(folder_path, "contraction.txt")
            if meta is not None and os.path.isfile(contraction_file):
                time_ms, signal = load_tsv(contraction_file)
                peak_indices, peak_times, sawtooth_peak = detect_peaks(time_ms, signal)
                ibi_ms = compute_ibi(peak_times) if len(peak_times) >= 2 else np.array([], dtype=float)
                sample_label = f"{meta['exposure']}_{meta['concentration']}_{meta['well']}.{meta['fish']}"
                os.makedirs(output_dir, exist_ok=True)
                plot_path = os.path.join(output_dir, f"{sample_label}_hrv.png")
                plot_contraction_with_peaks(
                    np.asarray(time_ms, dtype=float),
                    np.asarray(signal, dtype=float),
                    np.asarray(peak_indices, dtype=int),
                    np.asarray(peak_times, dtype=float),
                    np.asarray(sawtooth_peak, dtype=bool),
                    np.asarray(ibi_ms, dtype=float),
                    sample_label,
                    plot_path,
                )
        return None

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{sample_data['sample']}_hrv.png")
        plot_contraction_with_peaks(
            np.asarray(sample_data["time_ms"], dtype=float),
            np.asarray(sample_data["contraction_values"], dtype=float),
            np.asarray(sample_data["peak_indices"], dtype=int),
            np.asarray(sample_data["peak_times_ms"], dtype=float),
            np.asarray(sample_data.get("sawtooth_peak", []), dtype=bool),
            np.asarray(sample_data["ibi_values_ms"], dtype=float),
            sample_data["sample"],
            plot_path,
        )

    summary = {key: sample_data[key] for key in SUMMARY_COLUMNS}
    summary["ibi_values_ms"] = sample_data["ibi_values_ms"]
    return summary


def load_all_sample_timeseries(
    results_dir,
    arrhythmia_threshold=ARRHYTHMIA_DEFAULT_THRESHOLD,
    verbose=False,
):
    """Load detailed per-sample series for all valid MUSCLEMOTION folders."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"results directory not found: {results_dir}")

    folders = list_result_folders(results_dir)
    if not folders:
        raise ValueError(f"No MUSCLEMOTION result folders found in {results_dir}")

    all_results = []
    for folder_name in tqdm(folders, desc="Loading sample timeseries"):
        folder_path = os.path.join(results_dir, folder_name)
        result = analyse_sample_timeseries(
            folder_path,
            arrhythmia_threshold=arrhythmia_threshold,
            verbose=verbose,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        raise ValueError("No valid results produced.")

    return all_results


def run_analysis(
    results_dir,
    output_dir,
    arrhythmia_threshold=ARRHYTHMIA_DEFAULT_THRESHOLD,
):
    """Iterate over all sample folders in *results_dir* and produce a summary."""
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    folders = list_result_folders(results_dir)
    if not folders:
        print(f"No MUSCLEMOTION result folders found in {results_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for folder_name in tqdm(folders, desc="Analysing folders"):
        folder_path = os.path.join(results_dir, folder_name)
        result = analyse_sample(
            folder_path,
            output_dir=output_dir,
            arrhythmia_threshold=arrhythmia_threshold,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No valid results produced.")
        sys.exit(1)

    # Build summary DataFrame (exclude per-beat IBI lists for the CSV)
    full_df = pd.DataFrame(all_results)
    full_df = add_control_pvalues(full_df)
    df = full_df[SUMMARY_COLUMNS]
    csv_path = os.path.join(output_dir, "hrv_summary.csv")
    df.to_csv(csv_path, index=False)
    stats_paths = run_additional_statistical_tests(df, output_dir)
    unsupervised_paths = run_unsupervised_models(df, output_dir)
    print(f"\nSummary saved to {csv_path}")
    for label, path in stats_paths.items():
        print(f"{label} saved to {path}")
    for label, path in unsupervised_paths.items():
        print(f"{label} saved to {path}")
    print(df.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(script_dir, "..", "MM_Results")
    default_output = os.path.join(script_dir, "..", "output")

    parser = argparse.ArgumentParser(
        description="HRV analysis for MUSCLEMOTION zebrafish cardiac data."
    )
    parser.add_argument(
        "--results_dir",
        default=default_results,
        help="Path to MM_Results folder (default: ../MM_Results relative to script)",
    )
    parser.add_argument(
        "--output_dir",
        default=default_output,
        help="Path to store output CSV and plots (default: ../output)",
    )
    parser.add_argument(
        "--arrhythmia_threshold",
        type=float,
        default=ARRHYTHMIA_DEFAULT_THRESHOLD,
        help=(
            "Threshold for Arrhymia decision from heuristic risk score "
            "(range: 0 to 1, default: 0.5)"
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.arrhythmia_threshold <= 1.0:
        parser.error("--arrhythmia_threshold must be between 0 and 1.")

    run_analysis(
        os.path.abspath(args.results_dir),
        os.path.abspath(args.output_dir),
        arrhythmia_threshold=args.arrhythmia_threshold,
    )


if __name__ == "__main__":
    main()
