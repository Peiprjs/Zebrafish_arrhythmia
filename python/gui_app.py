"""Streamlit dashboard for zebrafish contraction and HRV exploration.

Usage:
    streamlit run gui_app.py
"""

import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from functions import paired_ttest_pvalue
from data_analysis import (
    SUMMARY_COLUMNS,
    load_all_sample_timeseries,
)


INTERP_POINTS = 300
MODEL_OUTPUT_FILES = {
    "linear_regression_summary": "linear_regression_summary.csv",
    "linear_regression_coefficients": "linear_regression_coefficients.csv",
    "logistic_regression_summary": "logistic_regression_summary.csv",
    "logistic_regression_coefficients": "logistic_regression_coefficients.csv",
    "linear_trend_summary": "linear_trend_summary.csv",
    "anova_concentration_summary": "anova_concentration_summary.csv",
    "unsupervised_cluster_summary": "unsupervised_cluster_summary.csv",
    "unsupervised_assignments": "unsupervised_assignments.csv",
    "unsupervised_pca_variance": "unsupervised_pca_variance.csv",
    "unsupervised_pca_loadings": "unsupervised_pca_loadings.csv",
}
DEFAULT_EXCLUDED_CASES = [
    "Terf_0_1.2",
    "Phe_100_2.1",
    "Terf_20_1.2",
    "Terf_20_2.3",
]
VARIABLE_GLOSSARY_MARKDOWN = """
- `sample`: sample identifier (`Exposure_Concentration_Well.Fish`).
- `exposure`: treatment family/group (e.g., `Phe`, `Terf`).
- `concentration`: dose level.
- `well`, `fish`: plate well index and fish index.
- `n_peaks`: number of detected contraction peaks.
- `sawtooth_peak`: marks detected peaks that were grouped as sawtooth events.
- `IBI`: inter-beat interval (time between consecutive peaks, ms).
- `mean_ibi_ms`: mean IBI.
- `HRV`: heart-rate variability.
- `sdnn_ms` (SDNN): standard deviation of IBI.
- `rmssd_ms` (RMSSD): root mean square of successive IBI differences.
- `cv_ibi` (CV): coefficient of variation of IBI (`SDNN / mean IBI`).
- `pnn50` (pNN50): percent of successive IBI differences > 50 ms.
- `mean_hr_bpm`: mean heart rate in beats per minute.
- `arrhythmia_risk_score`: heuristic irregularity score (0 to 1; higher = more irregular).
- `arrhythmia_probability`: compatibility alias of `arrhythmia_risk_score`.
- `arrhythmia_score_cv`, `arrhythmia_score_rmssd`, `arrhythmia_score_outlier`: component scores used to build the risk score.
- `arrhythmia_outlier_fraction`: fraction of IBIs treated as outliers.
- `arrhythmia_data_sufficient`: whether enough IBI data were available for scoring.
- `arrhythmia_quality_flag`: scoring quality flag (`ok`, `low_ibi_count`, `insufficient_ibi`).
- `arrhythmia_ibi_count`: number of IBIs used.
- `arrhythmia_threshold`: cutoff used for binary decision.
- `Arrhymia`: boolean decision from risk score and threshold.
- `snr`: Signal-to-Noise Ratio (dB) estimated from the contraction signal.
- `baseline_drift`: maximum variation in the contraction signal's baseline.
- `paired_ttest_pvalue_vs_control0_mean_ibi`: paired t-test p-value vs concentration-0 control mean IBI profile (same exposure).
- `cluster_*`: unsupervised cluster labels from KMeans/hierarchical clustering.
- `pca_pc1`, `pca_pc2`: first two PCA component scores.
"""
TEXT_WRAP_CSS = """
<style>
.stMarkdown p, .stMarkdown li, .stCaption, .stText {
    overflow-wrap: anywhere;
    word-break: break-word;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] .stText {
    overflow-wrap: anywhere;
    word-break: break-word;
}
section[data-testid="stSidebar"] code {
    white-space: pre-wrap;
    word-break: break-word;
}
</style>
"""


def _safe_float_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _group_label(record):
    return f"{record['exposure']}_{record['concentration']}"


def _pretty_metric_label(name):
    label_map = {
        "arrhythmia_risk_score": "Arrhythmia risk score",
        "arrhythmia_probability": "Arrhythmia risk score (alias)",
        "mean_ibi_ms": "Mean inter-beat interval (ms)",
        "sdnn_ms": "Standard deviation of inter-beat interval (ms)",
        "rmssd_ms": "Root mean square of successive interval differences (ms)",
        "cv_ibi": "Coefficient of variation of inter-beat interval",
        "pnn50": "Percentage of successive interval differences greater than 50 ms (%)",
        "mean_hr_bpm": "Mean heart rate (beats per minute)",
        "snr": "Signal-to-Noise Ratio (dB)",
        "baseline_drift": "Baseline drift",
        "contraction_amplitude_mean": "Mean contraction amplitude",
        "contraction_amplitude_range": "Contraction amplitude range",
        "force_of_contraction_mean_au": "Mean force of contraction (a.u.)",
        "transient_rise_time_mean_ms": "Mean transient rise time (ms)",
        "transient_decay_time_mean_ms": "Mean transient decay time (ms)",
        "transient_fwhm_mean_ms": "Mean transient FWHM (ms)",
        "transient_duration_mean_ms": "Mean transient duration (ms)",
    }
    return label_map.get(name, name.replace("_", " "))


def _pretty_group_label(name):
    label_map = {
        "concentration": "dose concentration",
        "exposure": "exposure condition",
    }
    return label_map.get(name, name.replace("_", " "))


def _interpolate_to_relative_grid(time_ms, values, n_points=INTERP_POINTS):
    time_ms = np.asarray(time_ms, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(time_ms) < 2 or len(values) < 2 or len(time_ms) != len(values):
        return None

    duration = time_ms[-1] - time_ms[0]
    if duration <= 0:
        return None

    relative_time = (time_ms - time_ms[0]) / duration
    unique_time, unique_idx = np.unique(relative_time, return_index=True)
    unique_values = values[unique_idx]
    if len(unique_time) < 2:
        return None

    grid = np.linspace(0.0, 1.0, n_points)
    interp_values = np.interp(grid, unique_time, unique_values)
    return grid, interp_values


def _aggregate_group_profiles(records, time_key, value_key, n_points=INTERP_POINTS):
    grouped = {}
    for record in tqdm(records, desc="Aggregating group profiles"):
        interpolated = _interpolate_to_relative_grid(record[time_key], record[value_key], n_points)
        if interpolated is None:
            continue

        _, values = interpolated
        label = _group_label(record)
        grouped.setdefault(label, []).append(values)

    if not grouped:
        return None, {}

    grid_pct = np.linspace(0.0, 100.0, n_points)
    aggregate = {}
    for label, series_list in grouped.items():
        stacked = np.vstack(series_list)
        aggregate[label] = {
            "mean": np.mean(stacked, axis=0),
            "std": np.std(stacked, axis=0),
            "n": stacked.shape[0],
        }

    return grid_pct, aggregate


def _group_significance(grouped_values, labels, group_by, alpha=0.05):
    """Return per-group significance flags and legend text for box-plot annotations."""
    values_list = [np.asarray(values, dtype=float) for values in grouped_values]
    significance_flags = []
    p_values = []

    if group_by == "concentration":
        baseline_index = next((idx for idx, label in enumerate(labels) if str(label) == "0"), None)
        if baseline_index is None:
            return [False] * len(values_list), [np.nan] * len(values_list), "* paired t-test p < 0.05 vs concentration 0 mean"

        baseline_values = values_list[baseline_index]
        for idx, values in enumerate(tqdm(values_list, desc="Calculating group significance")):
            if idx == baseline_index or len(values) < 2 or len(baseline_values) < 2:
                significance_flags.append(False)
                p_values.append(np.nan)
                continue

            p_value = paired_ttest_pvalue(values, baseline_values, paired=False)
            p_values.append(p_value)
            significance_flags.append(bool(np.isfinite(p_value) and p_value < alpha))

        return significance_flags, p_values, "* paired t-test p < 0.05 vs concentration 0 mean"

    for idx, values in enumerate(values_list):
        rest_values = [
            other_values
            for j, other_values in enumerate(values_list)
            if j != idx and len(other_values) > 0
        ]
        if len(values) < 2 or not rest_values:
            significance_flags.append(False)
            p_values.append(np.nan)
            continue

        rest = np.concatenate(rest_values)
        if len(rest) < 2:
            significance_flags.append(False)
            p_values.append(np.nan)
            continue

        p_value = paired_ttest_pvalue(values, rest, paired=False)
        p_values.append(p_value)
        significance_flags.append(bool(np.isfinite(p_value) and p_value < alpha))

    return significance_flags, p_values, "* paired t-test p < 0.05 vs rest mean"


@st.cache_data(show_spinner=False)
def _load_dashboard_data(results_dir):
    records = load_all_sample_timeseries(results_dir, verbose=False)
    summary_df = pd.DataFrame(records)[SUMMARY_COLUMNS].copy()
    return records, summary_df


@st.cache_data(show_spinner=False)
def _load_model_output_tables(output_dir):
    tables = {}
    for key, filename in MODEL_OUTPUT_FILES.items():
        path = os.path.join(output_dir, filename)
        if os.path.isfile(path):
            tables[key] = pd.read_csv(path)
        else:
            tables[key] = None
    return tables


def _plot_fish_profile(record):
    time_ms = np.asarray(record["time_ms"], dtype=float)
    recording_start_ms = float(time_ms[0]) if len(time_ms) > 0 else 0.0
    time_s = (time_ms - recording_start_ms) / 1000.0
    contraction = np.asarray(record["contraction_values"], dtype=float)
    peak_indices = np.asarray(record["peak_indices"], dtype=int)
    sawtooth_peak = np.asarray(record.get("sawtooth_peak", []), dtype=bool)
    peak_times_s = (np.asarray(record["peak_times_ms"], dtype=float) - recording_start_ms) / 1000.0

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(time_s, contraction, linewidth=0.8, label="Contraction signal")
    if len(peak_indices) > 0:
        if sawtooth_peak.size == len(peak_indices):
            normal_mask = ~sawtooth_peak
            sawtooth_mask = sawtooth_peak
            if np.any(normal_mask):
                ax.plot(
                    peak_times_s[normal_mask],
                    contraction[peak_indices[normal_mask]],
                    "rv",
                    markersize=6,
                    label="Detected peaks",
                )
            if np.any(sawtooth_mask):
                ax.plot(
                    peak_times_s[sawtooth_mask],
                    contraction[peak_indices[sawtooth_mask]],
                    "bv",
                    markersize=6,
                    label="Detected sawtooth peaks",
                )
        else:
            ax.plot(peak_times_s, contraction[peak_indices], "rv", markersize=6, label="Detected peaks")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Recording time (s)")
    ax.set_ylabel("Contraction amplitude (a.u.)")
    ax.set_title(f"Cardiac contraction signal with detected peaks - {record['sample']}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def _plot_speed_profile(record):
    speed_time = np.asarray(record["speed_time_ms"], dtype=float)
    speed_values = np.asarray(record["speed_values"], dtype=float)
    if len(speed_time) == 0:
        st.info("No speed-of-contraction profile available for this fish.")
        return

    contraction_time_ms = np.asarray(record["time_ms"], dtype=float)
    recording_start_ms = (
        float(contraction_time_ms[0])
        if len(contraction_time_ms) > 0
        else float(speed_time[0])
    )
    speed_time_s = (speed_time - recording_start_ms) / 1000.0

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(speed_time_s, speed_values, linewidth=0.8, color="tab:orange")
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Recording time (s)")
    ax.set_ylabel("Contraction speed (a.u.)")
    ax.set_title(f"Speed of contraction across recording - {record['sample']}")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def _plot_fish_hrv(record):
    contraction_time_ms = np.asarray(record["time_ms"], dtype=float)
    recording_start_ms = float(contraction_time_ms[0]) if len(contraction_time_ms) > 0 else 0.0
    ibi_time_s = (np.asarray(record["ibi_time_ms"], dtype=float) - recording_start_ms) / 1000.0
    ibi_values = np.asarray(record["ibi_values_ms"], dtype=float)
    rolling_time_s = (
        np.asarray(record["rolling_rmssd_time_ms"], dtype=float) - recording_start_ms
    ) / 1000.0
    rolling_values = np.asarray(record["rolling_rmssd_ms"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)

    if len(ibi_time_s) > 0:
        ax1.plot(ibi_time_s, ibi_values, "o-", markersize=3.5, linewidth=1.0, label="Inter-beat interval")
        ax1.set_xlim(left=0.0)
        ax1.set_xlabel("Recording time (s)")
        ax1.set_ylabel("Inter-beat interval (ms)")
        ax1.set_title("Beat-to-beat inter-beat interval across recording")
        ax1.grid(alpha=0.2)

    if len(rolling_time_s) > 0:
        ax2.plot(rolling_time_s, rolling_values, "o-", markersize=3.5, linewidth=1.0, color="tab:green")
        ax2.set_xlim(left=0.0)
        ax2.set_ylabel("Rolling root mean square of successive interval differences (ms)")
        ax2.set_xlabel("Recording time (s)")
        ax2.set_title(
            "Short-term heart rate variability trend "
            "(rolling root mean square of successive interval differences)"
        )
        ax2.grid(alpha=0.2)
    else:
        ax2.text(
            0.02,
            0.5,
            "Not enough beats to compute rolling root mean square of successive interval differences",
            transform=ax2.transAxes,
        )
        ax2.set_axis_off()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_group_aggregate(grid_pct, aggregate, title, y_label):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label in sorted(aggregate.keys()):
        mean = aggregate[label]["mean"]
        std = aggregate[label]["std"]
        n = aggregate[label]["n"]
        ax.plot(grid_pct, mean, linewidth=1.5, label=f"{label} (n={n})")
        ax.fill_between(grid_pct, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel("Relative progress through recording (%)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def _apply_text_wrap_css():
    """Inject CSS to keep long sidebar/content text from overflowing."""
    st.markdown(TEXT_WRAP_CSS, unsafe_allow_html=True)



def _render_tab_fish(selected_sample, record_by_sample):
    fish_record = record_by_sample[selected_sample]
    _plot_fish_profile(fish_record)
    _plot_fish_hrv(fish_record)

def _render_tab_dose(filtered_records):
    grid_pct, aggregate = _aggregate_group_profiles(
        filtered_records,
        time_key="time_ms",
        value_key="contraction_values",
    )
    if not aggregate:
        st.info("Not enough data to compute dose-level contraction profiles.")
    else:
        _plot_group_aggregate(
            grid_pct,
            aggregate,
            title="Mean contraction waveform by exposure and dose (+/-1 SD)",
            y_label="Contraction amplitude (a.u.)",
        )

def _render_tab_hrv(filtered_records):
    grid_pct, aggregate = _aggregate_group_profiles(
        filtered_records,
        time_key="rolling_rmssd_time_ms",
        value_key="rolling_rmssd_ms",
    )
    if not aggregate:
        st.info(
            "Not enough beats to compute rolling root mean square of "
            "successive interval differences for selected samples."
        )
    else:
        _plot_group_aggregate(
            grid_pct,
            aggregate,
            title=(
                "Mean rolling root mean square of successive interval differences "
                "by exposure and dose (+/-1 SD)"
            ),
            y_label="Rolling root mean square of successive interval differences (ms)",
        )


def _render_tab_data_table(filtered_df):
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

def _render_tab_statistical_analysis(filtered_df):
    numeric_cols = [
        col for col in filtered_df.columns
        if pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    if not numeric_cols:
        st.info("No numeric columns available for statistical analysis.")
        return
        
    preferred_metrics = [
        "mean_ibi_ms",
        "arrhythmia_risk_score",
        "arrhythmia_probability",
    ]
    default_metric = next(
        (metric_name for metric_name in preferred_metrics if metric_name in numeric_cols),
        numeric_cols[0],
    )
    metric = st.selectbox("Metric for statistical tests", numeric_cols, index=numeric_cols.index(default_metric), key="stat_metric")
    group_options = ["exposure", "concentration"]
    default_group_by = "concentration"
    group_by = st.selectbox(
        "Group by",
        group_options,
        index=group_options.index(default_group_by),
        key="stat_group"
    )

    grouped_values = []
    labels = []
    if group_by == "concentration":
        grouped_items = sorted(
            filtered_df.groupby(group_by),
            key=lambda item: _safe_float_sort_key(item[0]),
        )
    else:
        grouped_items = sorted(
            filtered_df.groupby(group_by),
            key=lambda item: str(item[0]),
        )

    for name, group in grouped_items:
        values = group[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        grouped_values.append(values)
        labels.append(str(name))

    if grouped_values:
        significance_flags, p_values, significance_label = _group_significance(
            grouped_values, labels, group_by, alpha=0.05
        )
        group_label = _pretty_group_label(group_by)
        
        stat_df = pd.DataFrame({
            group_label.capitalize(): labels,
            "p-value": p_values,
            "Significant (p < 0.05)": significance_flags
        })
        st.markdown(f"**Statistical significance of {metric} grouped by {group_by}**")
        st.caption(significance_label)
        st.dataframe(stat_df, use_container_width=True)

def _render_tab_graphs(filtered_df):
    numeric_cols = [
        col for col in filtered_df.columns
        if pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    if not numeric_cols:
        st.info("No numeric columns available for graphing.")
        return

    preferred_metrics = [
        "mean_ibi_ms",
        "arrhythmia_risk_score",
        "arrhythmia_probability",
    ]
    default_metric = next(
        (metric_name for metric_name in preferred_metrics if metric_name in numeric_cols),
        numeric_cols[0],
    )
    metric = st.selectbox("Metric distribution", numeric_cols, index=numeric_cols.index(default_metric), key="graph_metric")
    group_options = ["exposure", "concentration"]
    default_group_by = "concentration"
    group_by = st.selectbox(
        "Group by",
        group_options,
        index=group_options.index(default_group_by),
        key="graph_group"
    )

    grouped_values = []
    labels = []
    if group_by == "concentration":
        grouped_items = sorted(
            filtered_df.groupby(group_by),
            key=lambda item: _safe_float_sort_key(item[0]),
        )
    else:
        grouped_items = sorted(
            filtered_df.groupby(group_by),
            key=lambda item: str(item[0]),
        )

    for name, group in grouped_items:
        values = group[metric].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        grouped_values.append(values)
        labels.append(str(name))

    if grouped_values:
        significance_flags, p_values, significance_label = _group_significance(
            grouped_values, labels, group_by, alpha=0.05
        )
        display_labels = []
        for label, is_significant, p_value in zip(labels, significance_flags, p_values):
            significance_marker = "*" if is_significant else ""
            if np.isfinite(p_value):
                display_labels.append(f"{label}{significance_marker}\n(p={p_value:.3g})")
            else:
                display_labels.append(f"{label}{significance_marker}\n(p=n/a)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(
            grouped_values,
            tick_labels=display_labels,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "tab:blue", "alpha": 0.25, "edgecolor": "tab:blue"},
            medianprops={"color": "tab:orange", "linewidth": 1.8},
            meanprops={
                "marker": "D",
                "markerfacecolor": "tab:green",
                "markeredgecolor": "tab:green",
                "markersize": 5,
            },
        )
        metric_label = _pretty_metric_label(metric)
        group_label = _pretty_group_label(group_by)
        ax.set_title(f"Distribution of {metric_label} by {group_label}")
        ax.set_xlabel(group_label.capitalize())
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(
            handles=[
                Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25, label="IQR (Q1-Q3)"),
                Line2D([0], [0], color="tab:orange", lw=1.8, label="Median"),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="tab:green",
                    markerfacecolor="tab:green",
                    linestyle="None",
                    label="Mean",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="crimson",
                    markerfacecolor="crimson",
                    linestyle="None",
                    label=significance_label,
                ),
            ],
            loc="best",
            fontsize=8,
        )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    
def _plot_metric_boxplots(filtered_df, metrics, title_prefix):
    """Helper to plot boxplots for a set of metrics grouped by concentration."""
    group_by = "concentration"
    
    for metric in metrics:
        if metric not in filtered_df.columns:
            st.warning(f"Metric {metric} not found in the data.")
            continue
            
        grouped_values = []
        labels = []
        grouped_items = sorted(
            filtered_df.groupby(group_by),
            key=lambda item: _safe_float_sort_key(item[0]),
        )

        for name, group in grouped_items:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            grouped_values.append(values)
            labels.append(str(name))

        if not grouped_values:
            continue

        significance_flags, p_values, significance_label = _group_significance(
            grouped_values, labels, group_by, alpha=0.05
        )
        display_labels = []
        for label, is_significant, p_value in zip(labels, significance_flags, p_values):
            significance_marker = "*" if is_significant else ""
            if np.isfinite(p_value):
                display_labels.append(f"{label}{significance_marker}\n(p={p_value:.3g})")
            else:
                display_labels.append(f"{label}{significance_marker}\n(p=n/a)")
                
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(
            grouped_values,
            tick_labels=display_labels,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "tab:blue", "alpha": 0.25, "edgecolor": "tab:blue"},
            medianprops={"color": "tab:orange", "linewidth": 1.8},
            meanprops={
                "marker": "D",
                "markerfacecolor": "tab:green",
                "markeredgecolor": "tab:green",
                "markersize": 5,
            },
        )
        metric_label = _pretty_metric_label(metric)
        group_label = _pretty_group_label(group_by)
        ax.set_title(f"{title_prefix}: {metric_label} by {group_label}")
        ax.set_xlabel(group_label.capitalize())
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(
            handles=[
                Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25, label="IQR (Q1-Q3)"),
                Line2D([0], [0], color="tab:orange", lw=1.8, label="Median"),
                Line2D(
                    [0],
                    [0],
                    marker="D",
                    color="tab:green",
                    markerfacecolor="tab:green",
                    linestyle="None",
                    label="Mean",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="crimson",
                    markerfacecolor="crimson",
                    linestyle="None",
                    label=significance_label,
                ),
            ],
            loc="best",
            fontsize=8,
        )
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def _render_tab_contraction_amplitude(filtered_df):
    st.subheader("Contraction Amplitude Analysis")
    metrics = ["contraction_amplitude_mean", "contraction_amplitude_range"]
    _plot_metric_boxplots(filtered_df, metrics, "Amplitude Distribution")
    
def _render_tab_contraction_force(filtered_df, selected_sample, record_by_sample):
    st.subheader("Contraction Force Analysis")
    st.markdown(f"**Force Profile for {selected_sample}**")
    if selected_sample in record_by_sample:
        fish_record = record_by_sample[selected_sample]
        _plot_speed_profile(fish_record)
        
    metrics = [
        "force_of_contraction_mean_au",
        "force_of_contraction_std_au",
        "force_of_contraction_peak_au"
    ]
    _plot_metric_boxplots(filtered_df, metrics, "Force Distribution")
    
def _render_tab_transients(filtered_df):
    st.subheader("Transients Analysis")
    metrics = [
        "transient_rise_time_mean_ms",
        "transient_decay_time_mean_ms", 
        "transient_duration_mean_ms",
        "transient_fwhm_mean_ms"
    ]
    _plot_metric_boxplots(filtered_df, metrics, "Transient Distribution")

def _render_tab_models(output_dir, model_tables):
    st.caption(f"Model outputs loaded from `{output_dir}`")
    tables = model_tables

    st.subheader("Regression summary")
    linear_summary = tables["linear_regression_summary"]
    logistic_summary = tables["logistic_regression_summary"]
    trend_summary = tables["linear_trend_summary"]
    if linear_summary is None and logistic_summary is None and trend_summary is None:
        st.info("No regression output files found. Run `python data_analysis.py` to generate them.")
    else:
        if linear_summary is not None and not linear_summary.empty:
            c1, c2 = st.columns(2)
            c1.metric("Linear R²", f"{linear_summary.iloc[0]['r_squared']:.3f}")
            c2.metric("Linear n", int(linear_summary.iloc[0]["n_samples"]))
        if logistic_summary is not None and not logistic_summary.empty:
            c3, c4 = st.columns(2)
            c3.metric("Logistic accuracy (0.5)", f"{logistic_summary.iloc[0]['accuracy_at_0_5']:.3f}")
            c4.metric("Logistic McFadden R²", f"{logistic_summary.iloc[0]['mcfadden_pseudo_r2']:.3f}")
        if trend_summary is not None and not trend_summary.empty:
            st.dataframe(trend_summary, use_container_width=True)

        if tables["linear_regression_coefficients"] is not None:
            st.markdown("**Linear regression coefficients**")
            st.dataframe(tables["linear_regression_coefficients"], use_container_width=True)
        if tables["logistic_regression_coefficients"] is not None:
            st.markdown("**Logistic regression coefficients**")
            st.dataframe(tables["logistic_regression_coefficients"], use_container_width=True)

    st.subheader("ANOVA concentration summary")
    anova_summary = tables["anova_concentration_summary"]
    if anova_summary is None:
        st.info("No ANOVA output files found. Run `python data_analysis.py` to generate them.")
    else:
        st.dataframe(anova_summary, use_container_width=True)

    st.subheader("Unsupervised learning summary")
    cluster_summary = tables["unsupervised_cluster_summary"]
    pca_variance = tables["unsupervised_pca_variance"]
    if cluster_summary is None and pca_variance is None:
        st.info("No unsupervised output files found. Run `python data_analysis.py` to generate them.")
    else:
        if cluster_summary is not None:
            st.markdown("**Cluster composition**")
            st.dataframe(cluster_summary, use_container_width=True)
        if pca_variance is not None:
            st.markdown("**PCA explained variance**")
            st.dataframe(pca_variance, use_container_width=True)
        if tables["unsupervised_pca_loadings"] is not None:
            st.markdown("**PCA loadings**")
            st.dataframe(tables["unsupervised_pca_loadings"], use_container_width=True)
        if tables["unsupervised_assignments"] is not None:
            st.markdown("**Sample assignments (preview)**")
            st.dataframe(
                tables["unsupervised_assignments"].head(20),
                use_container_width=True,
            )

def _render_tab_conclusions(filtered_df, model_tables):
    st.subheader("Conclusions for current filter selection")
    risk_series = filtered_df["arrhythmia_risk_score"].dropna()

    k1, k2, k3 = st.columns(3)
    k1.metric("Samples in view", int(len(filtered_df)))
    k2.metric(
        "Arrhymia rate",
        f"{100.0 * float(filtered_df['Arrhymia'].astype(float).mean()):.1f}%",
    )
    k3.metric(
        "Mean risk score",
        f"{float(risk_series.mean()):.3f}" if len(risk_series) > 0 else "n/a",
    )

    top_cols = [
        col for col in [
            "sample",
            "exposure",
            "concentration",
            "arrhythmia_risk_score",
            "Arrhymia",
            "paired_ttest_pvalue_vs_control0_mean_ibi",
        ]
        if col in filtered_df.columns
    ]
    if top_cols:
        st.markdown("**Top 5 highest-risk samples**")
        top_risk = filtered_df.sort_values(
            "arrhythmia_risk_score",
            ascending=False,
            na_position="last",
        ).head(5)
        st.dataframe(top_risk[top_cols], use_container_width=True)

    trend_summary = model_tables["linear_trend_summary"]
    if trend_summary is not None and not trend_summary.empty:
        slope = float(trend_summary.iloc[0]["slope"])
        p_value = float(trend_summary.iloc[0]["p_value"])
        direction = "increases" if slope > 0 else "decreases"
        st.markdown(
            f"- Concentration trend: risk score **{direction}** with concentration "
            f"(slope={slope:.4f}, p={p_value:.4g})."
        )

    logistic_summary = model_tables["logistic_regression_summary"]
    if logistic_summary is not None and not logistic_summary.empty:
        acc = float(logistic_summary.iloc[0]["accuracy_at_0_5"])
        pseudo_r2 = float(logistic_summary.iloc[0]["mcfadden_pseudo_r2"])
        st.markdown(
            f"- Logistic model summary: accuracy={acc:.3f}, McFadden R²={pseudo_r2:.3f}."
        )

    anova_summary = model_tables.get("anova_concentration_summary")
    if anova_summary is not None and not anova_summary.empty:
        significant_anova = anova_summary[anova_summary["p_value"] < 0.05]
        if not significant_anova.empty:
            for _, row in significant_anova.iterrows():
                st.markdown(
                    f"- ANOVA significant finding for exposure **{row['exposure']}**: "
                    f"Concentration affects **{row['metric']}** "
                    f"(p={row['p_value']:.4g})."
                )

    cluster_summary = model_tables["unsupervised_cluster_summary"]
    if cluster_summary is not None and not cluster_summary.empty:
        highest_cluster = cluster_summary.sort_values(
            "arrhythmia_rate",
            ascending=False,
        ).iloc[0]
        st.markdown(
            f"- Highest-risk discovered cluster: `{highest_cluster['model']}` "
            f"cluster {int(highest_cluster['cluster_id'])} with "
            f"arrhythmia_rate={float(highest_cluster['arrhythmia_rate']):.3f} "
            f"(n={int(highest_cluster['n_samples'])})."
        )


def _render_tab_technical(results_dir, output_dir):
    st.subheader("Technical architecture")
    st.markdown(
        f"The dashboard consumes sample folders from `{results_dir}` and optional model CSV outputs "
        f"from `{output_dir}`. Rendering is deterministic for a given set of sidebar filters and settings."
    )

    st.markdown("### Data ingestion and data model")
    st.markdown(
        "- `_load_dashboard_data(results_dir)` calls `load_all_sample_timeseries(...)` and materializes:\n"
        "  - `records`: per-sample dictionaries containing raw time-series arrays and derived metrics.\n"
        "  - `summary_df`: DataFrame projection restricted to `SUMMARY_COLUMNS`.\n"
        "- `record_by_sample` is a direct sample-id lookup map used by the fish-level tab.\n"
        "- Sidebar filters (`exposure`, `concentration`, `excluded cases`) are applied to `summary_df`; "
        "the resulting `sample` list is then used to derive `filtered_records`."
    )

    st.markdown("### Cache boundaries and invalidation")
    st.markdown(
        "- `_load_dashboard_data(results_dir)` is `@st.cache_data`: cache key boundary is the function input "
        "(the resolved results directory path).\n"
        "- `_load_model_output_tables(output_dir)` is `@st.cache_data`: cache key boundary is the resolved "
        "output directory path and file contents read in that call.\n"
        "- The **Reload data** button executes `st.cache_data.clear()`, invalidating both caches in one step."
    )

    st.markdown("### Peak detection and sawtooth masking")
    st.markdown(
        "Peak extraction is executed upstream in `data_analysis.detect_peaks(...)`:\n"
        "- Baseline removal uses a rolling median (default 800 ms window), then `find_peaks` on detrended data.\n"
        "- Prominence uses `prominence_factor * spread`, where spread is outlier-trimmed when "
        "`range / trimmed_range > 2.0`.\n"
        "- A fallback raw-signal peak pass is used if detrended detection yields fewer than two peaks.\n"
        "- Candidate peaks are grouped into sawtooth clusters if temporal spacing is below a dynamic window "
        "`min(500 ms, 0.5 * median_ibi)` and group span is bounded by median IBI.\n"
        "- Each cluster is reduced to the maximum-amplitude peak; cluster-derived peaks are marked "
        "`sawtooth_peak=True`."
    )
    st.markdown(
        "In the fish profile plot, non-sawtooth peaks are rendered as red markers and sawtooth-derived peaks "
        "as blue markers."
    )

    st.markdown("### HRV formulas and rolling features")
    st.markdown(
        "- `IBI_i = peak_time_i - peak_time_{i-1}` (ms)\n"
        "- `mean_ibi_ms = mean(IBI)`\n"
        "- `sdnn_ms = std(IBI, ddof=1)`\n"
        "- `rmssd_ms = sqrt(mean(diff(IBI)^2))`\n"
        "- `cv_ibi = sdnn_ms / mean_ibi_ms`\n"
        "- `pnn50 = 100 * mean(|diff(IBI)| > 50 ms)`\n"
        "- `mean_hr_bpm = 60000 / mean_ibi_ms`\n"
        "- Rolling RMSSD uses a fixed window (`ROLLING_RMSSD_WINDOW = 5`) over consecutive IBI segments."
    )

    st.markdown("### Contraction amplitude analysis pipeline")
    st.markdown(
        "Amplitude features are computed upstream in `data_analysis.compute_contraction_amplitudes(...)`:\n"
        "- For each detected peak, the local trough is taken from the interval between the previous peak boundary "
        "and the current peak.\n"
        "- Beat amplitude is `peak_value - trough_value`, clipped to `>= 0`.\n"
        "- Per-sample summary metrics are persisted as `contraction_amplitude_mean`, "
        "`contraction_amplitude_std`, and `contraction_amplitude_range`.\n"
        "- The **Contraction amplitude analysis** tab renders concentration-grouped boxplots for "
        "`contraction_amplitude_mean` and `contraction_amplitude_range`, with significance annotations."
    )

    st.markdown("### Contraction force analysis pipeline")
    st.markdown(
        "Force proxy features are derived from `speed-of-contraction.txt` and "
        "`data_analysis.compute_force_of_contraction(...)`:\n"
        "- If speed data exists, the dashboard records `speed_time_ms` and `speed_values`; otherwise empty arrays "
        "are carried through.\n"
        "- Force metrics use non-negative speed (`clip(speed, 0, +inf)`) and report "
        "`force_of_contraction_mean_au`, `force_of_contraction_std_au`, and `force_of_contraction_peak_au`.\n"
        "- If finite speed values are unavailable, these metrics are `NaN`.\n"
        "- The **Contraction force analysis** tab combines a fish-level speed profile view "
        "(`_plot_speed_profile`) with concentration-grouped boxplots for the three force metrics."
    )

    st.markdown("### Transients analysis pipeline")
    st.markdown(
        "Transient timing features are computed in `data_analysis.compute_transient_metrics(...)` on a "
        "per-beat window:\n"
        "- Beat windows are bounded by midpoints between adjacent peaks.\n"
        "- Rise time: from left-side local minimum to peak.\n"
        "- Decay time: from peak to right-side local minimum.\n"
        "- Duration: from left minimum to right minimum.\n"
        "- FWHM: crossing width around half-max, where baseline is the minimum of the left/right minima.\n"
        "- Per-sample means are exposed as `transient_rise_time_mean_ms`, `transient_decay_time_mean_ms`, "
        "`transient_duration_mean_ms`, and `transient_fwhm_mean_ms`.\n"
        "- The **Transients analysis** tab renders concentration-grouped boxplots with significance annotations "
        "for these four metrics."
    )

    st.markdown("### Arrhythmia risk score and decision rule")
    st.markdown(
        "The risk score is a heuristic composite from `compute_arrhythmia_risk(...)`:\n"
        "- `score_cv = sigmoid(30 * (cv - 0.15))`\n"
        "- `score_rmssd = sigmoid(30 * (rmssd_rel - 0.15))`, where `rmssd_rel = rmssd / mean_ibi`\n"
        "- `outlier_frac = mean(|IBI - median(IBI)| / median(IBI) > 0.30)`\n"
        "- `score_outlier = sigmoid(20 * (outlier_frac - 0.15))`\n"
        "- `arrhythmia_risk_score = mean(score_cv, score_rmssd, score_outlier)`\n"
        "- `Arrhymia` decision is `risk_score > arrhythmia_threshold` with data sufficiency gating "
        "(minimum IBI count checks)."
    )

    st.markdown("### Model outputs and advanced-table hydration")
    st.markdown(
        "Model tables are loaded from `output_dir` using `MODEL_OUTPUT_FILES`. Missing files are represented "
        "as `None`, allowing partial output sets. Advanced tabs then consume these tables for regression "
        "metrics, coefficient tables, clustering summaries, PCA variance/loadings, and sample assignments."
    )

    st.markdown("### Rendering flow")
    st.markdown(
        "1. Initialize page config, CSS, and session-state defaults.\n"
        "2. Resolve absolute paths for results and output directories.\n"
        "3. Load cached data/model tables.\n"
        "4. Apply sidebar filters and build `filtered_df` and `filtered_records`.\n"
        "5. Construct tabs as: base tabs + optional advanced tabs + optional technical tab.\n"
        "6. Execute tab-specific renderers (`_render_tab_*`) to produce plots, tables, and summaries."
    )


def main():
    st.set_page_config(page_title="Zebrafish HRV Dashboard", layout="wide")
    _apply_text_wrap_css()
    st.title("Zebrafish contraction and HRV dashboard")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.abspath(os.path.join(script_dir, "..", "MM_Results"))
    default_output = os.path.abspath(os.path.join(script_dir, "..", "output"))

    if "results_dir" not in st.session_state:
        st.session_state["results_dir"] = default_results
    if "output_dir" not in st.session_state:
        st.session_state["output_dir"] = default_output


    results_dir = os.path.abspath(st.session_state["results_dir"])
    output_dir = os.path.abspath(st.session_state["output_dir"])
    try:
        records, summary_df = _load_dashboard_data(results_dir)
    except Exception as exc:
        st.error(f"Failed to load data from {results_dir}: {exc}")
        st.stop()

    record_by_sample = {record["sample"]: record for record in records}

    st.sidebar.header("Filters")
    exposures = sorted(summary_df["exposure"].unique().tolist())
    selected_exposures = st.sidebar.multiselect("Exposure", exposures, default=exposures)

    filtered_df = summary_df[summary_df["exposure"].isin(selected_exposures)].copy()

    concentration_options = sorted(
        filtered_df["concentration"].astype(str).unique().tolist(),
        key=_safe_float_sort_key,
    )
    selected_concentrations = st.sidebar.multiselect(
        "Dose (concentration)", concentration_options, default=concentration_options
    )
    filtered_df = filtered_df[
        filtered_df["concentration"].astype(str).isin(selected_concentrations)
    ].copy()

    removable_cases = sorted(filtered_df["sample"].tolist())
    default_excluded_cases = [
        sample_name for sample_name in DEFAULT_EXCLUDED_CASES
        if sample_name in removable_cases
    ]
    excluded_cases = st.sidebar.multiselect(
        "Remove cases",
        removable_cases,
        default=default_excluded_cases,
        help="Exclude selected cases from all current dashboard plots and tables.",
    )
    if excluded_cases:
        filtered_df = filtered_df[~filtered_df["sample"].isin(excluded_cases)].copy()

    if filtered_df.empty:
        st.warning("No samples match the selected filters.")
        st.stop()

    sample_options = filtered_df["sample"].tolist()
    selected_sample = st.sidebar.selectbox("Fish sample", sample_options)
    with st.sidebar.expander("Variables & abbreviations", expanded=False):
        st.markdown(VARIABLE_GLOSSARY_MARKDOWN)
    with st.sidebar.expander("Settings", expanded=False):
        all_tab_specs = [
            ("Fish profile", "fish"),
            ("Dose profile", "dose"),
            ("HRV over time", "hrv"),
            ("Contraction amplitude analysis", "amplitude"),
            ("Contraction force analysis", "force"),
            ("Transients analysis", "transients"),
            ("Data table", "data_table"),
            ("Statistical analysis", "statistical_analysis"),
            ("Graphs", "graphs"),
            ("Model summaries", "models"),
            ("Conclusions", "conclusions"),
            ("Technical architecture", "technical")
        ]

        all_tab_labels = [label for label, _ in all_tab_specs]

        st.markdown("### View Options")
        selected_tabs = st.segmented_control(
            "Select visible tabs",
            options=all_tab_labels,
            default=["Fish profile", "Statistical analysis", "Graphs"],
            selection_mode="multi"
        )
        st.caption("Data source")
        st.text_input("MM_Results folder", key="results_dir")
        st.text_input("Output folder", key="output_dir")
        if st.button("Reload data"):
            st.cache_data.clear()
    output_dir = os.path.abspath(st.session_state["output_dir"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19038805.svg)]"
        "(https://doi.org/10.5281/zenodo.19038805)"
    )

    filtered_records = [record_by_sample[sample] for sample in sample_options if sample in record_by_sample]
    model_tables = _load_model_output_tables(output_dir)

    c1, c2, c3 = st.columns(3)
    c1.metric("Fish samples", len(filtered_df))
    c2.metric("Exposure groups", int(filtered_df["exposure"].nunique()))
    c3.metric("Dose groups", int(filtered_df["concentration"].nunique()))

    
    if not selected_tabs:
        st.info("Please select at least one tab to view.")
        return
        
    visible_tab_specs = [(label, key) for label, key in all_tab_specs if label in selected_tabs]
    tab_objects = st.tabs([label for label, _ in visible_tab_specs])
    tab_by_key = {key: tab for (_, key), tab in zip(visible_tab_specs, tab_objects)}

    if "fish" in tab_by_key:
        with tab_by_key["fish"]:
            _render_tab_fish(selected_sample, record_by_sample)

    if "dose" in tab_by_key:
        with tab_by_key["dose"]:
            _render_tab_dose(filtered_records)

    if "hrv" in tab_by_key:
        with tab_by_key["hrv"]:
            _render_tab_hrv(filtered_records)
            
    if "amplitude" in tab_by_key:
        with tab_by_key["amplitude"]:
            _render_tab_contraction_amplitude(filtered_df)
            
    if "force" in tab_by_key:
        with tab_by_key["force"]:
            _render_tab_contraction_force(filtered_df, selected_sample, record_by_sample)
            
    if "transients" in tab_by_key:
        with tab_by_key["transients"]:
            _render_tab_transients(filtered_df)

    if "data_table" in tab_by_key:
        with tab_by_key["data_table"]:
            _render_tab_data_table(filtered_df)
            
    if "statistical_analysis" in tab_by_key:
        with tab_by_key["statistical_analysis"]:
            _render_tab_statistical_analysis(filtered_df)
            
    if "graphs" in tab_by_key:
        with tab_by_key["graphs"]:
            _render_tab_graphs(filtered_df)

    if "models" in tab_by_key:
        with tab_by_key["models"]:
            _render_tab_models(output_dir, model_tables)

    if "conclusions" in tab_by_key:
        with tab_by_key["conclusions"]:
            _render_tab_conclusions(filtered_df, model_tables)

    if "technical" in tab_by_key:
        with tab_by_key["technical"]:
            _render_tab_technical(results_dir, output_dir)


if __name__ == "__main__":
    main()
