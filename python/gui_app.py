"""Streamlit dashboard for zebrafish contraction and HRV exploration.

Usage:
    streamlit run gui_app.py
"""

import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import f_oneway
from scipy.signal import detrend

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
    "unsupervised_cluster_summary": "unsupervised_cluster_summary.csv",
    "unsupervised_assignments": "unsupervised_assignments.csv",
    "unsupervised_pca_variance": "unsupervised_pca_variance.csv",
    "unsupervised_pca_loadings": "unsupervised_pca_loadings.csv",
}
EXPOSURE_COLORS = {
    "Phe": {"primary": "tab:blue", "light": "#6baed6", "dark": "#08519c"},
    "Terf": {"primary": "tab:red", "light": "#fc9272", "dark": "#a50f15"},
    "0": {"primary": "tab:gray", "light": "#969696", "dark": "#525252"},
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


def _get_exposure_color(exposure, shade="primary"):
    """Get color for exposure type from predefined color scheme."""
    if exposure in EXPOSURE_COLORS:
        return EXPOSURE_COLORS[exposure].get(shade, EXPOSURE_COLORS[exposure]["primary"])
    return "tab:gray"


def _filter_records_by_exposure(records, exposure):
    """Filter records list by exposure type."""
    return [r for r in records if r.get("exposure") == exposure]


def _filter_df_by_exposure(df, exposure):
    """Filter dataframe by exposure type."""
    return df[df["exposure"] == exposure].copy()


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


def _calculate_anova_for_metric(df, metric, exposure_filter=None, group_by_exposure=False):
    """
    Calculate one-way ANOVA for a given metric.
    
    Args:
        df: DataFrame with sample data
        metric: Column name to test
        exposure_filter: If provided, filter to only this exposure type
        group_by_exposure: If True, group by exposure instead of concentration
    
    Returns:
        dict with F-statistic, p-value, df_between, df_within, eta_squared, interpretation
    """
    working_df = df.copy()
    
    if exposure_filter is not None:
        working_df = working_df[working_df["exposure"] == exposure_filter]
    
    if len(working_df) < 3:
        return None
    
    group_column = "exposure" if group_by_exposure else "concentration"
    
    groups = []
    group_labels = []
    for name, group_df in working_df.groupby(group_column):
        values = group_df[metric].dropna().values
        if len(values) >= 2:
            groups.append(values)
            group_labels.append(str(name))
    
    if len(groups) < 2:
        return None
    
    f_stat, p_value = f_oneway(*groups)
    
    n_total = sum(len(g) for g in groups)
    n_groups = len(groups)
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((val - grand_mean)**2 for g in groups for val in g)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    
    if p_value < 0.001:
        interpretation = "Highly significant (p < 0.001)"
    elif p_value < 0.01:
        interpretation = "Very significant (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Significant (p < 0.05)"
    else:
        interpretation = "Not significant (p ≥ 0.05)"
    
    return {
        "F-statistic": f_stat,
        "p-value": p_value,
        "df_between": df_between,
        "df_within": df_within,
        "eta_squared": eta_squared,
        "interpretation": interpretation,
        "n_groups": n_groups,
        "n_total": n_total,
    }


def _run_all_anova_tests(df, metrics):
    """
    Run ANOVA tests for all specified metrics across different groupings.
    
    Returns:
        dict of DataFrames with results for different test types
    """
    results = {
        "phe_only": [],
        "terf_only": [],
        "both_exposures": [],
        "grouped_by_exposure": [],
    }
    
    for metric in metrics:
        if metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
            continue
        
        phe_result = _calculate_anova_for_metric(df, metric, exposure_filter="Phe")
        if phe_result:
            phe_result["metric"] = metric
            phe_result["test_type"] = "Phe only (by concentration)"
            results["phe_only"].append(phe_result)
        
        terf_result = _calculate_anova_for_metric(df, metric, exposure_filter="Terf")
        if terf_result:
            terf_result["metric"] = metric
            terf_result["test_type"] = "Terf only (by concentration)"
            results["terf_only"].append(terf_result)
        
        both_result = _calculate_anova_for_metric(df, metric, exposure_filter=None, group_by_exposure=False)
        if both_result:
            both_result["metric"] = metric
            both_result["test_type"] = "Both exposures (by concentration)"
            results["both_exposures"].append(both_result)
        
        exposure_result = _calculate_anova_for_metric(df, metric, exposure_filter=None, group_by_exposure=True)
        if exposure_result:
            exposure_result["metric"] = metric
            exposure_result["test_type"] = "Grouped by exposure"
            results["grouped_by_exposure"].append(exposure_result)
    
    result_dfs = {}
    for key, result_list in results.items():
        if result_list:
            result_dfs[key] = pd.DataFrame(result_list)
        else:
            result_dfs[key] = None
    
    return result_dfs


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


def _plot_group_aggregate(grid_pct, aggregate, title, y_label, group_by_exposure=False):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label in sorted(aggregate.keys()):
        mean = aggregate[label]["mean"]
        std = aggregate[label]["std"]
        n = aggregate[label]["n"]
        
        # Extract exposure from label (format: "Exposure_Concentration")
        if group_by_exposure:
            exposure = label.split("_")[0] if "_" in label else "0"
            color = _get_exposure_color(exposure)
            ax.plot(grid_pct, mean, linewidth=1.5, label=f"{label} (n={n})", color=color)
            ax.fill_between(grid_pct, mean - std, mean + std, alpha=0.2, color=color)
        else:
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

def _render_tab_dose(filtered_records, group_by_exposure=False):
    if group_by_exposure:
        # Get unique exposures from filtered records
        exposures = sorted(set(r.get("exposure", "0") for r in filtered_records))
        
        if len(exposures) <= 1:
            # Only one exposure, show single graph
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
                    group_by_exposure=True,
                )
        else:
            # Multiple exposures, show side-by-side
            cols = st.columns(len(exposures))
            for idx, exposure in enumerate(exposures):
                exposure_records = _filter_records_by_exposure(filtered_records, exposure)
                grid_pct, aggregate = _aggregate_group_profiles(
                    exposure_records,
                    time_key="time_ms",
                    value_key="contraction_values",
                )
                with cols[idx]:
                    if not aggregate:
                        st.info(f"Not enough data for {exposure}.")
                    else:
                        _plot_group_aggregate(
                            grid_pct,
                            aggregate,
                            title=f"{exposure} - Contraction waveform by dose (+/-1 SD)",
                            y_label="Contraction amplitude (a.u.)",
                            group_by_exposure=True,
                        )
    else:
        # Original behavior - show all together with color coding
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
                group_by_exposure=True,
            )

def _render_tab_hrv(filtered_records, group_by_exposure=False):
    if group_by_exposure:
        # Get unique exposures from filtered records
        exposures = sorted(set(r.get("exposure", "0") for r in filtered_records))
        
        if len(exposures) <= 1:
            # Only one exposure, show single graph
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
                    group_by_exposure=True,
                )
        else:
            # Multiple exposures, show side-by-side
            cols = st.columns(len(exposures))
            for idx, exposure in enumerate(exposures):
                exposure_records = _filter_records_by_exposure(filtered_records, exposure)
                grid_pct, aggregate = _aggregate_group_profiles(
                    exposure_records,
                    time_key="rolling_rmssd_time_ms",
                    value_key="rolling_rmssd_ms",
                )
                with cols[idx]:
                    if not aggregate:
                        st.info(f"Not enough data for {exposure}.")
                    else:
                        _plot_group_aggregate(
                            grid_pct,
                            aggregate,
                            title=f"{exposure} - Rolling RMSSD by dose (+/-1 SD)",
                            y_label="Rolling root mean square of successive interval differences (ms)",
                            group_by_exposure=True,
                        )
    else:
        # Original behavior - show all together with color coding
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
                group_by_exposure=True,
            )


def _render_tab_data_table(filtered_df):
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

def _render_tab_statistical_analysis(filtered_df):
    st.subheader("Statistical Analysis")
    
    numeric_cols = [
        col for col in filtered_df.columns
        if pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    if not numeric_cols:
        st.info("No numeric columns available for statistical analysis.")
        return
    
    st.markdown("### ANOVA Tests")
    st.markdown("""
    Analysis of Variance (ANOVA) tests whether there are significant differences between group means.
    Tests are performed for key HRV metrics across different groupings.
    """)
    
    anova_metrics = ["mean_ibi_ms", "sdnn_ms", "rmssd_ms"]
    available_anova_metrics = [m for m in anova_metrics if m in filtered_df.columns]
    
    if not available_anova_metrics:
        st.warning("Required HRV metrics (mean_ibi_ms, sdnn_ms, rmssd_ms) not found in data.")
    else:
        with st.spinner("Calculating ANOVA tests..."):
            anova_results = _run_all_anova_tests(filtered_df, available_anova_metrics)
        
        tabs = st.tabs([
            "Phe Only (by Concentration)",
            "Terf Only (by Concentration)",
            "Both Exposures (by Concentration)",
            "Grouped by Exposure"
        ])
        
        with tabs[0]:
            st.markdown("**ANOVA for Phe exposure only, grouped by concentration**")
            if anova_results["phe_only"] is not None:
                display_df = anova_results["phe_only"][["metric", "F-statistic", "p-value", 
                                                          "df_between", "df_within", "eta_squared", 
                                                          "n_groups", "n_total", "interpretation"]].copy()
                display_df["F-statistic"] = display_df["F-statistic"].round(3)
                display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4g}")
                display_df["eta_squared"] = display_df["eta_squared"].round(4)
                st.dataframe(display_df, use_container_width=True)
                
                sig_results = anova_results["phe_only"][anova_results["phe_only"]["p-value"] < 0.05]
                if not sig_results.empty:
                    st.success(f"Found {len(sig_results)} significant result(s) at p < 0.05")
                else:
                    st.info("No significant differences found at p < 0.05")
            else:
                st.info("Not enough data for ANOVA test")
        
        with tabs[1]:
            st.markdown("**ANOVA for Terf exposure only, grouped by concentration**")
            if anova_results["terf_only"] is not None:
                display_df = anova_results["terf_only"][["metric", "F-statistic", "p-value", 
                                                           "df_between", "df_within", "eta_squared", 
                                                           "n_groups", "n_total", "interpretation"]].copy()
                display_df["F-statistic"] = display_df["F-statistic"].round(3)
                display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4g}")
                display_df["eta_squared"] = display_df["eta_squared"].round(4)
                st.dataframe(display_df, use_container_width=True)
                
                sig_results = anova_results["terf_only"][anova_results["terf_only"]["p-value"] < 0.05]
                if not sig_results.empty:
                    st.success(f"Found {len(sig_results)} significant result(s) at p < 0.05")
                else:
                    st.info("No significant differences found at p < 0.05")
            else:
                st.info("Not enough data for ANOVA test")
        
        with tabs[2]:
            st.markdown("**ANOVA for both exposures together, grouped by concentration**")
            if anova_results["both_exposures"] is not None:
                display_df = anova_results["both_exposures"][["metric", "F-statistic", "p-value", 
                                                                "df_between", "df_within", "eta_squared", 
                                                                "n_groups", "n_total", "interpretation"]].copy()
                display_df["F-statistic"] = display_df["F-statistic"].round(3)
                display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4g}")
                display_df["eta_squared"] = display_df["eta_squared"].round(4)
                st.dataframe(display_df, use_container_width=True)
                
                sig_results = anova_results["both_exposures"][anova_results["both_exposures"]["p-value"] < 0.05]
                if not sig_results.empty:
                    st.success(f"Found {len(sig_results)} significant result(s) at p < 0.05")
                else:
                    st.info("No significant differences found at p < 0.05")
            else:
                st.info("Not enough data for ANOVA test")
        
        with tabs[3]:
            st.markdown("**ANOVA comparing Phe vs Terf exposures**")
            if anova_results["grouped_by_exposure"] is not None:
                display_df = anova_results["grouped_by_exposure"][["metric", "F-statistic", "p-value", 
                                                                     "df_between", "df_within", "eta_squared", 
                                                                     "n_groups", "n_total", "interpretation"]].copy()
                display_df["F-statistic"] = display_df["F-statistic"].round(3)
                display_df["p-value"] = display_df["p-value"].apply(lambda x: f"{x:.4g}")
                display_df["eta_squared"] = display_df["eta_squared"].round(4)
                st.dataframe(display_df, use_container_width=True)
                
                sig_results = anova_results["grouped_by_exposure"][anova_results["grouped_by_exposure"]["p-value"] < 0.05]
                if not sig_results.empty:
                    st.success(f"Found {len(sig_results)} significant result(s) at p < 0.05")
                else:
                    st.info("No significant differences found at p < 0.05")
            else:
                st.info("Not enough data for ANOVA test")
        
        st.markdown("""
        **Interpretation Guide:**
        - **F-statistic**: Ratio of between-group variance to within-group variance. Higher values indicate greater differences.
        - **p-value**: Probability of observing this result by chance. p < 0.05 typically indicates significance.
        - **df_between**: Degrees of freedom between groups (k - 1, where k = number of groups)
        - **df_within**: Degrees of freedom within groups (N - k, where N = total samples)
        - **eta_squared (η²)**: Effect size. 0.01 = small, 0.06 = medium, 0.14 = large effect
        - **n_groups**: Number of groups compared
        - **n_total**: Total number of samples across all groups
        """)
    
    st.markdown("---")
    st.markdown("### Pairwise Comparisons (T-tests)")
        
    preferred_metrics = [
        "mean_ibi_ms",
        "arrhythmia_risk_score",
        "arrhythmia_probability",
    ]
    default_metric = next(
        (metric_name for metric_name in preferred_metrics if metric_name in numeric_cols),
        numeric_cols[0],
    )
    metric = st.selectbox("Metric for pairwise tests", numeric_cols, index=numeric_cols.index(default_metric), key="stat_metric")
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


def _render_tab_graphs(filtered_df, group_by_exposure=False):
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

    if group_by_exposure and group_by == "concentration":
        # Show side-by-side graphs for each exposure
        exposures = sorted(filtered_df["exposure"].unique())
        
        if len(exposures) <= 1:
            # Only one exposure, show single graph
            _render_single_graph(filtered_df, metric, group_by, group_by_exposure=True)
        else:
            # Multiple exposures, show side-by-side
            cols = st.columns(len(exposures))
            for idx, exposure in enumerate(exposures):
                exposure_df = _filter_df_by_exposure(filtered_df, exposure)
                with cols[idx]:
                    st.markdown(f"**{exposure}**")
                    _render_single_graph(exposure_df, metric, group_by, group_by_exposure=True, exposure=exposure)
    else:
        # Original behavior
        _render_single_graph(filtered_df, metric, group_by, group_by_exposure=False)


def _render_single_graph(filtered_df, metric, group_by, group_by_exposure=False, exposure=None):
    """Helper function to render a single boxplot graph."""
    grouped_values = []
    labels = []
    exposures_for_colors = []
    
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
        # Get the exposure for this group (for coloring)
        if group_by_exposure and group_by == "concentration":
            if exposure:
                exposures_for_colors.append(exposure)
            else:
                group_exposures = group["exposure"].unique()
                exposures_for_colors.append(group_exposures[0] if len(group_exposures) > 0 else "0")

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
        
        if group_by_exposure and group_by == "concentration" and exposures_for_colors:
            # Create box plots with exposure-specific colors
            bp = ax.boxplot(
                grouped_values,
                tick_labels=display_labels,
                showmeans=True,
                patch_artist=True,
                medianprops={"color": "tab:orange", "linewidth": 1.8},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "tab:green",
                    "markeredgecolor": "tab:green",
                    "markersize": 5,
                },
            )
            # Color each box based on its exposure
            for patch, exp in zip(bp['boxes'], exposures_for_colors):
                color = _get_exposure_color(exp)
                patch.set_facecolor(color)
                patch.set_alpha(0.25)
                patch.set_edgecolor(color)
        else:
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
        title = f"Distribution of {metric_label} by {group_label}"
        if exposure:
            title = f"{exposure} - {title}"
        ax.set_title(title)
        ax.set_xlabel(group_label.capitalize())
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
        
        legend_handles = [
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
        ]
        
        if group_by_exposure and group_by == "concentration" and exposures_for_colors:
            # Add exposure color legend
            unique_exposures = sorted(set(exposures_for_colors))
            for exp in unique_exposures:
                color = _get_exposure_color(exp)
                legend_handles.insert(0, Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"{exp} IQR"))
        else:
            legend_handles.insert(0, Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25, label="IQR (Q1-Q3)"))
        
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


    
def _plot_metric_boxplots(filtered_df, metrics, title_prefix, group_by_exposure=False):
    """Helper to plot boxplots for a set of metrics grouped by concentration."""
    group_by = "concentration"
    
    if group_by_exposure:
        # Get unique exposures
        exposures = sorted(filtered_df["exposure"].unique())
        
        if len(exposures) <= 1:
            # Only one exposure, show single set of plots with color-coding
            _plot_single_metric_boxplots(filtered_df, metrics, title_prefix, group_by_exposure=True)
        else:
            # Multiple exposures, show side-by-side for each metric
            for metric in metrics:
                if metric not in filtered_df.columns:
                    st.warning(f"Metric {metric} not found in the data.")
                    continue
                
                st.markdown(f"**{_pretty_metric_label(metric)}**")
                cols = st.columns(len(exposures))
                for idx, exposure in enumerate(exposures):
                    exposure_df = _filter_df_by_exposure(filtered_df, exposure)
                    with cols[idx]:
                        st.caption(f"{exposure} Exposure")
                        _plot_single_metric_boxplots(
                            exposure_df, 
                            [metric], 
                            title_prefix, 
                            group_by_exposure=True,
                            show_title=False
                        )
    else:
        # Original behavior - show all together without color coding
        _plot_single_metric_boxplots(filtered_df, metrics, title_prefix, group_by_exposure=False)


def _plot_single_metric_boxplots(filtered_df, metrics, title_prefix, group_by_exposure=False, show_title=True):
    """Helper to plot boxplots for a set of metrics - single exposure or combined."""
    group_by = "concentration"
    
    for metric in metrics:
        if metric not in filtered_df.columns:
            st.warning(f"Metric {metric} not found in the data.")
            continue
            
        grouped_values = []
        labels = []
        exposures_for_colors = []
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
            # Get the exposure for this group (for coloring)
            if group_by_exposure:
                group_exposures = group["exposure"].unique()
                exposures_for_colors.append(group_exposures[0] if len(group_exposures) > 0 else "0")

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
        
        if group_by_exposure and exposures_for_colors:
            # Create box plots with exposure-specific colors
            bp = ax.boxplot(
                grouped_values,
                tick_labels=display_labels,
                showmeans=True,
                patch_artist=True,
                medianprops={"color": "tab:orange", "linewidth": 1.8},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "tab:green",
                    "markeredgecolor": "tab:green",
                    "markersize": 5,
                },
            )
            # Color each box based on its exposure
            for patch, exp in zip(bp['boxes'], exposures_for_colors):
                color = _get_exposure_color(exp)
                patch.set_facecolor(color)
                patch.set_alpha(0.25)
                patch.set_edgecolor(color)
        else:
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
        if show_title:
            ax.set_title(f"{title_prefix}: {metric_label} by {group_label}")
        ax.set_xlabel(group_label.capitalize())
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=8)
        
        legend_handles = [
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
        ]
        
        if group_by_exposure and exposures_for_colors:
            # Add exposure color legend
            unique_exposures = sorted(set(exposures_for_colors))
            for exp in unique_exposures:
                color = _get_exposure_color(exp)
                legend_handles.insert(0, Patch(facecolor=color, edgecolor=color, alpha=0.25, label=f"{exp} IQR"))
        else:
            legend_handles.insert(0, Patch(facecolor="tab:blue", edgecolor="tab:blue", alpha=0.25, label="IQR (Q1-Q3)"))
        
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def _render_tab_contraction_amplitude(filtered_df, group_by_exposure=False):
    st.subheader("Contraction Amplitude Analysis")
    metrics = ["contraction_amplitude_mean", "contraction_amplitude_range"]
    _plot_metric_boxplots(filtered_df, metrics, "Amplitude Distribution", group_by_exposure=group_by_exposure)
    
def _render_tab_contraction_force(filtered_df, selected_sample, record_by_sample, group_by_exposure=False):
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
    _plot_metric_boxplots(filtered_df, metrics, "Force Distribution", group_by_exposure=group_by_exposure)
    
def _render_tab_transients(filtered_df, group_by_exposure=False):
    st.subheader("Transients Analysis")
    metrics = [
        "transient_rise_time_mean_ms",
        "transient_decay_time_mean_ms", 
        "transient_duration_mean_ms",
        "transient_fwhm_mean_ms"
    ]
    _plot_metric_boxplots(filtered_df, metrics, "Transient Distribution", group_by_exposure=group_by_exposure)

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


def _render_tab_distribution(filtered_df):
    """Render case distribution statistics with interactive bar charts."""
    st.subheader("Case Distribution Analysis")
    
    # Filter options for distribution views
    st.markdown("### Distribution Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        show_arrhythmia_only = st.checkbox("Show only arrhythmic cases", value=False, key="dist_arrhythmia_filter")
    with col2:
        show_percentages = st.checkbox("Show percentages on bars", value=True, key="dist_show_pct")
    
    # Apply arrhythmia filter if selected
    display_df = filtered_df.copy()
    if show_arrhythmia_only:
        if "Arrhymia" in display_df.columns:
            display_df = display_df[display_df["Arrhymia"] == True]
            st.info(f"Showing {len(display_df)} arrhythmic cases out of {len(filtered_df)} total cases")
        else:
            st.warning("Arrhythmia column not found in data")
    
    if len(display_df) == 0:
        st.warning("No cases match the current filters.")
        return
    
    # 1. Distribution by Exposure Type
    st.markdown("### Distribution by Exposure Type")
    if "exposure" in display_df.columns:
        exposure_counts = display_df["exposure"].value_counts().reset_index()
        exposure_counts.columns = ["Exposure", "Count"]
        exposure_counts["Percentage"] = (exposure_counts["Count"] / exposure_counts["Count"].sum() * 100).round(1)
        
        # Map exposure colors using EXPOSURE_COLORS constant
        color_map = {
            "Phe": EXPOSURE_COLORS["Phe"]["primary"],
            "Terf": EXPOSURE_COLORS["Terf"]["primary"],
            "0": EXPOSURE_COLORS["0"]["primary"]
        }
        
        fig1 = px.bar(
            exposure_counts,
            x="Exposure",
            y="Count",
            title="Number of Cases by Exposure Type",
            color="Exposure",
            text="Count" if not show_percentages else exposure_counts.apply(
                lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1
            ),
            color_discrete_map=color_map
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        with st.expander("Show data table"):
            st.dataframe(exposure_counts, use_container_width=True)
    else:
        st.info("Exposure column not found in data")
    
    # 2. Distribution by Concentration Level
    st.markdown("### Distribution by Concentration Level")
    if "concentration" in display_df.columns:
        # Sort concentrations numerically
        conc_counts = display_df["concentration"].value_counts().reset_index()
        conc_counts.columns = ["Concentration", "Count"]
        conc_counts["Percentage"] = (conc_counts["Count"] / conc_counts["Count"].sum() * 100).round(1)
        
        # Try to sort numerically
        try:
            conc_counts["Concentration_num"] = pd.to_numeric(conc_counts["Concentration"])
            conc_counts = conc_counts.sort_values("Concentration_num")
            conc_counts = conc_counts.drop("Concentration_num", axis=1)
        except:
            conc_counts = conc_counts.sort_values("Concentration")
        
        fig2 = px.bar(
            conc_counts,
            x="Concentration",
            y="Count",
            title="Number of Cases by Concentration Level",
            color="Count",
            text="Count" if not show_percentages else conc_counts.apply(
                lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1
            ),
            color_continuous_scale="Blues"
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        with st.expander("Show data table"):
            st.dataframe(conc_counts, use_container_width=True)
    else:
        st.info("Concentration column not found in data")
    
    # 3. Distribution by Exposure + Concentration Combination
    st.markdown("### Distribution by Exposure × Concentration")
    if "exposure" in display_df.columns and "concentration" in display_df.columns:
        combo_counts = display_df.groupby(["exposure", "concentration"]).size().reset_index(name="Count")
        combo_counts["Percentage"] = (combo_counts["Count"] / combo_counts["Count"].sum() * 100).round(1)
        combo_counts["Label"] = combo_counts["exposure"] + "_" + combo_counts["concentration"].astype(str)
        
        # Sort by exposure then concentration
        try:
            combo_counts["concentration_num"] = pd.to_numeric(combo_counts["concentration"])
            combo_counts = combo_counts.sort_values(["exposure", "concentration_num"])
            combo_counts = combo_counts.drop("concentration_num", axis=1)
        except:
            combo_counts = combo_counts.sort_values(["exposure", "concentration"])
        
        # Use EXPOSURE_COLORS for consistent color coding
        color_map = {
            "Phe": EXPOSURE_COLORS["Phe"]["primary"],
            "Terf": EXPOSURE_COLORS["Terf"]["primary"],
            "0": EXPOSURE_COLORS["0"]["primary"]
        }
        
        fig3 = px.bar(
            combo_counts,
            x="Label",
            y="Count",
            title="Number of Cases by Exposure and Concentration",
            color="exposure",
            text="Count" if not show_percentages else combo_counts.apply(
                lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1
            ),
            labels={"Label": "Exposure_Concentration"},
            color_discrete_map=color_map
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)
        
        with st.expander("Show data table"):
            st.dataframe(combo_counts.drop("Label", axis=1), use_container_width=True)
    else:
        st.info("Exposure and/or concentration columns not found in data")
    
    # 4. Distribution by Well
    st.markdown("### Distribution by Well")
    if "well" in display_df.columns:
        well_counts = display_df["well"].value_counts().reset_index()
        well_counts.columns = ["Well", "Count"]
        well_counts["Percentage"] = (well_counts["Count"] / well_counts["Count"].sum() * 100).round(1)
        well_counts = well_counts.sort_values("Well")
        
        fig4 = px.bar(
            well_counts,
            x="Well",
            y="Count",
            title="Number of Cases by Well",
            color="Count",
            text="Count" if not show_percentages else well_counts.apply(
                lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1
            ),
            color_continuous_scale="Greens"
        )
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        
        with st.expander("Show data table"):
            st.dataframe(well_counts, use_container_width=True)
    else:
        st.info("Well column not found in data")
    
    # 5. Arrhythmia Distribution (if available)
    st.markdown("### Arrhythmia Status Distribution")
    if "Arrhymia" in display_df.columns:
        arrhythmia_counts = display_df["Arrhymia"].value_counts().reset_index()
        arrhythmia_counts.columns = ["Arrhythmia", "Count"]
        arrhythmia_counts["Arrhythmia"] = arrhythmia_counts["Arrhythmia"].map({True: "Arrhythmic", False: "Normal"})
        arrhythmia_counts["Percentage"] = (arrhythmia_counts["Count"] / arrhythmia_counts["Count"].sum() * 100).round(1)
        
        fig5 = px.bar(
            arrhythmia_counts,
            x="Arrhythmia",
            y="Count",
            title="Distribution of Arrhythmia Status",
            color="Arrhythmia",
            text="Count" if not show_percentages else arrhythmia_counts.apply(
                lambda row: f"{row['Count']} ({row['Percentage']:.1f}%)", axis=1
            ),
            color_discrete_map={"Arrhythmic": "#ff6b6b", "Normal": "#51cf66"}
        )
        fig5.update_traces(textposition="outside")
        fig5.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig5, use_container_width=True)
        
        with st.expander("Show data table"):
            st.dataframe(arrhythmia_counts, use_container_width=True)
        
        # Arrhythmia by exposure
        st.markdown("### Arrhythmia Rate by Exposure")
        if "exposure" in display_df.columns:
            arrhythmia_by_exposure = display_df.groupby("exposure")["Arrhymia"].agg([
                ("Total", "count"),
                ("Arrhythmic", "sum"),
            ]).reset_index()
            arrhythmia_by_exposure["Arrhythmia_Rate_%"] = (
                arrhythmia_by_exposure["Arrhythmic"] / arrhythmia_by_exposure["Total"] * 100
            ).round(1)
            
            # Use EXPOSURE_COLORS for consistent color coding
            color_map = {
                "Phe": EXPOSURE_COLORS["Phe"]["primary"],
                "Terf": EXPOSURE_COLORS["Terf"]["primary"],
                "0": EXPOSURE_COLORS["0"]["primary"]
            }
            
            fig6 = px.bar(
                arrhythmia_by_exposure,
                x="exposure",
                y="Arrhythmia_Rate_%",
                title="Arrhythmia Rate by Exposure Type (%)",
                color="exposure",
                text="Arrhythmia_Rate_%",
                labels={"exposure": "Exposure", "Arrhythmia_Rate_%": "Arrhythmia Rate (%)"},
                color_discrete_map=color_map
            )
            fig6.update_traces(textposition="outside", texttemplate='%{text:.1f}%')
            fig6.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig6, use_container_width=True)
            
            with st.expander("Show data table"):
                st.dataframe(arrhythmia_by_exposure, use_container_width=True)
        
        # Arrhythmia by concentration
        st.markdown("### Arrhythmia Rate by Concentration")
        if "concentration" in display_df.columns:
            arrhythmia_by_conc = display_df.groupby("concentration")["Arrhymia"].agg([
                ("Total", "count"),
                ("Arrhythmic", "sum"),
            ]).reset_index()
            arrhythmia_by_conc["Arrhythmia_Rate_%"] = (
                arrhythmia_by_conc["Arrhythmic"] / arrhythmia_by_conc["Total"] * 100
            ).round(1)
            
            # Sort by concentration
            try:
                arrhythmia_by_conc["concentration_num"] = pd.to_numeric(arrhythmia_by_conc["concentration"])
                arrhythmia_by_conc = arrhythmia_by_conc.sort_values("concentration_num")
                arrhythmia_by_conc = arrhythmia_by_conc.drop("concentration_num", axis=1)
            except:
                arrhythmia_by_conc = arrhythmia_by_conc.sort_values("concentration")
            
            fig7 = px.bar(
                arrhythmia_by_conc,
                x="concentration",
                y="Arrhythmia_Rate_%",
                title="Arrhythmia Rate by Concentration Level (%)",
                color="Arrhythmia_Rate_%",
                text="Arrhythmia_Rate_%",
                labels={"concentration": "Concentration", "Arrhythmia_Rate_%": "Arrhythmia Rate (%)"},
                color_continuous_scale="Reds"
            )
            fig7.update_traces(textposition="outside", texttemplate='%{text:.1f}%')
            fig7.update_layout(height=500)
            st.plotly_chart(fig7, use_container_width=True)
            
            with st.expander("Show data table"):
                st.dataframe(arrhythmia_by_conc, use_container_width=True)
    else:
        st.info("Arrhythmia column not found in data")
    
    # Summary statistics
    st.markdown("### Summary Statistics")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Total Cases", len(display_df))
    if "exposure" in display_df.columns:
        summary_cols[1].metric("Exposure Types", display_df["exposure"].nunique())
    if "concentration" in display_df.columns:
        summary_cols[2].metric("Concentration Levels", display_df["concentration"].nunique())
    if "Arrhymia" in display_df.columns:
        arrhythmia_rate = (display_df["Arrhymia"].sum() / len(display_df) * 100)
        summary_cols[3].metric("Overall Arrhythmia Rate", f"{arrhythmia_rate:.1f}%")


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


def _render_tab_waveform_overlay(record_by_sample, all_samples):
    """Render tab for overlaying contraction waveforms from selected samples with alignment and detrending."""
    st.subheader("Waveform Overlay Utility")
    st.markdown(
        "Select multiple samples to overlay their contraction waveforms. "
        "Waveforms are aligned by their first detected heartbeat peak and baseline-corrected "
        "for visual comparison of cardiac contraction patterns across different samples."
    )
    
    # Sample selection widget
    selected_samples = st.multiselect(
        "Select samples to overlay",
        options=sorted(all_samples),
        default=[],
        help="Choose multiple samples to compare their contraction waveforms"
    )
    
    # Processing options
    col1, col2 = st.columns(2)
    with col1:
        apply_alignment = st.checkbox(
            "Align by first peak", 
            value=True, 
            help="Align all waveforms so their first detected peak occurs at time=0"
        )
    with col2:
        apply_detrend = st.checkbox(
            "Remove baseline drift", 
            value=True, 
            help="Apply linear detrending to remove baseline wandering"
        )
    
    if not selected_samples:
        st.info("Please select at least one sample to display waveforms.")
        return
    
    if len(selected_samples) > 20:
        st.warning("You've selected more than 20 samples. The plot may be cluttered.")
    
    # Load and process data for selected samples
    import plotly.graph_objects as go
    
    fig = go.Figure()
    summary_data = []
    color_idx = 0
    
    # Define color palette (expand beyond EXPOSURE_COLORS)
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
    ]
    
    for sample_id in selected_samples:
        if sample_id not in record_by_sample:
            st.warning(f"Sample {sample_id} not found in loaded data.")
            continue
            
        record = record_by_sample[sample_id]
        time_ms = np.asarray(record.get("time_ms", []), dtype=float)
        contraction_values = np.asarray(record.get("contraction_values", []), dtype=float).copy()
        peak_indices = np.asarray(record.get("peak_indices", []), dtype=int)
        
        if len(time_ms) < 2 or len(contraction_values) < 2:
            st.warning(f"Insufficient data for sample {sample_id}")
            continue
        
        # Extract exposure from sample_id for color mapping
        exposure = sample_id.split("_")[0] if "_" in sample_id else None
        
        # Determine color
        if exposure and exposure in EXPOSURE_COLORS:
            color = EXPOSURE_COLORS[exposure]["primary"]
        else:
            color = color_palette[color_idx % len(color_palette)]
            color_idx += 1
        
        # Apply baseline correction (detrending)
        if apply_detrend:
            contraction_detrended = detrend(contraction_values, type='linear')
        else:
            contraction_detrended = contraction_values
        
        # Apply peak alignment
        if apply_alignment and len(peak_indices) > 0:
            # Align by first peak
            first_peak_idx = peak_indices[0]
            first_peak_time = time_ms[first_peak_idx]
            aligned_time_ms = time_ms - first_peak_time
        else:
            # Just normalize to start at 0
            aligned_time_ms = time_ms - time_ms[0]
        
        # Convert to seconds for display
        time_s = aligned_time_ms / 1000.0
        
        # Add trace to plot
        fig.add_trace(go.Scatter(
            x=time_s,
            y=contraction_detrended,
            mode='lines',
            name=sample_id,
            line=dict(width=1.5, color=color),
            opacity=0.7
        ))
        
        # Collect summary statistics
        summary_data.append({
            "Sample": sample_id,
            "Exposure": exposure if exposure else "N/A",
            "Num Peaks": len(peak_indices),
            "Mean Amplitude": f"{np.mean(contraction_detrended):.2f}",
            "Std Amplitude": f"{np.std(contraction_detrended):.2f}",
            "Min Amplitude": f"{np.min(contraction_detrended):.2f}",
            "Max Amplitude": f"{np.max(contraction_detrended):.2f}",
            "Duration (s)": f"{(time_ms[-1] - time_ms[0]) / 1000.0:.2f}",
            "Data Points": len(contraction_values)
        })
    
    # Update layout
    title = "Overlaid Contraction Waveforms"
    if apply_alignment and apply_detrend:
        title += " (Peak-aligned & Detrended)"
    elif apply_alignment:
        title += " (Peak-aligned)"
    elif apply_detrend:
        title += " (Detrended)"
    
    x_axis_label = "Time relative to first peak (s)" if apply_alignment else "Time (s)"
    y_axis_label = "Detrended Contraction (a.u.)" if apply_detrend else "Contraction Amplitude (a.u.)"
    
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary statistics
    if summary_data:
        st.markdown("### Summary Statistics")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


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
    
    # GroupBy Exposure toggle
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Display Options")
    group_by_exposure = st.sidebar.checkbox(
        "Group by Exposure",
        value=False,
        help="When enabled, show separate side-by-side graphs for each exposure (Phe and Terf)"
    )
    
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
            ("Distribution", "distribution"),
            ("Waveform Overlay", "waveform_overlay"),
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
            _render_tab_dose(filtered_records, group_by_exposure=group_by_exposure)

    if "hrv" in tab_by_key:
        with tab_by_key["hrv"]:
            _render_tab_hrv(filtered_records, group_by_exposure=group_by_exposure)
            
    if "amplitude" in tab_by_key:
        with tab_by_key["amplitude"]:
            _render_tab_contraction_amplitude(filtered_df, group_by_exposure=group_by_exposure)
            
    if "force" in tab_by_key:
        with tab_by_key["force"]:
            _render_tab_contraction_force(filtered_df, selected_sample, record_by_sample, group_by_exposure=group_by_exposure)
            
    if "transients" in tab_by_key:
        with tab_by_key["transients"]:
            _render_tab_transients(filtered_df, group_by_exposure=group_by_exposure)

    if "data_table" in tab_by_key:
        with tab_by_key["data_table"]:
            _render_tab_data_table(filtered_df)
            
    if "statistical_analysis" in tab_by_key:
        with tab_by_key["statistical_analysis"]:
            _render_tab_statistical_analysis(filtered_df)
            
    if "graphs" in tab_by_key:
        with tab_by_key["graphs"]:
            _render_tab_graphs(filtered_df, group_by_exposure=group_by_exposure)

    if "distribution" in tab_by_key:
        with tab_by_key["distribution"]:
            _render_tab_distribution(filtered_df)

    if "waveform_overlay" in tab_by_key:
        with tab_by_key["waveform_overlay"]:
            all_samples = list(record_by_sample.keys())
            _render_tab_waveform_overlay(record_by_sample, all_samples)

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
