"""Streamlit dashboard for zebrafish contraction and HRV exploration.

Usage:
    streamlit run gui_app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ttest_rel

from data_analysis import SUMMARY_COLUMNS, load_all_sample_timeseries


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


def _safe_float_sort_key(value):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _group_label(record):
    return f"{record['exposure']}_{record['concentration']}"


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
    for record in records:
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

    def _paired_ttest_vs_reference(values, reference_values):
        values = np.asarray(values, dtype=float)
        reference_values = np.asarray(reference_values, dtype=float)
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

    if group_by == "concentration":
        baseline_index = next((idx for idx, label in enumerate(labels) if str(label) == "0"), None)
        if baseline_index is None:
            return [False] * len(values_list), [np.nan] * len(values_list), "* paired t-test p < 0.05 vs concentration 0 mean"

        baseline_values = values_list[baseline_index]
        for idx, values in enumerate(values_list):
            if idx == baseline_index or len(values) < 2 or len(baseline_values) < 2:
                significance_flags.append(False)
                p_values.append(np.nan)
                continue

            p_value = _paired_ttest_vs_reference(values, baseline_values)
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

        p_value = _paired_ttest_vs_reference(values, rest)
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
    time_s = np.asarray(record["time_ms"], dtype=float) / 1000.0
    contraction = np.asarray(record["contraction_values"], dtype=float)
    peak_indices = np.asarray(record["peak_indices"], dtype=int)
    peak_times_s = np.asarray(record["peak_times_ms"], dtype=float) / 1000.0

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(time_s, contraction, linewidth=0.8, label="Contraction")
    if len(peak_indices) > 0:
        ax.plot(peak_times_s, contraction[peak_indices], "rv", markersize=6, label="Peaks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contraction (a.u.)")
    ax.set_title(f"Contraction profile – {record['sample']}")
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

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(speed_time / 1000.0, speed_values, linewidth=0.8, color="tab:orange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (a.u.)")
    ax.set_title(f"Speed-of-contraction profile – {record['sample']}")
    ax.grid(alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def _plot_fish_hrv(record):
    ibi_time_s = np.asarray(record["ibi_time_ms"], dtype=float) / 1000.0
    ibi_values = np.asarray(record["ibi_values_ms"], dtype=float)
    rolling_time_s = np.asarray(record["rolling_rmssd_time_ms"], dtype=float) / 1000.0
    rolling_values = np.asarray(record["rolling_rmssd_ms"], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=False)

    if len(ibi_time_s) > 0:
        ax1.plot(ibi_time_s, ibi_values, "o-", markersize=3.5, linewidth=1.0, label="IBI")
        ax1.set_ylabel("IBI (ms)")
        ax1.set_title("Inter-beat intervals over time")
        ax1.grid(alpha=0.2)

    if len(rolling_time_s) > 0:
        ax2.plot(rolling_time_s, rolling_values, "o-", markersize=3.5, linewidth=1.0, color="tab:green")
        ax2.set_ylabel("Rolling RMSSD (ms)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("HRV over time (rolling RMSSD)")
        ax2.grid(alpha=0.2)
    else:
        ax2.text(0.02, 0.5, "Not enough beats for rolling RMSSD", transform=ax2.transAxes)
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

    ax.set_xlabel("Relative recording time (%)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Zebrafish HRV Dashboard", layout="wide")
    st.title("Zebrafish contraction and HRV dashboard")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.abspath(os.path.join(script_dir, "..", "MM_Results"))
    default_output = os.path.abspath(os.path.join(script_dir, "..", "output"))

    results_dir = default_results
    output_dir = default_output
    with st.sidebar.expander("Data source (advanced)", expanded=False):
        results_dir = st.text_input("MM_Results folder", default_results)
        output_dir = st.text_input("Output folder", default_output)
        if st.button("Reload data"):
            st.cache_data.clear()

    results_dir = os.path.abspath(results_dir)
    output_dir = os.path.abspath(output_dir)
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

    if filtered_df.empty:
        st.warning("No samples match the selected filters.")
        st.stop()

    sample_options = filtered_df["sample"].tolist()
    selected_sample = st.sidebar.selectbox("Fish sample", sample_options)

    filtered_records = [record_by_sample[sample] for sample in sample_options if sample in record_by_sample]

    c1, c2, c3 = st.columns(3)
    c1.metric("Fish samples", len(filtered_df))
    c2.metric("Exposure groups", int(filtered_df["exposure"].nunique()))
    c3.metric("Dose groups", int(filtered_df["concentration"].nunique()))

    tab_fish, tab_dose, tab_hrv, tab_metrics, tab_models = st.tabs([
        "Fish profile",
        "Dose profile",
        "HRV over time",
        "All variables",
        "Model summaries",
    ])

    with tab_fish:
        fish_record = record_by_sample[selected_sample]
        _plot_fish_profile(fish_record)
        _plot_speed_profile(fish_record)
        _plot_fish_hrv(fish_record)

    with tab_dose:
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
                title="Average contraction profile by exposure/dose",
                y_label="Contraction (a.u.)",
            )

    with tab_hrv:
        grid_pct, aggregate = _aggregate_group_profiles(
            filtered_records,
            time_key="rolling_rmssd_time_ms",
            value_key="rolling_rmssd_ms",
        )
        if not aggregate:
            st.info("Not enough beats to compute rolling RMSSD trends for selected samples.")
        else:
            _plot_group_aggregate(
                grid_pct,
                aggregate,
                title="Average HRV (rolling RMSSD) over relative time",
                y_label="Rolling RMSSD (ms)",
            )

    with tab_metrics:
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

        numeric_cols = [
            col for col in filtered_df.columns
            if pd.api.types.is_numeric_dtype(filtered_df[col])
        ]
        if numeric_cols:
            if "arrhythmia_risk_score" in numeric_cols:
                default_metric = "arrhythmia_risk_score"
            elif "arrhythmia_probability" in numeric_cols:
                default_metric = "arrhythmia_probability"
            else:
                default_metric = numeric_cols[0]
            metric = st.selectbox("Metric distribution", numeric_cols, index=numeric_cols.index(default_metric))
            group_by = st.selectbox("Group by", ["exposure", "concentration"])

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
                significance_flags, _, significance_label = _group_significance(
                    grouped_values, labels, group_by, alpha=0.05
                )
                display_labels = [
                    f"{label}*" if is_significant else label
                    for label, is_significant in zip(labels, significance_flags)
                ]
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
                ax.set_title(f"{metric} grouped by {group_by}")
                ax.set_ylabel(metric)
                ax.grid(axis="y", alpha=0.2)
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

    with tab_models:
        st.caption(f"Model outputs loaded from `{output_dir}`")
        tables = _load_model_output_tables(output_dir)

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


if __name__ == "__main__":
    main()
