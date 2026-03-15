"""Streamlit dashboard for zebrafish contraction and HRV exploration.

Usage:
    streamlit run gui_app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_analysis import SUMMARY_COLUMNS, load_all_sample_timeseries


INTERP_POINTS = 300


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


@st.cache_data(show_spinner=False)
def _load_dashboard_data(results_dir):
    records = load_all_sample_timeseries(results_dir, verbose=False)
    summary_df = pd.DataFrame(records)[SUMMARY_COLUMNS].copy()
    return records, summary_df


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

    st.sidebar.header("Data source")
    results_dir = st.sidebar.text_input("MM_Results folder", default_results)
    if st.sidebar.button("Reload data"):
        st.cache_data.clear()

    results_dir = os.path.abspath(results_dir)
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

    tab_fish, tab_dose, tab_hrv, tab_metrics = st.tabs([
        "Fish profile",
        "Dose profile",
        "HRV over time",
        "All variables",
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
            default_metric = "arrhythmia_probability" if "arrhythmia_probability" in numeric_cols else numeric_cols[0]
            metric = st.selectbox("Metric distribution", numeric_cols, index=numeric_cols.index(default_metric))
            group_by = st.selectbox("Group by", ["exposure", "concentration", "Arrhymia"])

            grouped_values = []
            labels = []
            for name, group in filtered_df.groupby(group_by):
                values = group[metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                grouped_values.append(values)
                labels.append(str(name))

            if grouped_values:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.boxplot(grouped_values, tick_labels=labels, showmeans=True)
                ax.set_title(f"{metric} grouped by {group_by}")
                ax.set_ylabel(metric)
                ax.grid(axis="y", alpha=0.2)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


if __name__ == "__main__":
    main()
