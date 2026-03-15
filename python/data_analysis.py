"""
Heart Rate Variability (HRV) Analysis for Zebrafish MUSCLEMOTION Data
=====================================================================

Reads MUSCLEMOTION output folders from MM_Results/, extracts contraction
peaks, computes inter-beat intervals, instantaneous and global HRV metrics,
and estimates an arrhythmia probability for each sample.

Usage:
    python data_analysis.py [--results_dir PATH] [--output_dir PATH]
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


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


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def detect_peaks(time_ms, signal, prominence_factor=0.3, distance_ms=200.0):
    """Detect contraction peaks in the signal.

    Parameters
    ----------
    time_ms : array-like
        Time axis in milliseconds.
    signal : array-like
        Contraction signal values.
    prominence_factor : float
        Minimum peak prominence as a fraction of the signal range.
    distance_ms : float
        Minimum distance between peaks in milliseconds.

    Returns
    -------
    peak_indices : ndarray
        Indices of detected peaks.
    peak_times : ndarray
        Times (ms) of detected peaks.
    """
    dt = np.median(np.diff(time_ms))
    distance_samples = max(1, int(distance_ms / dt))
    prominence = prominence_factor * (np.max(signal) - np.min(signal))

    peak_indices, properties = find_peaks(
        signal, prominence=prominence, distance=distance_samples
    )
    return peak_indices, time_ms[peak_indices]


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


# ---------------------------------------------------------------------------
# Arrhythmia probability
# ---------------------------------------------------------------------------

def arrhythmia_probability(ibi_ms):
    """Estimate the probability that the contraction profile exhibits arrhythmia.

    The score is a composite of three normalized indicators, each mapped to [0, 1]
    via a logistic function and then averaged:

    1. **Coefficient of variation (CV)** of IBI – captures overall irregularity.
    2. **RMSSD / mean IBI** – captures beat-to-beat variability.
    3. **Outlier fraction** – proportion of IBIs that deviate from the median
       by more than 30%.

    Returns a float in [0, 1] where values closer to 1 indicate higher
    arrhythmia likelihood.
    """
    ibi = np.asarray(ibi_ms, dtype=float)
    if len(ibi) < 3:
        return 0.0

    def _logistic(x, midpoint, steepness):
        return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))

    # Indicator 1: coefficient of variation
    cv = np.std(ibi, ddof=1) / np.mean(ibi)
    score_cv = _logistic(cv, 0.15, 30)

    # Indicator 2: RMSSD relative to mean IBI
    successive_diff = np.diff(ibi)
    rmssd_rel = np.sqrt(np.mean(successive_diff ** 2)) / np.mean(ibi)
    score_rmssd = _logistic(rmssd_rel, 0.15, 30)

    # Indicator 3: fraction of outlier IBIs (>30 % from median)
    median_ibi = np.median(ibi)
    outlier_frac = np.mean(np.abs(ibi - median_ibi) / (median_ibi if median_ibi > 0 else 1.0) > 0.30)
    score_outlier = _logistic(outlier_frac, 0.15, 20)

    return float(np.mean([score_cv, score_rmssd, score_outlier]))


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_contraction_with_peaks(time_ms, signal, peak_indices, peak_times,
                                ibi_ms, sample_label, output_path):
    """Save a figure showing the contraction signal with detected peaks and IBIs."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Top: contraction + peaks
    ax = axes[0]
    ax.plot(time_ms / 1000, signal, linewidth=0.7, label="Contraction")
    ax.plot(peak_times / 1000, signal[peak_indices], "rv", markersize=8, label="Peaks")
    ax.set_ylabel("Contraction (a.u.)")
    ax.set_title(f"Contraction with Detected Peaks – {sample_label}")
    ax.legend(loc="upper right")

    # Bottom: IBI
    ax2 = axes[1]
    if len(ibi_ms) > 0:
        ax2.plot(peak_times[1:] / 1000, ibi_ms, "o-", markersize=4)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("IBI (ms)")
    ax2.set_title("Inter-Beat Intervals")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyse_sample(folder_path, output_dir=None):
    """Run the full HRV analysis on a single MUSCLEMOTION results folder.

    Returns a dict of results or None if the folder is not valid.
    """
    folder_name = os.path.basename(folder_path)
    meta = parse_folder_name(folder_name)
    if meta is None:
        return None

    contraction_file = os.path.join(folder_path, "contraction.txt")
    if not os.path.isfile(contraction_file):
        print(f"  [SKIP] No contraction.txt in {folder_name}")
        return None

    # Load contraction data
    time_ms, signal = load_tsv(contraction_file)

    # Detect peaks
    peak_indices, peak_times = detect_peaks(time_ms, signal)
    if len(peak_times) < 2:
        print(f"  [WARN] Fewer than 2 peaks detected in {folder_name}")
        return None

    # Inter-beat intervals
    ibi_ms = compute_ibi(peak_times)

    # HRV metrics
    hrv = compute_hrv_metrics(ibi_ms)

    # Arrhythmia probability
    arr_prob = arrhythmia_probability(ibi_ms)

    sample_label = f"{meta['exposure']}_{meta['concentration']}_{meta['well']}.{meta['fish']}"

    # Optional plot
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"{sample_label}_hrv.png")
        plot_contraction_with_peaks(
            time_ms, signal, peak_indices, peak_times, ibi_ms, sample_label, plot_path
        )

    return {
        "sample": sample_label,
        **meta,
        "n_peaks": len(peak_times),
        "ibi_values_ms": ibi_ms.tolist(),
        **hrv,
        "arrhythmia_probability": arr_prob,
    }


def run_analysis(results_dir, output_dir):
    """Iterate over all sample folders in *results_dir* and produce a summary."""
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    folders = sorted(
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.endswith("-Contr-Results")
    )

    if not folders:
        print(f"No MUSCLEMOTION result folders found in {results_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    for folder_name in folders:
        folder_path = os.path.join(results_dir, folder_name)
        print(f"Analysing {folder_name} ...")
        result = analyse_sample(folder_path, output_dir)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No valid results produced.")
        sys.exit(1)

    # Build summary DataFrame (exclude per-beat IBI lists for the CSV)
    summary_cols = [
        "sample", "exposure", "concentration", "well", "fish",
        "n_peaks", "mean_ibi_ms", "sdnn_ms", "rmssd_ms",
        "cv_ibi", "pnn50", "mean_hr_bpm", "arrhythmia_probability",
    ]
    df = pd.DataFrame(all_results)[summary_cols]
    csv_path = os.path.join(output_dir, "hrv_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
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
    args = parser.parse_args()

    run_analysis(
        os.path.abspath(args.results_dir),
        os.path.abspath(args.output_dir),
    )


if __name__ == "__main__":
    main()
