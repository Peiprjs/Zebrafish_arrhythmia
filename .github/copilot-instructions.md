# Copilot Instructions for Zebrafish Arrhythmia Analysis

## Project Overview

This is a research data analysis project for zebrafish cardiac motion. It processes output from MUSCLEMOTION software, detects heartbeat peaks in contraction signals, computes heart rate variability (HRV) metrics, and estimates arrhythmia risk. The codebase includes batch analysis scripts and an interactive Streamlit dashboard.

## Build, Test & Run

### Environment Setup

Always run Python tasks inside a **venv** or **Anaconda environment**. From the `python/` directory:

```bash
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Running Analysis

From `python/`:

```bash
# Full batch analysis (requires MM_Results/ folder with MUSCLEMOTION output)
python data_analysis.py --results_dir ../MM_Results --output_dir ../output --arrhythmia_threshold 0.5

# Run with specific excluded samples
python data_analysis.py --results_dir ../MM_Results --output_dir ../output --exclude_samples Phe_100_2.1,Terf_30_1.2

# Run with help to see all arguments
python data_analysis.py --help
```

### Running Dashboard

```bash
streamlit run gui_app.py
```

The dashboard loads results from the configured output folder and includes tabs for sample exploration, per-sample timeseries, statistical models (regression, clustering, PCA), and variable glossary.

### No formal test suite

There are no automated tests. Validation is done by running the full pipeline and checking output CSV files and plots.

## Architecture & Key Concepts

### Data Flow

1. **Input**: MUSCLEMOTION result folders named `{Exposure}_{Conc}_{Well}.{Fish}-Contr-Results`
   - Each contains `contraction.txt` (time vs. signal) and `speed-of-contraction.txt` TSV files
   - Data recorded at ~60 FPS (16.6667 ms per sample)

2. **Peak Detection** (`detect_peaks()`): Finds heartbeat peaks in contraction signal
   - Uses prominence-based filtering (not simple threshold)
   - Groups closely spaced peaks as "sawtooth" events (peaks within `sawtooth_group_distance_ms`)
   - Returns `(peak_indices, peak_times_ms, sawtooth_peak)` tuple where `sawtooth_peak` is boolean array
   - Estimates signal quality (SNR, baseline drift)

3. **HRV Computation**: From detected peak times, computes:
   - **IBI** (inter-beat interval): time between consecutive peaks
   - **SDNN**: standard deviation of IBI
   - **RMSSD**: root mean square of successive IBI differences
   - **pNN50**: percent of IBI differences > 50 ms
   - **CV_IBI**: coefficient of variation (SDNN / mean_IBI)

4. **Arrhythmia Risk Score**: Combines three normalized 0–1 signals:
   - `arrhythmia_score_cv`: from IBI coefficient of variation
   - `arrhythmia_score_rmssd`: from RMSSD
   - `arrhythmia_score_outlier`: from fraction of outlier IBIs
   - Final score = average of the three; higher = more irregular

5. **Quality Flagging**: Checks if enough IBIs were detected:
   - `ARRHYTHMIA_MIN_IBI_FOR_SCORE = 3`: minimum for any score
   - `ARRHYTHMIA_MIN_IBI_FOR_CONFIDENT_DECISION = 6`: minimum for confident decision
   - Quality flags: `ok`, `low_ibi_count`, or `insufficient_ibi`

6. **Output**: 
   - Main: `output/hrv_summary.csv` (one row per sample)
   - Statistical models: regression/logistic regression coefficients, linear trends, cluster assignments, PCA loadings
   - Plots: one PNG per sample with contraction signal + detected peaks (including sawtooth peaks)

### Key Constants

Located at top of `data_analysis.py`:

- `ROLLING_RMSSD_WINDOW = 5`: window size for rolling RMSSD
- `ARRHYTHMIA_DEFAULT_THRESHOLD = 0.5`: cutoff for binary Arrhythmia decision
- `CONTROL_CONCENTRATION = "0"`: label for control samples
- `PEAK_PROMINENCE_OUTLIER_RATIO_THRESHOLD = 2.0`: filter for peak outliers
- `PEAK_PROMINENCE_TRIM_PERCENTILES = (1.0, 99.0)`: percentile range for valid peaks

### Folder Metadata Parsing

Sample metadata extracted from folder names via `parse_folder_name()`:

```
{Exposure}_{Concentration}_{Well}.{Fish}-Contr-Results
Example: Phe_50_1.1-Contr-Results
→ exposure='Phe', concentration='50', well='1', fish='1'
```

Canonical sample label: `{Exposure}_{Concentration}_{Well}.{Fish}` (e.g., `Phe_50_1.1`)

## Code Conventions

### Function Organization

- **Data loading** (lines ~71–120): folder parsing, TSV loading, result folder listing
- **Signal processing** (lines ~132–360): peak detection (including sawtooth flagging), signal quality metrics
- **HRV metrics** (lines ~361–432): IBI, SDNN, RMSSD, pNN50 computation
- **Arrhythmia scoring** (lines ~433–515): risk score, decision logic, logistic scoring helper
- **Statistical tests** (lines ~530–779): paired t-tests, regression fitting, unsupervised clustering
- **Visualization** (lines ~990–1039): contraction plots with peak and sawtooth peak annotations
- **Main pipeline** (lines ~1121–end): orchestrates folder iteration, calls all analyses, writes outputs

### Peak Detection & Sawtooth Handling

- `detect_peaks()` returns three values: `peak_indices`, `peak_times_ms`, `sawtooth_peak` (boolean array)
- `sawtooth_peak[i]` is `True` if peak `i` was part of a sawtooth event
- Sawtooth grouping uses `sawtooth_group_distance_ms` parameter (default: 500 ms)
- In plots, normal peaks are red inverted triangles (ˇ), sawtooth peaks are blue (ˇ)
- In Streamlit dashboard, sawtooth peaks are extracted from `record.get("sawtooth_peak", [])` and must be plotted separately

### Argument Handling

- Uses `argparse` for CLI arguments
- Main arguments: `--results_dir`, `--output_dir`, `--arrhythmia_threshold`, `--exclude_samples`
- Defaults: results_dir=`./MM_Results`, output_dir=`./output`

### Output Columns

`SUMMARY_COLUMNS` defines CSV output schema. Always include when writing `hrv_summary.csv`:
- Metadata: sample, exposure, concentration, well, fish
- Signal quality: n_peaks, snr, baseline_drift
- HRV metrics: mean_ibi_ms, sdnn_ms, rmssd_ms, cv_ibi, pnn50, mean_hr_bpm
- Arrhythmia scores: arrhythmia_risk_score, arrhythmia_probability (alias), component scores, Arrhymia (binary)
- Quality flags: arrhythmia_data_sufficient, arrhythmia_quality_flag, arrhythmia_ibi_count
- Statistical: paired_ttest_pvalue_vs_control0_mean_ibi, arrhythmia_threshold

### NumPy/SciPy Conventions

- Peak detection uses `scipy.signal.find_peaks(prominence=...)`
- Statistical tests use `scipy.stats` (e.g., `linregress`, `ttest_rel`)
- Clustering uses `scipy.cluster` (hierarchical linkage, KMeans via `kmeans2`)
- PCA is custom-implemented with eigenvalue decomposition (not scikit-learn)

### Pandas Usage

- DataFrames are primary data structure for results
- Sample metadata is always a column (not index)
- Use `fillna(np.nan)` consistently for missing values
- GroupBy operations group by (exposure, concentration, well) or similar as needed

### Error Handling

- Missing folders or bad TSV files are logged as warnings but don't halt execution
- Samples with insufficient data get NaN scores and conservative arrhythmia decisions
- No exceptions are explicitly raised in main pipeline; errors are logged and skipped

### Imports

Standard structure at top of files:
```python
import argparse, os, re, sys  # stdlib
from tqdm import tqdm         # progress bars
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import linregress, ttest_rel
# local imports after external libraries
```

## Streamlit Dashboard Conventions

- `gui_app.py` imports core functions from `data_analysis.py` (e.g., `SUMMARY_COLUMNS`, `load_all_sample_timeseries`)
- Sidebar controls for filtering (exposure, concentration, well, sample selection)
- Tabs for different views (timeseries, model summaries, variables glossary)
- Model outputs read from CSV files in output directory (lazy-loaded to avoid startup delays)
- Constants like `DEFAULT_EXCLUDED_CASES` and `MODEL_OUTPUT_FILES` defined at top
- Fish view plots both normal peaks and sawtooth peaks separately using different colors and legend

### Plotting in Dashboard

- `_plot_fish_profile()` extracts sawtooth info from `record.get("sawtooth_peak", [])`
- Checks if `sawtooth_peak.size == len(peak_indices)` before separating masks
- Plots normal peaks in red, sawtooth peaks in blue with separate legend entries
- Fallback to single peak series if sawtooth array size doesn't match

## Typical Editing Tasks

### Adding a new HRV metric

1. Add computation function in signal processing section
2. Add column name to `SUMMARY_COLUMNS`
3. Update DataFrame column assignment in sample processing loop
4. If adding to dashboard, update `gui_app.py` variable glossary

### Modifying peak detection

- Parameters in `detect_peaks()` function signature and constants section
- Always validate on sample folder (e.g., `Phe_50_1.1-Contr-Results`) by running full pipeline and checking PNG plot

### Adding a regression model

- Implement in statistical tests section (around line 780)
- Call from `run_additional_statistical_tests()`
- Write outputs to appropriately named CSV files in output_dir

### Updating Streamlit visualizations

- Remember that `load_all_sample_timeseries()` stores `sawtooth_peak` in the record dictionary
- When extracting peak data, always handle the case where `sawtooth_peak` may be missing or empty
- Use boolean masking to separate normal and sawtooth peaks for distinct visualization

## Dependencies

All Python dependencies in `requirements.txt`. Core libraries:
- **numpy**, **scipy**: numerical computing and signal processing
- **pandas**: data analysis and tabulation
- **matplotlib**: plotting
- **streamlit**: interactive dashboard
- **tqdm**: progress bars

Note: No scikit-learn or other ML frameworks; clustering and PCA are custom-implemented to minimize dependencies.
