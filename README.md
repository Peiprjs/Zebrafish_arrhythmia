# Zebrafish Cardiac Motion Analysis

This repository analyzes zebrafish MUSCLEMOTION results to summarize heartbeat variability and estimate arrhythmia likelihood for each sample.

## What this code does

- Reads sample folders from `MM_Results/` (for example `Phe_50_1.1-Contr-Results`).
- Uses each sample’s `contraction.txt` signal to detect heartbeat peaks.
- Measures time between beats and computes summary metrics.
- Writes results to `output/hrv_summary.csv` and saves per-sample plots in `output/`.
- Provides an optional Streamlit dashboard (`python/gui_app.py`) for interactive review.
- Runs additional statistical tests and writes regression summaries to CSV files in `output/`.

## How arrhythmia risk is estimated

The script creates a heuristic `arrhythmia_risk_score` between **0** and **1** from heartbeat timing irregularity:

- **Overall irregularity**: how spread out the beat-to-beat timings are.
- **Peak-to-peak interval**: how much neighboring beats change from one beat to the next.
- **Unusual beats**: how often beat intervals are far from the sample’s typical interval.

Each of these three signals is converted to a 0-to-1 score and then averaged into one final risk score.

- A value closer to **1** means more irregular rhythm.
- A value closer to **0** means more regular rhythm.
- `arrhythmia_probability` is kept as a compatibility alias of `arrhythmia_risk_score`.
- The `Arrhymia` column is `True` when score `> arrhythmia_threshold` (default: `0.5`).

### Data sufficiency and quality flags

The output now includes:

- `arrhythmia_data_sufficient` (`True/False`)
- `arrhythmia_quality_flag` (`ok`, `low_ibi_count`, or `insufficient_ibi`)
- `arrhythmia_ibi_count` (number of inter-beat intervals used)
- `paired_ttest_pvalue_vs_control0_mean_ibi` (paired t-test p-value comparing each sample's IBI profile with the mean concentration-0 control profile within the same exposure)

If there are too few IBIs to score reliably, the risk score is set to `NaN` and `Arrhymia` is conservatively set to `False`.

> This output is a research-oriented signal quality/rhythm irregularity estimate, not a clinical diagnosis.

## Additional statistical outputs

After each run, the script also writes:

- `output/linear_regression_coefficients.csv`
- `output/linear_regression_summary.csv`
- `output/logistic_regression_coefficients.csv`
- `output/logistic_regression_summary.csv`
- `output/linear_trend_summary.csv`
- `output/unsupervised_assignments.csv`
- `output/unsupervised_cluster_summary.csv`
- `output/unsupervised_kmeans_centers.csv`
- `output/unsupervised_pca_loadings.csv`
- `output/unsupervised_pca_variance.csv`

These summarize linear/logistic regression analyses and unsupervised model outputs (KMeans, hierarchical clustering, PCA) based on the HRV/risk feature set.

## How to run

From `python/` (inside a venv or Conda environment):

```bash
pip install -r requirements.txt
python data_analysis.py --results_dir ../MM_Results --output_dir ../output --arrhythmia_threshold 0.5
```

Optional dashboard:

```bash
streamlit run gui_app.py
```

The dashboard includes a **Model summaries** tab that reads regression and unsupervised-learning CSV outputs from the configured output folder.
