import os
import re
import numpy as np
from scipy.stats import ttest_rel

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

def format_sample_label(meta):
    """Build canonical sample label from parsed folder metadata."""
    return f"{meta['exposure']}_{meta['concentration']}_{meta['well']}.{meta['fish']}"

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



