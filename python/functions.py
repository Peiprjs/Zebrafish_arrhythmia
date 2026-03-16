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
