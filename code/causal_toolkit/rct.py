from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _split_groups(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract outcome arrays for treated and control groups.

    Parameters
    ----------
    df:
        DataFrame containing at least:
        - ``T``: binary treatment indicator (0/1)
        - ``Y``: observed outcome

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(treat, control)`` outcome arrays as ``float`` ndarrays.

    Raises
    ------
    ValueError
        If required columns are missing or either group is empty.
    """
    # Minimal schema required by downstream Welch computations.
    required_cols = {"T", "Y"}
    missing = required_cols - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")

    # Convert to float arrays to ensure numeric stability and consistency.
    treat = df.loc[df["T"] == 1, "Y"].to_numpy(dtype=float)
    control = df.loc[df["T"] == 0, "Y"].to_numpy(dtype=float)
    if treat.size == 0 or control.size == 0:
        raise ValueError("Both treatment and control groups must be non-empty.")
    return treat, control


def _welch_stats(treat: np.ndarray, control: np.ndarray) -> Tuple[float, float, float]:
    """Compute Welch-style ATE summary statistics.

    Parameters
    ----------
    treat:
        Outcomes for treated units.
    control:
        Outcomes for control units.

    Returns
    -------
    Tuple[float, float, float]
        ``(ate, se, degrees_freedom)`` where:
        - ``ate`` is the mean difference ``mean_t - mean_c``
        - ``se`` is the Welch standard error
        - ``degrees_freedom`` is Welch-Satterthwaite df (or ``inf``)
    """
    # Group sizes and moments.
    n_t = treat.size
    n_c = control.size
    mean_t = float(np.mean(treat))
    mean_c = float(np.mean(control))
    # Use unbiased sample variance (ddof=1); default to 0 when only one sample.
    var_t = float(np.var(treat, ddof=1)) if n_t > 1 else 0.0
    var_c = float(np.var(control, ddof=1)) if n_c > 1 else 0.0

    # ATE estimate and its standard error under unequal variances.
    ate = mean_t - mean_c
    se = float(np.sqrt(var_t / n_t + var_c / n_c))

    # Zero SE means no sampling variability in this setup.
    if se == 0.0:
        return ate, se, float("inf")

    # Welch-Satterthwaite approximation for effective degrees of freedom.
    numerator = (var_t / n_t + var_c / n_c) ** 2
    denominator = (var_t / n_t) ** 2 / (n_t - 1) + (var_c / n_c) ** 2 / (n_c - 1)
    degrees_freedom = float("inf") if denominator == 0.0 else float(numerator / denominator)
    return ate, se, degrees_freedom


def calculate_ate_ci(data: pd.DataFrame, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Estimate ATE and a two-sided confidence interval.

    Parameters
    ----------
    data:
        DataFrame with columns ``T`` (binary treatment) and ``Y`` (outcome).
    alpha:
        Significance level for a two-sided interval (default 0.05 for 95% CI).

    Returns
    -------
    Tuple[float, float, float]
        ``(ate, lower, upper)`` confidence-interval tuple.

    Notes
    -----
    Uses a normal critical value (z-interval). This is a practical large-sample
    approximation for randomized experiments.
    """
    # Split outcomes by assignment group, then compute Welch summary stats.
    treat, control = _split_groups(data)
    ate, se, _ = _welch_stats(treat, control)

    # Degenerate case: no variability => point-mass interval at ATE.
    if se == 0.0:
        return ate, ate, ate

    # Standard normal critical value for two-sided CI.
    critical_value = float(stats.norm.ppf(1 - alpha / 2))
    lower = ate - critical_value * se
    upper = ate + critical_value * se
    return ate, lower, upper


def calculate_ate_pvalue(data: pd.DataFrame) -> Tuple[float, float, float]:
    """Estimate ATE, z-style test statistic, and two-sided p-value.

    Parameters
    ----------
    data:
        DataFrame with columns ``T`` (binary treatment) and ``Y`` (outcome).

    Returns
    -------
    Tuple[float, float, float]
        ``(ate, t_stat, p_value)`` where ``t_stat`` is ``ate / se``.

    Notes
    -----
    Despite the ``t_stat`` name, inference here uses the standard normal CDF.
    """
    # Group split and Welch-style standard error.
    treat, control = _split_groups(data)
    ate, se, _ = _welch_stats(treat, control)

    # Handle zero-SE edge cases explicitly.
    if se == 0.0:
        if ate == 0.0:
            return ate, 0.0, 1.0
        return ate, float("inf"), 0.0

    # Two-sided normal-approximation p-value.
    t_stat = ate / se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    return ate, float(t_stat), float(p_value)