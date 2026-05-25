from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LinearRegression, LogisticRegression


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that all required columns exist in ``df``.

    Parameters
    ----------
    df:
        Input DataFrame to validate.
    columns:
        Column names that must be present.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    # Use set arithmetic so order does not matter.
    missing = set(columns) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _design_matrix(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Build a Patsy design matrix from a formula and DataFrame.

    The returned matrix can include transformed terms (splines, interactions,
    one-hot encodings, etc.) depending on the Patsy formula.
    """
    return dmatrix(formula, df, return_type="dataframe")


def _fit_propensity_model(X: pd.DataFrame, treatment: np.ndarray) -> LogisticRegression:
    """Fit logistic propensity model ``P(T=1 | X)``.

    Notes
    -----
    - ``fit_intercept=False`` assumes the design matrix already contains an
      intercept term when needed (which Patsy usually does by default).
    - The large ``C`` in the fallback branch approximates weak regularization.
    """
    # Start with a baseline configuration.
    base_model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        fit_intercept=False,
    )
    # Some sklearn configurations may permit unpenalized fitting directly.
    if base_model.penalty is None:
        model = base_model
    else:
        # Otherwise approximate no-penalty behavior with very weak L2 penalty.
        model = LogisticRegression(
            solver="lbfgs",
            C=1e6,
            max_iter=1000,
            fit_intercept=False,
        )
    # Fit the binary treatment model and return it for score prediction.
    model.fit(X, treatment)
    return model


def ipw(df: pd.DataFrame, ps_formula: str, T: str, Y: str) -> float:
    """Estimate ATE via inverse probability weighting (IPW).

    Parameters
    ----------
    df:
        Analysis dataset containing treatment, outcome, and covariates.
    ps_formula:
        Patsy formula used to build propensity-model covariates.
    T:
        Treatment-column name (binary: 0/1).
    Y:
        Outcome-column name.

    Returns
    -------
    float
        Estimated average treatment effect.

    Notes
    -----
    Uses the estimating equation:
    ``E[((T - e(X)) / (e(X)(1-e(X)))) * Y]`` where ``e(X)=P(T=1|X)``.
    """
    # Ensure key columns are present before any matrix/model work.
    _require_columns(df, [T, Y])

    # Build covariate matrix from formula and extract arrays for numeric ops.
    X = _design_matrix(df, ps_formula)
    treatment = df[T].to_numpy(dtype=float)
    outcome = df[Y].to_numpy(dtype=float)

    # Estimate propensity scores and trim away exact 0/1 probabilities.
    model = _fit_propensity_model(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

    # Horvitz-Thompson style transformed-outcome weights.
    weights = (treatment - propensity_scores) / (propensity_scores * (1 - propensity_scores))
    # Sample average of transformed outcome yields the IPW ATE estimate.
    ate = float(np.mean(weights * outcome))
    return ate


def doubly_robust(df: pd.DataFrame, formula: str, T: str, Y: str) -> float:
    """Estimate ATE with augmented IPW (doubly robust estimator).

    Parameters
    ----------
    df:
        Analysis dataset containing treatment, outcome, and covariates.
    formula:
        Patsy formula used for both propensity and outcome regressions.
    T:
        Treatment-column name (binary: 0/1).
    Y:
        Outcome-column name.

    Returns
    -------
    float
        Doubly robust ATE estimate.

    Notes
    -----
    The estimator remains consistent if either:
    - the propensity model is correctly specified, or
    - the outcome regressions are correctly specified.
    """
    # Validate required treatment/outcome columns.
    _require_columns(df, [T, Y])

    # Shared design matrix and core arrays.
    X = _design_matrix(df, formula)
    treatment = df[T].to_numpy(dtype=float)
    outcome = df[Y].to_numpy(dtype=float)

    # Propensity model for inverse-probability correction term.
    model = _fit_propensity_model(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    propensity_scores = np.clip(propensity_scores, 1e-6, 1 - 1e-6)

    # Fit separate outcome models by treatment arm.
    mask_treated = treatment == 1
    mask_control = treatment == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    model_treated = LinearRegression(fit_intercept=False)
    model_control = LinearRegression(fit_intercept=False)
    model_treated.fit(X.loc[mask_treated], outcome[mask_treated])
    model_control.fit(X.loc[mask_control], outcome[mask_control])

    # Predicted potential outcomes from arm-specific regressions.
    mu1 = model_treated.predict(X)
    mu0 = model_control.predict(X)

    # Augmented IPW decomposition of treated and control means.
    treated_term = treatment * (outcome - mu1) / propensity_scores + mu1
    control_term = (1 - treatment) * (outcome - mu0) / (1 - propensity_scores) + mu0
    # ATE is average contrast between augmented treated/control terms.
    ate = float(np.mean(treated_term - control_term))
    return ate