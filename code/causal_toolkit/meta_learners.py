from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Ensure a DataFrame contains all columns needed by an estimator.

    Parameters
    ----------
    df:
        Input table to validate.
    columns:
        Column names that must exist in ``df``.

    Raises
    ------
    ValueError
        If any required column is missing. The error message includes the
        missing names for easier debugging.
    """
    # Set subtraction gives an order-independent "missing columns" check.
    missing = set(columns) - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _as_list(values: Iterable[str]) -> List[str]:
    """Materialize an iterable of column names into a concrete list.

    This keeps downstream indexing predictable when callers pass tuples,
    generators, pandas Index objects, etc.
    """
    return list(values)


def _fit_propensity(train: pd.DataFrame, X: List[str], T: str) -> LogisticRegression:
    """Fit a propensity model ``P(T=1 | X)`` on training data.

    A lightly regularized logistic model is used so the X-learner can blend
    treated/control pseudo-effect models using estimated treatment propensity.
    """
    # Very large C approximates weak regularization (near-MLE behavior).
    model = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    model.fit(train[X], train[T])
    return model


def s_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE with an S-learner for binary treatment.

    Method
    ------
    1. Fit one outcome model ``E[Y | X, T]`` using both features and treatment.
    2. For each test row, create two counterfactual copies:
       - treated version with ``T=1``
       - control version with ``T=0``
    3. Predict both potential outcomes and compute
       ``cate = mu1(x) - mu0(x)``.

    Returns
    -------
    pd.DataFrame
        Copy of ``test`` with a new ``cate`` column.
    """
    # Validate schema early for clearer errors.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    # Single model uses treatment indicator as an additional feature.
    features = _as_list(X) + [T]
    model = LGBMRegressor()
    model.fit(train[features], train[y])

    # Start from covariates only, then toggle treatment to build counterfactuals.
    base = test[_as_list(X)].copy()
    test_treated = base.copy()
    test_treated[T] = 1
    test_control = base.copy()
    test_control[T] = 0

    # Potential-outcome predictions under forced treatment/control.
    mu1 = model.predict(test_treated[features])
    mu0 = model.predict(test_control[features])

    # Individual uplift estimate is the difference in predicted potential outcomes.
    result = test.copy()
    result["cate"] = mu1 - mu0
    return result


def t_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE with a T-learner for binary treatment.

    Method
    ------
    1. Split training data into treated and control subsets.
    2. Fit separate outcome models:
       - ``mu1(x) = E[Y | X=x, T=1]``
       - ``mu0(x) = E[Y | X=x, T=0]``
    3. Predict both outcomes on test covariates and compute
       ``cate = mu1(x) - mu0(x)``.

    Returns
    -------
    pd.DataFrame
        Copy of ``test`` with a new ``cate`` column.
    """
    # Ensure required columns exist before model fitting.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    # Binary split by treatment assignment.
    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    # Independent response models for each arm.
    model_treated = LGBMRegressor()
    model_control = LGBMRegressor()
    model_treated.fit(train.loc[mask_treated, X], train.loc[mask_treated, y])
    model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

    # Counterfactual-style predictions from arm-specific models.
    mu1 = model_treated.predict(test[X])
    mu0 = model_control.predict(test[X])

    # Per-row uplift estimate.
    result = test.copy()
    result["cate"] = mu1 - mu0
    return result


def x_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE with an X-learner for binary treatment.

    Method
    ------
    1. Fit base outcome models ``mu1`` and ``mu0`` (as in T-learner).
    2. Construct pseudo-treatment effects on observed groups:
       - control rows: ``tau0 = mu1(x) - y``
       - treated rows: ``tau1 = y - mu0(x)``
    3. Fit separate effect models for ``tau0`` and ``tau1``.
    4. Fit propensity model ``e(x)=P(T=1|X=x)``.
    5. Blend effect predictions:
       ``cate = e(x)*tau0_hat + (1-e(x))*tau1_hat``.

    This approach can be more stable than plain T-learner when treatment share
    is imbalanced and/or outcomes are sparse.
    """
    # Input validation for required feature/treatment/outcome columns.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    # Partition by observed treatment assignment.
    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    # Stage 1: base outcome models for each arm.
    model_treated = LGBMRegressor()
    model_control = LGBMRegressor()
    model_treated.fit(train.loc[mask_treated, X], train.loc[mask_treated, y])
    model_control.fit(train.loc[mask_control, X], train.loc[mask_control, y])

    # Cross-arm counterfactual predictions on observed groups.
    mu1_on_control = model_treated.predict(train.loc[mask_control, X])
    mu0_on_treated = model_control.predict(train.loc[mask_treated, X])

    # Pseudo-effects represent imputed treatment effect targets by group.
    tau0 = mu1_on_control - train.loc[mask_control, y].to_numpy()
    tau1 = train.loc[mask_treated, y].to_numpy() - mu0_on_treated

    # Stage 2: effect models for each pseudo-effect target.
    tau0_model = LGBMRegressor()
    tau1_model = LGBMRegressor()
    tau0_model.fit(train.loc[mask_control, X], tau0)
    tau1_model.fit(train.loc[mask_treated, X], tau1)

    # Propensity is used as a data-adaptive blending weight.
    propensity_model = _fit_propensity(train, X, T)
    e_test = propensity_model.predict_proba(test[X])[:, 1]

    # Predicted pseudo-effects on test covariates.
    tau0_hat = tau0_model.predict(test[X])
    tau1_hat = tau1_model.predict(test[X])

    # Final blended CATE estimate.
    result = test.copy()
    result["cate"] = e_test * tau0_hat + (1 - e_test) * tau1_hat
    return result


def r_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE with an R-learner for binary treatment.

    Method
    ------
    1. Cross-fit nuisance models on training data:
       - outcome regression ``m(x) = E[Y|X=x]``
       - propensity ``e(x) = P(T=1|X=x)``
    2. Build residuals:
       - ``Y_tilde = Y - m_hat(X)``
       - ``T_tilde = T - e_hat(X)``
    3. Form Robinson-style pseudo-outcome ``Y_tilde / T_tilde`` with
       small-value protection and fit the final stage with weights
       ``T_tilde^2``.
    4. Predict ``cate`` on test covariates.

    Notes
    -----
    Cross-fitting reduces overfitting bias in nuisance estimates before the
    final effect model is learned.
    """
    # Validate required columns and ensure both treatment arms are present.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    X_train = train[X]
    treatment_train = train[T].to_numpy()
    outcome_train = train[y].to_numpy()

    # Out-of-fold nuisance predictions for orthogonalization.
    outcome_hat = np.zeros(len(train))
    propensity_hat = np.zeros(len(train))

    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    for fit_idx, hold_idx in kf.split(X_train):
        X_fit = X_train.iloc[fit_idx]
        X_hold = X_train.iloc[hold_idx]

        outcome_model = LGBMRegressor()
        outcome_model.fit(X_fit, outcome_train[fit_idx])
        outcome_hat[hold_idx] = outcome_model.predict(X_hold)

        propensity_model = _fit_propensity(train.iloc[fit_idx], X, T)
        propensity_hat[hold_idx] = propensity_model.predict_proba(X_hold)[:, 1]

    # Guard against near-zero denominators and extreme propensity values.
    propensity_hat = np.clip(propensity_hat, 1e-3, 1 - 1e-3)
    treatment_residual = treatment_train - propensity_hat
    outcome_residual = outcome_train - outcome_hat

    epsilon = 1e-6
    safe_treatment_residual = np.where(
        np.abs(treatment_residual) < epsilon,
        np.where(treatment_residual >= 0, epsilon, -epsilon),
        treatment_residual,
    )

    pseudo_outcome = outcome_residual / safe_treatment_residual
    weights = treatment_residual**2

    # Final-stage heterogeneous effect model.
    tau_model = LGBMRegressor()
    tau_model.fit(X_train, pseudo_outcome, sample_weight=weights)

    result = test.copy()
    result["cate"] = tau_model.predict(test[X])
    return result


def double_ml_cate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE using a simple cross-fitted Double-ML style procedure.

    Method
    ------
    1. Cross-fit nuisance models on training data:
       - treatment model ``m(x) ~ T``
       - outcome model ``g(x) ~ Y``
    2. Compute residualized treatment and outcome:
       - ``T_tilde = T - m_hat(X)``
       - ``Y_tilde = Y - g_hat(X)``
    3. Form pseudo-outcome ``Y_tilde / T_tilde`` with small-value protection.
    4. Fit final CATE model on ``X`` using weights ``T_tilde^2``.
    5. Predict CATE on test rows.

    Notes
    -----
    This implementation is intentionally lightweight and practical, not a full
    orthogonal-score framework with extensive diagnostics.
    """
    # Ensure train/test include all required fields.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    X_train = train[X]
    treatment_train = train[T].to_numpy()
    outcome_train = train[y].to_numpy()

    # Out-of-fold predictions for nuisance models (cross-fitting).
    treatment_hat = np.zeros(len(train))
    outcome_hat = np.zeros(len(train))

    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    for train_idx, hold_idx in kf.split(X_train):
        # Fit nuisance models on one fold...
        model_treatment = LGBMRegressor()
        model_outcome = LGBMRegressor()
        X_train_fold = X_train.iloc[train_idx]
        X_holdout_fold = X_train.iloc[hold_idx]

        model_treatment.fit(X_train_fold, treatment_train[train_idx])
        model_outcome.fit(X_train_fold, outcome_train[train_idx])

        # ...and predict nuisance components on the holdout fold.
        treatment_hat[hold_idx] = model_treatment.predict(X_holdout_fold)
        outcome_hat[hold_idx] = model_outcome.predict(X_holdout_fold)

    # Residualize treatment and outcome.
    treatment_residual = treatment_train - treatment_hat
    outcome_residual = outcome_train - outcome_hat

    # Guard against division by values too close to zero.
    epsilon = 1e-6
    safe_treatment_residual = np.where(
        np.abs(treatment_residual) < epsilon,
        epsilon,
        treatment_residual,
    )
    # Robinson-style pseudo-outcome and heteroskedasticity-aware weights.
    pseudo_outcome = outcome_residual / safe_treatment_residual
    weights = treatment_residual**2

    # Final stage learns heterogeneous effect as a function of X.
    tau_model = LGBMRegressor()
    tau_model.fit(X_train, pseudo_outcome, sample_weight=weights)

    # Return test data with estimated CATE.
    result = test.copy()
    result["cate"] = tau_model.predict(test[X])
    return result


def dr_learner_discrete(
    train: pd.DataFrame,
    test: pd.DataFrame,
    X: list[str],
    T: str,
    y: str,
) -> pd.DataFrame:
    """Estimate CATE with a DR-learner for binary treatment.

    Method
    ------
    1. Cross-fit nuisance models on training data:
       - outcome models ``mu1(x)=E[Y|X=x,T=1]`` and ``mu0(x)=E[Y|X=x,T=0]``
       - propensity model ``e(x)=P(T=1|X=x)``
    2. Build doubly robust pseudo-outcome on each training row:
       ``phi = mu1 - mu0 + T*(Y-mu1)/e - (1-T)*(Y-mu0)/(1-e)``
    3. Fit a final-stage effect model ``tau(x)`` on ``(X, phi)``.
    4. Predict ``cate`` on test covariates.

    Notes
    -----
    The pseudo-outcome is doubly robust in the ATE sense and practically useful
    for CATE ranking when paired with cross-fitting and propensity clipping.
    """
    # Validate schema and ensure both treatment arms appear in train.
    _require_columns(train, [*X, T, y])
    _require_columns(test, [*X, T, y])

    mask_treated = train[T] == 1
    mask_control = train[T] == 0
    if not mask_treated.any() or not mask_control.any():
        raise ValueError("Both treatment and control groups must be non-empty.")

    X_train = train[X]
    treatment_train = train[T].to_numpy()
    outcome_train = train[y].to_numpy()

    # Out-of-fold nuisance predictions.
    mu1_hat = np.zeros(len(train))
    mu0_hat = np.zeros(len(train))
    propensity_hat = np.zeros(len(train))

    kf = KFold(n_splits=2, shuffle=True, random_state=123)
    for fit_idx, hold_idx in kf.split(X_train):
        fit_df = train.iloc[fit_idx]
        X_hold = X_train.iloc[hold_idx]

        fold_treated = fit_df[T] == 1
        fold_control = fit_df[T] == 0
        if not fold_treated.any() or not fold_control.any():
            raise ValueError("Each cross-fitting fold must include both treatment arms.")

        mu1_model = LGBMRegressor()
        mu0_model = LGBMRegressor()
        mu1_model.fit(fit_df.loc[fold_treated, X], fit_df.loc[fold_treated, y])
        mu0_model.fit(fit_df.loc[fold_control, X], fit_df.loc[fold_control, y])

        mu1_hat[hold_idx] = mu1_model.predict(X_hold)
        mu0_hat[hold_idx] = mu0_model.predict(X_hold)

        propensity_model = _fit_propensity(fit_df, X, T)
        propensity_hat[hold_idx] = propensity_model.predict_proba(X_hold)[:, 1]

    # Clip propensity to avoid unstable inverse-probability corrections.
    propensity_hat = np.clip(propensity_hat, 1e-3, 1 - 1e-3)

    # Doubly robust pseudo-outcome for heterogeneous effect learning.
    dr_pseudo_outcome = (
        mu1_hat
        - mu0_hat
        + treatment_train * (outcome_train - mu1_hat) / propensity_hat
        - (1 - treatment_train) * (outcome_train - mu0_hat) / (1 - propensity_hat)
    )

    tau_model = LGBMRegressor()
    tau_model.fit(X_train, dr_pseudo_outcome)

    result = test.copy()
    result["cate"] = tau_model.predict(test[X])
    return result
