"""Class-based meta-learners for causal effect estimation.

This module currently provides a robust, class-oriented R-learner implementation
for binary treatment settings. It is designed to mirror the core logic used in
function-based R-learner code while giving callers explicit control over model
objects and hyperparameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


class RLearnerRobust:
    """Estimate CATE using a class-based, cross-fitted R-learner.

    Method overview
    ---------------
    1. Cross-fit nuisance models on training rows:
       - outcome model m(x) ~= E[Y | X=x]
       - propensity model e(x) ~= P(T=1 | X=x)
    2. Compute residuals:
       - y_residual = Y - m_hat(X)
       - t_residual = T - e_hat(X)
    3. Build pseudo-outcome and sample weights:
       - pseudo_outcome = y_residual / t_residual
       - sample_weight = t_residual^2
       with safeguards for near-zero denominators.
    4. Fit a final-stage tau model to learn heterogeneous treatment effects.

    Notes
    -----
    - Binary treatment (0/1) is expected.
    - Propensity clipping and epsilon denominator protection improve numerical
      stability in sparse or imbalanced settings.
    - This class is intended for robustness/sensitivity checks and model
      experimentation where a stateful API is convenient.
    """

    def __init__(
        self,
        outcome_model,
        propensity_model,
        tau_model,
        n_splits: int = 2,
        min_propensity: float = 1e-3,
        max_propensity: float = 1 - 1e-3,
        epsilon: float = 1e-6,
        random_state: int = 123,
        stratify_propensity: bool = True,
    ) -> None:
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.tau_model = tau_model
        self.n_splits = n_splits
        self.min_propensity = min_propensity
        self.max_propensity = max_propensity
        self.epsilon = epsilon
        self.random_state = random_state
        self.stratify_propensity = stratify_propensity

    def _cross_fit_regression(self, X: pd.DataFrame, y: np.ndarray, model) -> np.ndarray:
        """Return out-of-fold predictions for a regression nuisance model."""
        splitter = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        preds = np.zeros(len(X), dtype=float)

        for fit_idx, hold_idx in splitter.split(X):
            fitted_model = clone(model)
            fitted_model.fit(X.iloc[fit_idx], y[fit_idx])
            preds[hold_idx] = fitted_model.predict(X.iloc[hold_idx])

        return preds

    def _cross_fit_propensity(self, X: pd.DataFrame, t: np.ndarray, model) -> np.ndarray:
        """Return out-of-fold propensity scores P(T=1|X).

        Stratified folds are preferred for treatment classification to preserve
        class proportions when treatment assignment is imbalanced.
        """
        if self.stratify_propensity and len(np.unique(t)) > 1:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X, t)
        else:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            split_iter = splitter.split(X)

        preds = np.zeros(len(X), dtype=float)

        for fit_idx, hold_idx in split_iter:
            fitted_model = clone(model)
            fitted_model.fit(X.iloc[fit_idx], t[fit_idx])
            preds[hold_idx] = fitted_model.predict_proba(X.iloc[hold_idx])[:, 1]

        return preds

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray):
        """Fit the robust class-based R-learner on training data."""
        self.X_columns_ = X.columns.tolist()

        y = np.asarray(outcome, dtype=float)
        t = np.asarray(treatment, dtype=float)

        # Cross-fitted nuisance predictions reduce overfitting bias.
        self.m_hat_ = self._cross_fit_regression(X, y, self.outcome_model)
        self.e_hat_ = self._cross_fit_propensity(X, t, self.propensity_model)

        # Clip propensity away from 0 and 1 to stabilize residualization.
        self.e_hat_ = np.clip(self.e_hat_, self.min_propensity, self.max_propensity)

        self.y_residual_ = y - self.m_hat_
        self.t_residual_ = t - self.e_hat_

        # Prevent division blow-ups when treatment residual is very close to 0.
        safe_t_residual = np.where(
            np.abs(self.t_residual_) < self.epsilon,
            np.where(self.t_residual_ >= 0, self.epsilon, -self.epsilon),
            self.t_residual_,
        )

        pseudo_outcome = self.y_residual_ / safe_t_residual
        sample_weight = self.t_residual_**2

        # Final-stage model learns tau(x) with heteroskedasticity-aware weights.
        self.tau_model_ = clone(self.tau_model)
        self.tau_model_.fit(X, pseudo_outcome, sample_weight=sample_weight)

        return self

    def predict_tau(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CATE/ITE scores from fitted tau model."""
        return self.tau_model_.predict(X[self.X_columns_])

