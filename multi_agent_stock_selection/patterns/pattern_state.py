"""
State manager for pattern clusters (Gaussian Mixture Models).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture
try:  # pragma: no cover - optional dependency
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

from multi_agent_stock_selection.patterns.models import ThesisPatternRecord

logger = logging.getLogger(__name__)


class PatternState:
    """Keeps track of the Gaussian mixture and cumulative performance metrics."""

    def __init__(
        self,
        n_components: int,
        decay_lambda: float,
        random_state: int = 42,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 200,
        beta: float = 1.0,
        distance_metric: str = "euclidean",
    ) -> None:
        if not (0.0 <= decay_lambda <= 1.0):
            raise ValueError("decay_lambda must be within [0, 1].")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        self.n_components = n_components
        self.decay_lambda = decay_lambda
        self.beta = beta
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        if distance_metric not in {"euclidean", "manhattan"}:
            raise ValueError("distance_metric must be 'euclidean' or 'manhattan'.")
        self.distance_metric = distance_metric

        self.model: Optional[GaussianMixture] = None
        self.feature_dim: Optional[int] = None
        self.p_scores = np.zeros(self.n_components, dtype=np.float32)

    def _build_model(self, feature_dim: int) -> GaussianMixture:
        self.feature_dim = feature_dim
        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
            warm_start=False,
            init_params="kmeans",
            n_init=1,
        )
        return model

    def fit(
        self, embeddings: np.ndarray, prev_means: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[List[Tuple[int, int, float]]], np.ndarray]:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array.")
        n_samples = embeddings.shape[0]
        if n_samples < self.n_components:
            raise ValueError(
                f"GaussianMixture requires at least {self.n_components} samples, got {n_samples}. "
                "Consider lowering the number of patterns."
            )
        model = self._build_model(embeddings.shape[1])
        model.fit(embeddings)
        alignment, predicted_prev = self._align_scores(prev_means, model.means_)
        self.model = model
        return model.predict_proba(embeddings), alignment, predicted_prev

    def predict_proba(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2-D array.")
        return self.model.predict_proba(embeddings)

    def _align_scores(
        self, prev_means: Optional[np.ndarray], new_means: Optional[np.ndarray]
    ) -> Tuple[Optional[List[Tuple[int, int, float]]], np.ndarray]:
        predicted_prev = np.zeros(self.n_components, dtype=np.float32)
        if prev_means is None or new_means is None:
            return None, predicted_prev
        if prev_means.shape[0] != new_means.shape[0]:
            return None, predicted_prev

        distance_matrix = self._compute_distance_matrix(new_means, prev_means)
        mapping = self._solve_assignment(distance_matrix)

        old_p = self.p_scores.copy()
        new_p = np.zeros_like(old_p)

        alignment: List[Tuple[int, int, float]] = []
        for new_idx, prev_idx in enumerate(mapping):
            dist = float(distance_matrix[new_idx, prev_idx])
            new_p[new_idx] = old_p[prev_idx]
            # decay = math.exp(-((self.beta ** 2) * dist))
            decay = 1 / (1 + self.beta * dist)
            predicted_prev[prev_idx] = old_p[prev_idx] * decay
            alignment.append((new_idx, int(prev_idx), dist))

        self.p_scores = new_p
        return alignment, predicted_prev

    def _compute_distance_matrix(self, new_means: np.ndarray, prev_means: np.ndarray) -> np.ndarray:
        diff = new_means[:, None, :] - prev_means[None, :, :]
        if self.distance_metric == "manhattan":
            return np.sum(np.abs(diff), axis=2)
        return np.linalg.norm(diff, axis=2)

    def _solve_assignment(self, cost_matrix: np.ndarray) -> np.ndarray:
        n = cost_matrix.shape[0]
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:  # pragma: no cover - fallback to greedy matching
            row_ind = list(range(n))
            available = set(range(n))
            col_ind = []
            for i in row_ind:
                best_j = min(available, key=lambda j, i=i: cost_matrix[i, j])
                col_ind.append(best_j)
                available.remove(best_j)
        mapping = np.zeros(n, dtype=int)
        for r, c in zip(row_ind, col_ind):
            mapping[r] = c
        return mapping

    def update_performance(
        self,
        thesis_records: Sequence[ThesisPatternRecord],
        excess_return_lookup: Dict[tuple[str, int], float],
    ) -> None:
        if not thesis_records:
            return

        aggregated = np.zeros(self.n_components, dtype=np.float32)
        weights = np.zeros(self.n_components, dtype=np.float32)

        for record in thesis_records:
            ret = excess_return_lookup.get((record.stock, record.date))
            if ret is None:
                continue
            perf = record.polarity_sign * ret
            aggregated += perf * record.responsibilities
            weights += record.responsibilities

        if not np.any(weights > 0):
            logger.debug("No valid thesis performance to update pattern scores.")
            return

        performance = np.zeros_like(aggregated)
        valid_mask = weights > 0
        performance[valid_mask] = aggregated[valid_mask] / weights[valid_mask]

        self.p_scores = self.decay_lambda * self.p_scores + (1 - self.decay_lambda) * performance
