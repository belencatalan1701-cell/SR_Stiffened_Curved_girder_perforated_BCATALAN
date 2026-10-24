"""Buckling prediction utilities for perforated curved steel girders."""

from __future__ import annotations

import numpy as np


class BucklingPredictor:
    """Simple linear predictor for buckling coefficient estimation."""

    def __init__(self):
        self.coefficients_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature column.")

        design_matrix = np.hstack([X, np.ones((X.shape[0], 1))])
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)

        self.coefficients_ = coefficients[:-1].copy()
        self.intercept_ = float(coefficients[-1])
        return self

    def predict(self, X):
        if self.coefficients_ is None:
            raise ValueError("The predictor must be fitted before calling predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.coefficients_ + self.intercept_


class GeneticAlgorithm:
    """Minimal genetic optimizer for continuous-valued problems."""

    def __init__(self, population_size=32, generations=50, mutation_rate=0.1, seed=None):
        self.population_size = int(population_size)
        self.generations = int(generations)
        self.mutation_rate = float(mutation_rate)
        self.rng = np.random.default_rng(seed)

    def _random_candidate(self, dimensions, bounds):
        candidate = np.empty(dimensions)
        for index in range(dimensions):
            lower, upper = bounds[index]
            candidate[index] = self.rng.uniform(lower, upper)
        return candidate

    def optimize(self, objective, dimensions=1, bounds=None):
        if bounds is None:
            bounds = [(-1.0, 1.0)] * dimensions
        if len(bounds) != dimensions:
            raise ValueError("The bounds length must match dimensions.")

        population = np.array(
            [self._random_candidate(dimensions, bounds) for _ in range(self.population_size)],
            dtype=float,
        )

        best_candidate = None
        best_score = -np.inf

        for _ in range(self.generations):
            scores = np.array([objective(candidate) for candidate in population], dtype=float)
            order = np.argsort(scores)[::-1]
            population = population[order]
            scores = scores[order]

            if scores[0] > best_score:
                best_score = float(scores[0])
                best_candidate = population[0].copy()

            elite_count = max(1, self.population_size // 5)
            next_population = population[:elite_count].copy()

            while next_population.shape[0] < self.population_size:
                parent_a = population[self.rng.integers(0, self.population_size)]
                parent_b = population[self.rng.integers(0, self.population_size)]
                child = 0.5 * (parent_a + parent_b)

                if self.rng.random() < self.mutation_rate:
                    child = child + self.rng.normal(0.0, 0.5, size=dimensions)

                lower = np.asarray([bound[0] for bound in bounds], dtype=float)
                upper = np.asarray([bound[1] for bound in bounds], dtype=float)
                child = np.clip(child, lower, upper)
                next_population = np.vstack([next_population, child])

            population = next_population[: self.population_size]

        if best_candidate is None:
            best_candidate = population[0].copy()

        return best_candidate.tolist()


__all__ = ["BucklingPredictor", "GeneticAlgorithm"]
