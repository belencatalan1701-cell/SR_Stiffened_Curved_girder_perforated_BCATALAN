import numpy as np

from sr_stiffened_curved_girder_perforated_bcatalan import (
    BucklingPredictor,
    GeneticAlgorithm,
)


def test_buckling_predictor_fits_and_predicts_reasonably():
    X = np.array([
        [0.20, 0.50, 0.90],
        [0.25, 0.60, 0.95],
        [0.30, 0.65, 1.00],
        [0.40, 0.72, 1.10],
        [0.50, 0.80, 1.20],
    ], dtype=float)
    y = np.array([0.84, 0.91, 1.00, 1.14, 1.25], dtype=float)

    model = BucklingPredictor()
    model.fit(X, y)
    prediction = model.predict(np.array([[0.35, 0.68, 1.05]], dtype=float))[0]

    assert model.coefficients_.shape == (3,)
    assert prediction > 0
    assert abs(prediction - 1.06) < 0.25


def test_genetic_algorithm_can_optimize_a_simple_objective():
    objective = lambda candidate: -abs(candidate[0] - 2.5) - abs(candidate[1] - 0.7)
    algorithm = GeneticAlgorithm(population_size=24, generations=60, mutation_rate=0.15)

    best = algorithm.optimize(objective, dimensions=2, bounds=[(-5, 5), (-5, 5)])

    assert len(best) == 2
    assert abs(best[0] - 2.5) < 1.0
    assert abs(best[1] - 0.7) < 1.0
