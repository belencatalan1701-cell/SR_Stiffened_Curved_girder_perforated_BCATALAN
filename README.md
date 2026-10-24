# SR_Stiffened_Curved_girder_perforated_BCATALAN

Desarrollo de un algoritmo genético mediante librerías abiertas para predecir una expresión de coeficientes de pandeo específico para vigas de acero de almas curvas con perforación circular. Esta rutina se apoya en una base de datos de valores despejados de kf obtenidos con FCR y ANSYS.

## Objetivo

Este repositorio sirve como base inicial para un proyecto de modelado predictivo orientado a la estimación de coeficientes de pandeo en secciones metálicas con geometrías complejas. El enfoque incluye un predictor lineal para ajuste rápido y un algoritmo genético para optimización continua de funciones objetivo.

## Estructura del proyecto

- `sr_stiffened_curved_girder_perforated_bcatalan/__init__.py`: implementación principal del predictor y del algoritmo genético.
- `tests/test_genetic_algorithm.py`: pruebas de regresión para validar el comportamiento esperado.
- `requirements.txt`: dependencias del proyecto.

## Instalación

```bash
python -m pip install -r requirements.txt
```

## Uso rápido

```python
import numpy as np
from sr_stiffened_curved_girder_perforated_bcatalan import BucklingPredictor, GeneticAlgorithm

X = np.array([
    [0.20, 0.50, 0.90],
    [0.25, 0.60, 0.95],
    [0.30, 0.65, 1.00],
], dtype=float)
y = np.array([0.84, 0.91, 1.00], dtype=float)

predictor = BucklingPredictor().fit(X, y)
print(predictor.predict(np.array([[0.28, 0.62, 0.97]], dtype=float)))

objective = lambda candidate: -(candidate[0] - 2.5) ** 2 - (candidate[1] - 0.7) ** 2
result = GeneticAlgorithm(population_size=24, generations=40, mutation_rate=0.15).optimize(
    objective,
    dimensions=2,
    bounds=[(-5, 5), (-5, 5)],
)
print(result)
```

## Validación

```bash
pytest -q
```

El proyecto está preparado para ampliarse con una base de datos real de valores de kf, regresiones no lineales y validaciones más rigurosas con casos de ingeniería.
