import numpy as np
from indicators import calcular

ITER_DEFECTO = 1000
ITER_MIN, ITER_MAX = 100, 100_000  # editables: bajo 100 la distribución es inestable; sobre 100.000 la web se ralentiza


def cuotas_aleatorias(n, iteraciones=1, rng=None):
    """Dirichlet(1,...,1): uniforme sobre el simplex, cada fila suma exactamente 1."""
    rng = rng or np.random.default_rng()
    return rng.dirichlet(np.ones(n), size=iteraciones)


def simular(clave, n, iteraciones=ITER_DEFECTO, k=4, seed=None):
    rng = np.random.default_rng(seed)
    cuotas = cuotas_aleatorias(n, iteraciones, rng)
    assert np.allclose(cuotas.sum(axis=1), 1.0)  # restricción: suma de cuotas = 1
    return calcular(clave, cuotas, k)


def percentil(distribucion, valor):
    """% de simulaciones con valor menor o igual al del caso particular."""
    return float((np.asarray(distribucion) <= valor).mean() * 100)
