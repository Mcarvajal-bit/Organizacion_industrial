import numpy as np


def cr_k(s, k=4):
    """Ratio de concentración: suma de las k mayores cuotas."""
    s = np.asarray(s, float)
    return np.sort(s, axis=-1)[..., ::-1][..., :k].sum(axis=-1)


def ihh(s):
    """Herfindahl-Hirschman en escala 0-1 (x10.000 para la escala clásica)."""
    return (np.asarray(s, float) ** 2).sum(axis=-1)


def dominancia(s):
    """Índice de Dominancia (García Alba): suma de h_i^2, con h_i = s_i^2 / IHH."""
    sq = np.asarray(s, float) ** 2
    h = sq / sq.sum(axis=-1, keepdims=True)
    return (h ** 2).sum(axis=-1)


def entropia(s):
    """Entropía de Shannon: -sum(s_i * ln s_i). Máximo = ln(N)."""
    s = np.asarray(s, float)
    seguro = np.where(s > 0, s, 1.0)  # evita log(0)
    return -(s * np.log(seguro)).sum(axis=-1)


# Para agregar o cambiar un indicador a futuro, edita solo este diccionario
INDICADORES = {
    "CRk": {"nombre": "Ratio de Concentración (CRk)", "eje": "CRk (proporción, 0-1)", "f": cr_k},
    "IHH": {"nombre": "Índice Herfindahl-Hirschman (IHH)", "eje": "IHH (escala 0-1)", "f": ihh},
    "ID":  {"nombre": "Índice de Dominancia (ID)", "eje": "ID (adimensional)", "f": dominancia},
    "IE":  {"nombre": "Índice de Entropía (IE)", "eje": "IE (nats)", "f": entropia},
}


def calcular(clave, s, k=4):
    f = INDICADORES[clave]["f"]
    return f(s, k) if clave == "CRk" else f(s)
