import numpy as np

CATEGORIAS = ["Baja", "Moderada", "Alta"]

# EDITA AQUÍ los umbrales (corte_bajo, corte_alto)
UMBRALES = {
    "IHH": (1500, 2500),   # escala 0-10.000 (criterio DOJ 2010)
    "CRk": (0.40, 0.70),   # proporción
    "ID":  (0.25, 0.50),
    "IE":  (0.80, 0.50),   # sobre IE/ln(N): >=0.8 baja, <0.5 alta
}


def clasificar(clave, valor, n):
    lo, hi = UMBRALES[clave]
    if clave == "IHH":
        v = valor * 10_000
        txt = f"IHH = {v:,.0f} (escala 0-10.000). Baja < {lo}, Moderada {lo}-{hi}, Alta > {hi}."
    elif clave == "IE":
        v = valor / np.log(n)
        txt = (f"IE normalizada (IE/ln N) = {v:.3f}. Baja >= {lo}, Moderada {hi}-{lo}, Alta < {hi}. "
               "A menor entropía, mayor concentración.")
        return ("Baja" if v >= lo else "Moderada" if v >= hi else "Alta"), txt
    else:
        v = valor
        txt = f"Valor = {v:.3f}. Baja < {lo}, Moderada {lo}-{hi}, Alta > {hi}."
    return ("Baja" if v < lo else "Moderada" if v <= hi else "Alta"), txt


def evaluar(clave, valor, n, respuesta, pct):
    correcta, detalle = clasificar(clave, valor, n)
    msg = (f"Concentración **{correcta}**. {detalle} "
           f"El caso supera al **{pct:.1f}%** de los mercados simulados.")
    return respuesta == correcta, msg
