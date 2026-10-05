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
import numpy as np
import streamlit as st
from indicators import INDICADORES, calcular

st.set_page_config(page_title="Simulador de Concentración", layout="wide")
st.title("Simulador de Concentración de Mercado")

with st.sidebar:
    st.header("Parámetros")
    clave = st.selectbox("Indicador", list(INDICADORES),
                         format_func=lambda c: INDICADORES[c]["nombre"])
    n = st.number_input("Número de empresas (N)", min_value=2, max_value=100, value=10)
    k = st.number_input("k (solo CRk)", 1, int(n), min(4, int(n))) if clave == "CRk" else 4

# Prueba: mercado con cuotas iguales
s = np.full(int(n), 1 / n)
st.write(f"{clave} con cuotas iguales: **{float(calcular(clave, s, int(k))):.4f}**")
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from indicators import INDICADORES, calcular
from simulation import (simular, cuotas_aleatorias, percentil,
                        ITER_DEFECTO, ITER_MIN, ITER_MAX)
from evaluator import CATEGORIAS, evaluar

st.set_page_config(page_title="Simulador de Concentración", layout="wide")
st.title("Simulador de Concentración de Mercado (Monte Carlo)")

# ---------- PASO 1 + 3: parámetros (indicador, N, iteraciones, advertencia) ----------
with st.sidebar:
    st.header("Parámetros")
    clave = st.selectbox("Indicador", list(INDICADORES),
                         format_func=lambda c: INDICADORES[c]["nombre"])
    n = st.number_input("Número de empresas (N)", 2, 100, 10)
    k = st.number_input("k (solo CRk)", 1, int(n), min(4, int(n))) if clave == "CRk" else 4
    iters = st.number_input("Iteraciones", ITER_MIN, ITER_MAX, ITER_DEFECTO, step=100,
                            help=f"Rango {ITER_MIN}-{ITER_MAX:,}: menos da una distribución inestable; "
                                 "más ralentiza la web.")
    if iters > 10_000:
        st.warning("⚠️ Más iteraciones aumentan el tiempo de respuesta, la memoria y el uso de CPU.")
    seed = st.number_input("Semilla (0 = aleatoria)", 0, 10**6, 0)

# ---------- PASO 4: caso particular (manual o aleatorio) ----------
st.subheader("1. Caso particular")
modo = st.radio("Modo", ["Entrada manual", "Generación aleatoria"], horizontal=True)
cuotas = None

if modo == "Entrada manual":
    por_defecto = ", ".join([f"{100/n:.2f}"] * int(n))
    txt = st.text_area("Cuotas en % separadas por coma (deben sumar 100)", por_defecto)
    try:
        v = np.array([float(x) for x in txt.replace(";", ",").split(",") if x.strip()])
        if len(v) != n:
            st.error(f"Ingresaste {len(v)} cuotas y N = {n}.")
        elif (v < 0).any() or (v > 100).any():
            st.error("Cada cuota debe estar entre 0% y 100%.")
        elif not np.isclose(v.sum(), 100, atol=0.01):
            st.error(f"Las cuotas suman {v.sum():.2f}%, deben sumar 100%.")
        else:
            cuotas = v / 100
    except ValueError:
        st.error("Formato inválido: usa números separados por coma.")
else:
    if st.button("Generar caso aleatorio") or "caso" not in st.session_state:
        st.session_state["caso"] = cuotas_aleatorias(int(n))[0]
    if len(st.session_state["caso"]) != n:
        st.session_state["caso"] = cuotas_aleatorias(int(n))[0]
    cuotas = st.session_state["caso"]
    st.write(", ".join(f"{x*100:.2f}%" for x in cuotas))

if cuotas is not None:
    # ---------- PASO 2 + 5: simulación, valor del caso y percentil ----------
    dist = simular(clave, int(n), int(iters), int(k), seed or None)
    valor = float(calcular(clave, cuotas, int(k)))
    pct = percentil(dist, valor)

    st.subheader("2. Distribución de Monte Carlo")
    c1, c2 = st.columns(2)
    c1.metric(f"{clave} del caso", f"{valor:.4f}")
    c2.metric("Percentil", f"{pct:.1f}")

    # ---------- PASO 5: histograma + línea vertical del caso ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dist, bins=40, color="#6c8ebf", edgecolor="white", density=True)
    ax.axvline(valor, color="red", lw=2, label=f"Caso particular = {valor:.4f}")
    ax.set_xlabel(INDICADORES[clave]["eje"])
    ax.set_ylabel("Densidad")
    ax.set_title(f"{int(iters):,} simulaciones, N = {int(n)}")
    ax.legend()
    st.pyplot(fig)

    # ---------- PASO 6: evaluador ----------
    st.subheader("3. Evaluador")
    resp = st.radio("¿Qué nivel de concentración tiene el caso particular?",
                    CATEGORIAS, horizontal=True)
    if st.button("Verificar respuesta"):
        ok, msg = evaluar(clave, valor, int(n), resp, pct)
        (st.success if ok else st.error)(("✅ Correcto. " if ok else "❌ Incorrecto. ") + msg)
        
