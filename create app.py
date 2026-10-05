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
      
