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
