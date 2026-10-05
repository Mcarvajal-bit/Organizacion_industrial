pip install -r requirements.txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
scipy>=1.10.0
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Concentración de Mercado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Simulador Interactivo y Evaluador de Concentración de Mercado")
st.markdown("Organización Industrial - Método de Monte Carlo")

# -----------------------------------------------------------------------------
# FUNCIONES MATEMÁTICAS (MÓDULO 1: INDICADORES)
# -----------------------------------------------------------------------------
def calcular_crk(s, k=4):
    """Ratio de Concentración CRk."""
    s_sorted = np.sort(s)[::-1]
    return float(np.sum(s_sorted[:k]))

def calcular_ihh(s):
    """Índice de Herfindahl-Hirschman (en escala 0 a 10,000)."""
    return float(np.sum((s * 100) ** 2))

def calcular_id(s):
    """Índice de Dominancia (ID)."""
    s_sorted = np.sort(s)[::-1]
    h = np.sum(s ** 2)
    if h == 0:
        return 0.0
    # Fórmula estandarizada de ID basada en asimetría de cuotas
    id_val = np.sum((s_sorted ** 4) / (h ** 2))
    return float(id_val)

def calcular_ie(s):
    """Índice de Entropía (IE)."""
    # Evitar logaritmo de cero
    s_safe = np.where(s > 0, s, 1e-12)
    return float(np.sum(s_safe * np.log(1.0 / s_safe)))

def calcular_indicador(s, nombre_indicador, k_val=4):
    if nombre_indicador == "Ratio de Concentración (CRk)":
        return calcular_crk(s, k=k_val)
    elif nombre_indicador == "Índice de Herfindahl-Hirschman (IHH)":
        return calcular_ihh(s)
    elif nombre_indicador == "Índice de Dominancia (ID)":
        return calcular_id(s)
    elif nombre_indicador == "Índice de Entropía (IE)":
        return calcular_ie(s)
    return 0.0

# -----------------------------------------------------------------------------
# MÓDULO 2: MOTOR DE SIMULACIÓN DE MONTE CARLO
# -----------------------------------------------------------------------------
def simular_monte_carlo(n_empresas, n_iteraciones, indicador_nombre, k_val=4):
    """
    Genera vectores de cuotas usando la distribución Dirichlet(1,...,1),
    garantizando estrictamente sum(s_i) = 1.0 en cada iteración.
    """
    # Dirichlet uniformemente distribuida sobre el símplex sum(s_i) = 1
    alpha = np.ones(n_empresas)
    cuotas_simuladas = np.random.dirichlet(alpha, size=n_iteraciones)
    
    resultados = np.zeros(n_iteraciones)
    for i in range(n_iteraciones):
        resultados[i] = calcular_indicador(cuotas_simuladas[i], indicador_nombre, k_val)
        
    return resultados

# -----------------------------------------------------------------------------
# BARRA LATERAL: PARÁMETROS Y CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Mercado")

# Selección de Indicador
indicador_sel = st.sidebar.selectbox(
    "Seleccione el Indicador de Concentración:",
    [
        "Índice de Herfindahl-Hirschman (IHH)",
        "Ratio de Concentración (CRk)",
        "Índice de Dominancia (ID)",
        "Índice de Entropía (IE)"
    ]
)

k_param = 4
if indicador_sel == "Ratio de Concentración (CRk)":
    k_param = st.sidebar.slider("Valor de k (para CRk):", min_value=1, max_value=10, value=4)

# Número de Empresas (N entre 2 y 100)
n_empresas = st.sidebar.number_input(
    "Número de Empresas (N):",
    min_value=2,
    max_value=100,
    value=10,
    step=1
)

st.sidebar.subheader("🎲 Parámetros de Monte Carlo")
n_iter = st.sidebar.number_input(
    "Número de Iteraciones:",
    min_value=100,
    max_value=50000,
    value=1000,
    step=500
)

# Advertencia de Carga Computacional
if n_iter > 5000:
    st.sidebar.warning(
        "⚠️ **Advertencia de Carga Computacional:** "
        "Incrementar las iteraciones por encima de 5,000 puede aumentar la latencia "
        "y el uso de recursos de procesamiento web."
    )

# -----------------------------------------------------------------------------
# SECCIÓN PRINCIPAL: CASO PARTICULAR
# -----------------------------------------------------------------------------
st.header("📌 1. Definición del Caso Particular")

col1, col2 = st.columns([1, 2])

with col1:
    modo_ingreso = st.radio(
        "Método para definir cuotas del caso particular:",
        ["Generación Aleatoria Puntual", "Entrada Manual"]
    )

cuotas_caso = []

if modo_ingreso == "Generación Aleatoria Puntual":
    if st.button("🎲 Generar Nuevo Caso Aleatorio") or "cuotas_aleatorias" not in st.session_state:
        raw_cuotas = np.random.dirichlet(np.ones(n_empresas))
        st.session_state["cuotas_aleatorias"] = raw_cuotas
    cuotas_caso = st.session_state["cuotas_aleatorias"]
else:
    st.markdown("Ingrese el porcentaje de cuota de mercado para cada empresa (%):")
    cuotas_input = []
    cols_input = st.columns(min(n_empresas, 5))
    for i in range(n_empresas):
        col_idx = i % 5
        val = cols_input[col_idx].number_input(
            f"Empresa {i+1} (%)",
            min_value=0.0,
            max_value=100.0,
            value=round(100.0 / n_empresas, 2),
            key=f"emp_{i}"
        )
        cuotas_input.append(val)
    
    # Validar que sumen 100%
    suma_ingresada = sum(cuotas_input)
    if not np.isclose(suma_ingresada, 100.0, atol=0.1):
        st.error(f"❌ La suma de las cuotas debe ser 100%. Suma actual: {suma_ingresada:.2f}%")
        st.stop()
    else:
        cuotas_caso = np.array(cuotas_input) / 100.0

# Calcular indicador para el caso particular
valor_caso = calcular_indicador(cuotas_caso, indicador_sel, k_param)

with col2:
    st.subheader("Cuotas de Mercado del Caso Particular")
    df_cuotas = pd.DataFrame({
        "Empresa": [f"Empresa {i+1}" for i in range(n_empresas)],
        "Cuota (%)": np.round(cuotas_caso * 100, 2)
    })
    
    fig_pie = px.pie(df_cuotas, values="Cuota (%)", names="Empresa", title="Distribución del Caso Particular")
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO 3: SIMULACIÓN DE MONTE CARLO Y GRÁFICO COMPARATIVO
# -----------------------------------------------------------------------------
st.header("📈 2. Simulación de Monte Carlo y Comparación")

# Ejecutar simulación
simulaciones = simular_monte_carlo(n_empresas, n_iter, indicador_sel, k_param)

# Calcular Percentil del Caso Particular
percentil = (np.sum(simulaciones <= valor_caso) / len(simulaciones)) * 100

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Valor Caso Particular", f"{valor_caso:.4f}")
col_m2.metric("Promedio Simulado", f"{np.mean(simulaciones):.4f}")
col_m3.metric("Posición (Percentil)", f"{percentil:.1f}%")

# Graficar Distribución con Marcador del Caso Particular
fig_dist = go.Figure()

# Histograma de Monte Carlo
fig_dist.add_trace(go.Histogram(
    x=simulaciones,
    histnorm='probability density',
    name='Simulación Monte Carlo',
    marker_color='#1f77b4',
    opacity=0.75
))

# Línea vertical del Caso Particular
fig_dist.add_vline(
    x=valor_caso,
    line_width=3,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Caso Particular ({valor_caso:.2f})",
    annotation_position="top right"
)

fig_dist.update_layout(
    title=f"Distribución Estocástica de Monte Carlo ({n_iter} iteraciones)",
    xaxis_title=f"Valor del Indicador: {indicador_sel}",
    yaxis_title="Densidad de Probabilidad",
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_dist, use_container_width=True)

# -----------------------------------------------------------------------------
# MÓDULO 4: EVALUADOR DE CONCENTRACIÓN Y RETROALIMENTACIÓN
# -----------------------------------------------------------------------------
st.header("📝 3. Evaluador de Concentración y Retroalimentación Pedagógica")

st.subheader("Cuestionario de Evaluación")

opciones_eval = ["Baja Concentración / Mercado Competitivo", "Concentración Moderada", "Alta Concentración / Mercado Concentrado"]
respuesta_usuario = st.radio(
    "Según los resultados obtenidos, ¿cómo clasificaría el nivel de concentración de este caso particular?",
    opciones_eval
)

if st.button("Evaluar Respuesta"):
    # Lógica de umbrales según el indicador seleccionado
    clasificacion_teorica = ""
    justificacion = ""

    if indicador_sel == "Índice de Herfindahl-Hirschman (IHH)":
        if valor_caso < 1500:
            clasificacion_teorica = "Baja Concentración / Mercado Competitivo"
            justificacion = "Un IHH inferior a 1,500 puntos indica un mercado desconcentrado o competitivo según los estándares internacionales (ej. DoJ / FTC)."
        elif 1500 <= valor_caso <= 2500:
            clasificacion_teorica = "Concentración Moderada"
            justificacion = "Un IHH entre 1,500 y 2,500 puntos corresponde a un mercado moderadamente concentrado."
        else:
            clasificacion_teorica = "Alta Concentración / Mercado Concentrado"
            justificacion = "Un IHH superior a 2,500 puntos representa un mercado altamente concentrado."

    elif indicador_sel == "Ratio de Concentración (CRk)":
        cr_pct = valor_caso * 100 if valor_caso <= 1.0 else valor_caso
        if cr_pct < 40:
            clasificacion_teorica = "Baja Concentración / Mercado Competitivo"
            justificacion = f"Un CR{k_param} menor al 40% refleja una estructura competitiva o atomizada."
        elif 40 <= cr_pct <= 70:
            clasificacion_teorica = "Concentración Moderada"
            justificacion = f"Un CR{k_param} entre 40% y 70% sugiere una concentración moderada u oligopolio débil."
        else:
            clasificacion_teorica = "Alta Concentración / Mercado Concentrado"
            justificacion = f"Un CR{k_param} superior al 70% indica un oligopolio estrecho o alta concentración."

    else:
        # Umbrales basados en percentiles de la simulación para ID e IE
        if percentil < 33:
            clasificacion_teorica = "Baja Concentración / Mercado Competitivo"
            justificacion = f"El valor del indicador se sitúa en el percentil {percentil:.1f}% de la distribución estocástica, lo que indica una concentración baja respecto a la norma simulada."
        elif 33 <= percentil <= 66:
            clasificacion_teorica = "Concentración Moderada"
            justificacion = f"El valor se sitúa en el percentil {percentil:.1f}%, reflejando una concentración intermedia."
        else:
            clasificacion_teorica = "Alta Concentración / Mercado Concentrado"
            justificacion = f"El valor se sitúa en el percentil {percentil:.1f}%, estando entre los escenarios de mayor concentración."

    # Retroalimentación automatizada
    if respuesta_usuario == clasificacion_teorica:
        st.success(f"✅ **¡Correcto!** {justificacion}")
    else:
        st.error(f"❌ **Incorrecto.** La clasificación adecuada es: **{clasificacion_teorica}**.")
        st.info(f"💡 **Justificación Técnica:** {justificacion}")

    st.markdown(f"**Posición Cuantitativa Relativa:** El caso particular se ubica en el **percentil {percentil:.1f}%** de la distribución empírica de Monte Carlo.")
    
