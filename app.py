import streamlit as st
import base64

# 🔹 Configuración de la página
st.set_page_config(
    page_title="Predicción USD/EUR – Grupo Procourval",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔹 Estilo visual corporativo
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.85);
                padding: 2rem;
                border-radius: 10px;
            }}
            h1, h2, h3 {{
                color: #003366;
            }}
            .metric-label {{
                font-weight: bold;
                color: #003366;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"No se pudo cargar el fondo: {e}")

set_background("fondo.jpg")  # Asegúrate de que exista en tu carpeta

# 🔹 Logo institucional
try:
    st.image("logo_completo.jpg", width=900)
except Exception as e:
    st.warning(f"⚠️ Error al cargar el logo: {e}")

# 🔹 Encabezado
st.title("Predicción del tipo de cambio USD/EUR")
st.subheader("Análisis y predicciones basadas en datos históricos y modelos avanzados de machine learning.")
st.markdown("Grupo Procourval – Departamento de Datos")

import requests
import pandas as pd
import time
from datetime import datetime

# 🔹 Consulta del tipo de cambio USD → EUR desde FRED
def get_usdeur_real():
    try:
        start = time.time()
        response = requests.get("https://api.stlouisfed.org/fred/series/observations", params={
            "series_id": "DEXUSEU",
            "api_key": "437ffc22620f0fe3615350b1764f112b",
            "file_type": "json"
        })
        latency = round((time.time() - start) * 1000, 2)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["observations"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["value"], inplace=True)
        latest = df.sort_values("date", ascending=False).iloc[0]

        return {
            "valor": round(latest["value"], 4),
            "fecha": latest["date"].strftime("%Y-%m-%d"),
            "latencia": latency,
            "estado": "✅ FRED OK"
        }
    except Exception as e:
        return {
            "valor": None,
            "fecha": None,
            "latencia": None,
            "estado": f"❌ Error FRED: {e}"
        }
# 🔹 Mostrar KPIs visuales
info = get_usdeur_real()
st.markdown("### Valor real del dólar frente al euro (USD → EUR)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("1 USD =", f"{info['valor']} EUR" if info['valor'] else "N/A")
col2.metric("⏱ Latencia", f"{info['latencia']} ms" if info['latencia'] else "N/A")
col3.markdown(f"**📅 Fecha:** {info['fecha']}")
col4.markdown(f"**🔄 Estado API:** {info['estado']}")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 🔹 Cargar archivo CSV generado desde Jupyter
df = pd.read_csv("comparativa_dxy_modelo.csv", parse_dates=["Fecha"])
df = df.set_index("Fecha")

# 🔹 Extraer series
y_real = df["DXY real"]
y_pred = df["DXY estimado"]

# 🔹 Calcular métricas clave
mae = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
bias = np.mean(y_pred - y_real)
direccion_correcta = np.mean(np.sign(np.diff(y_real)) == np.sign(np.diff(y_pred))) * 100

# 🔹 Resumen ejecutivo para audiencia no técnica
st.markdown("###  Resumen del rendimiento del modelo")
st.markdown(f"""
Durante el periodo analizado, el modelo logró una precisión notable al estimar el valor del dólar frente a una cesta de divisas (índice DXY).  
- El **error medio absoluto** fue de **{mae:.2f} puntos**, lo que indica una desviación promedio muy baja.  
- El **MAPE**, que mide el error relativo, se situó en **{mape:.2f}%**, lo que es aceptable para entornos financieros.  
- El modelo acertó la **dirección del movimiento** del dólar en un **{direccion_correcta:.2f}%** de los días, lo que lo hace útil para decisiones tácticas.  
- La **confianza media** del modelo fue de **{df['Confianza'].mean():.2f}**, basada en la dispersión de Prophet.

En resumen, el modelo muestra un comportamiento estable y fiable, especialmente en escenarios neutros.
""")

# 🔹 Gráfico Plotly comparativo
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index, y=y_real,
    mode='lines', name='DXY real',
    line=dict(color='black', width=2)
))

fig.add_trace(go.Scatter(
    x=df.index, y=y_pred,
    mode='lines', name='DXY estimado (escenario neutro)',
    line=dict(color='#003366', dash='dash')
))

fig.update_layout(
    title=" Evolución del dólar – Real vs Modelo",
    xaxis_title="Fecha",
    yaxis_title="Índice DXY",
    legend=dict(x=0, y=1),
    margin=dict(l=40, r=40, t=60, b=60),
    height=500,
    annotations=[
        dict(
            text="Esta gráfica representa el escenario neutro del modelo. En versiones anteriores se mostraban también los escenarios positivo y negativo, que encapsulaban el valor estimado del dólar dentro de un rango de confianza.",
            xref="paper", yref="paper",
            x=0, y=-0.3, showarrow=False,
            font=dict(size=12, color="gray")
        )
    ]
)

st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 🔹 Cargar archivo Excel con datos de julio
df = pd.read_excel("comparativa_julio_2025.xlsx", parse_dates=["Fecha"])
df = df.set_index("Fecha")

# 🔹 Métricas de rendimiento
total = len(df)
aciertos = df["Acierto"].value_counts().get("✔️", 0)
errores = df["Acierto"].value_counts().get("❌", 0)
confianza_media = df["Confianza"].mean()
porcentaje_acierto = (aciertos / total) * 100
porcentaje_error = (errores / total) * 100

# 🔹 Evaluación analítica con explicación clara
st.markdown("###  Diagnóstico financiero del modelo – Julio 2025")

if porcentaje_acierto >= 80 and confianza_media >= 0.15:
    st.success("✅ Alta fiabilidad: el modelo fue preciso y estable.")
    st.markdown("""
    El diagnóstico se basa en dos parámetros clave:
    - **Porcentaje de acierto**: el modelo acertó en más del 80 % de los días, lo que indica una excelente capacidad para anticipar correctamente la dirección del dólar.
    - **Confianza media**: con un valor superior a 0.15, el modelo mostró estabilidad en sus estimaciones, lo que refuerza su utilidad en entornos financieros.

    Este rendimiento sugiere que el modelo puede utilizarse para tomar decisiones operativas con seguridad, especialmente en contextos de riesgo controlado.
    """)
elif porcentaje_acierto >= 65:
    st.warning("⚠️ Fiabilidad moderada: útil con supervisión.")
    st.markdown("""
    Aunque el modelo acertó entre el 65 % y el 80 % de las veces, lo que indica una fiabilidad aceptable, no alcanza niveles óptimos.
    La confianza media es razonable, pero se recomienda que las decisiones basadas en este modelo sean revisadas por un analista, especialmente en días con alta volatilidad o eventos macroeconómicos relevantes.
    """)
else:
    st.error("❌ Fiabilidad baja: el modelo requiere revisión.")
    st.markdown("""
    El porcentaje de acierto fue inferior al 65 %, lo que indica que el modelo no logró anticipar correctamente la mayoría de los movimientos del dólar.
    Además, si la confianza media es baja, las predicciones pueden haber sido inconsistentes o erráticas.
    En este caso, se recomienda revisar el modelo, ajustar sus parámetros, o incorporar nuevas variables que mejoren su capacidad predictiva.
    """)

# 🔹 Gráfico refinado de aciertos vs errores
st.markdown("###  Distribución de aciertos y errores en julio")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
colores = ["#0072B5", "#D55E00"]  # Azul corporativo y rojo financiero
barras = ax.bar(["✔️ Aciertos", "❌ Errores"], [aciertos, errores], color=colores, width=0.5)

# Añadir etiquetas encima de cada barra
for barra in barras:
    altura = barra.get_height()
    ax.text(barra.get_x() + barra.get_width() / 2, altura + 0.5, f"{int(altura)}", ha='center', va='bottom', fontsize=10)

ax.set_ylabel("Número de días", fontsize=9)
ax.set_title("Predicciones correctas vs incorrectas – Julio 2025", fontsize=10, color="#003366")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis="y", linestyle="--", alpha=0.3)
st.pyplot(fig)
st.markdown("### ¿Qué entendemos por acierto y error en este modelo?")
st.markdown("""
En este contexto, un **acierto** significa que el modelo predijo correctamente la dirección del movimiento del dólar respecto al día anterior.  
Es decir, si el dólar subió y el modelo anticipó una subida, o si bajó y el modelo anticipó una bajada, se considera un acierto.

Por el contrario, un **error** ocurre cuando el modelo predijo una dirección contraria a la que realmente sucedió.  
Por ejemplo, si el modelo anticipó una subida pero el dólar bajó, eso se contabiliza como error.

---

###  Parámetros que se interpretan para cada caso:

- **DXY real**: es el valor real del índice del dólar en ese día, calculado frente a una cesta de divisas.
- **Variación real**: indica si el dólar subió o bajó respecto al día anterior.
- **Predicción modelo**: es la estimación que hizo el modelo sobre el comportamiento del dólar.
- **Acierto**: se marca como ✔️ si la predicción coincidió con la dirección real, y ❌ si no lo hizo.
- **Confianza**: representa la seguridad del modelo en su predicción, calculada a partir de la dispersión del intervalo de Prophet. Cuanto mayor sea la confianza, más estable es la estimación.

Este análisis no mide si el valor exacto fue idéntico, sino si el modelo fue capaz de anticipar correctamente la **tendencia** del mercado, lo cual es clave en entornos financieros donde la dirección del movimiento es más relevante que el valor absoluto.
""")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 🔹 Cargar predicciones futuras
df_pred = pd.read_csv("predicciones_2025_2029.csv", parse_dates=["Fecha"])
df_pred = df_pred.set_index("Fecha")

# 🔹 Convertir índice a lista de fechas disponibles
fechas_disponibles = df_pred.index.to_list()

# 🔹 Selector de fecha
st.markdown("###  Selecciona una fecha para consultar la predicción del dólar")
fecha_seleccionada = st.date_input(
    "Fecha de predicción",
    value=fechas_disponibles[0],
    min_value=fechas_disponibles[0],
    max_value=fechas_disponibles[-1]
)

# 🔹 Convertir fecha seleccionada a Timestamp
fecha_seleccionada = pd.to_datetime(fecha_seleccionada)

# 🔹 Verificar si hay predicción para esa fecha
if fecha_seleccionada not in df_pred.index:
    st.warning("⚠️ No hay predicción disponible para esa fecha. Intenta con un día laborable.")
else:
    fila = df_pred.loc[fecha_seleccionada]
    valor_estimado = fila["DXY estimado"]
    inferior = fila["yhat_lower"]
    superior = fila["yhat_upper"]
    confianza = fila["Confianza"]

    # 🔹 Mostrar métricas
    st.markdown(f"### Predicción para {fecha_seleccionada.strftime('%d/%m/%Y')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor estimado (DXY)", f"{valor_estimado:.2f}")
    col2.metric("Rango de confianza", f"{inferior:.2f} – {superior:.2f}")
    col3.metric("Confianza del modelo", f"{confianza:.2f}")

    # 🔹 Gráfico interactivo con Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[fecha_seleccionada], y=[valor_estimado],
        mode='markers', name='Estimación central',
        marker=dict(color="#003366", size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_seleccionada, fecha_seleccionada],
        y=[inferior, superior],
        mode='lines', name='Intervalo de confianza',
        line=dict(color='gray', width=2, dash='dot')
    ))
    fig.update_layout(
        title="Estimación del dólar con intervalo de confianza",
        xaxis_title="Fecha",
        yaxis_title="Índice DXY",
        height=400,
        showlegend=True,
        annotations=[
            dict(
                text="Esta estimación representa el escenario neutro del modelo. En versiones anteriores se mostraban también los escenarios positivo y negativo, que encapsulaban el valor del dólar dentro de un rango de confianza.",
                xref="paper", yref="paper",
                x=0, y=-0.3, showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

    # 🔹 Explicación para audiencia no técnica
    st.markdown("### ¿Cómo interpretar esta predicción?")
    st.markdown("""
    Esta estimación se basa en un modelo estadístico que incorpora variables macroeconómicas como:
    - **VIX**: índice de volatilidad del mercado
    - **Inflación en EE.UU.**
    - **Tasa de interés de la Reserva Federal

    El modelo calcula un valor central para el dólar (DXY) y un **rango de confianza** que representa la posible variación esperada.  
    Cuanto más estrecho sea ese rango, mayor es la **confianza** del modelo en su predicción.  
    Esta herramienta permite anticipar el comportamiento del dólar en fechas futuras, lo que puede ser útil para planificación financiera, cobertura de riesgo o toma de decisiones estratégicas.
    """)
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 🔹 Cargar archivo con escenarios
df_escenarios = pd.read_csv("escenarios_dxy_2025_2029.csv", parse_dates=["Fecha"])
df_escenarios = df_escenarios.set_index("Fecha")

# 🔹 Selector de fecha
st.markdown("###  Simulación de escenarios para el dólar")
fecha_simulada = st.date_input(
    "Selecciona una fecha",
    value=df_escenarios.index[0],
    min_value=df_escenarios.index.min(),
    max_value=df_escenarios.index.max()
)

fecha_simulada = pd.to_datetime(fecha_simulada)

if fecha_simulada not in df_escenarios.index:
    st.warning("⚠️ No hay datos disponibles para esa fecha.")
else:
    fila = df_escenarios.loc[fecha_simulada]
    dxy_neutro = fila["DXY_neutro"]
    dxy_positivo = fila["DXY_positivo"]
    dxy_negativo = fila["DXY_negativo"]
    inferior = fila["yhat_lower"]
    superior = fila["yhat_upper"]
    confianza = fila["Confianza"]

    # 🔹 Mostrar métricas
    st.markdown(f"### Predicciones para {fecha_simulada.strftime('%d/%m/%Y')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Escenario neutro", f"{dxy_neutro:.2f}")
    col2.metric("Escenario positivo", f"{dxy_positivo:.2f}")
    col3.metric("Escenario negativo", f"{dxy_negativo:.2f}")

    # 🔹 Gráfico interactivo
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[fecha_simulada], y=[dxy_neutro],
        mode='markers', name='Neutro',
        marker=dict(color="#003366", size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_simulada], y=[dxy_positivo],
        mode='markers', name='Positivo',
        marker=dict(color="green", size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_simulada], y=[dxy_negativo],
        mode='markers', name='Negativo',
        marker=dict(color="red", size=12)
    ))
    fig.add_trace(go.Scatter(
        x=[fecha_simulada, fecha_simulada],
        y=[inferior, superior],
        mode='lines', name='Intervalo de confianza (neutro)',
        line=dict(color='gray', width=2, dash='dot')
    ))
    fig.update_layout(
        title="Simulación de escenarios del dólar",
        xaxis_title="Fecha",
        yaxis_title="Índice DXY",
        height=450,
        showlegend=True,
        annotations=[
            dict(
                text="El escenario neutro representa condiciones macroeconómicas estables. Los escenarios positivo y negativo encapsulan el valor del dólar bajo condiciones favorables o adversas, respectivamente.",
                xref="paper", yref="paper",
                x=0, y=-0.3, showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

    # 🔹 Explicación para audiencia no técnica
    st.markdown("###  ¿Qué representan estos escenarios?")
    st.markdown("""
    - **Escenario neutro**: se basa en condiciones macroeconómicas promedio, sin shocks ni cambios bruscos.
    - **Escenario positivo**: simula un entorno favorable, con baja volatilidad, inflación controlada y política monetaria estable.
    - **Escenario negativo**: representa un contexto adverso, con alta volatilidad, inflación elevada o subidas agresivas de tipos.

    Estos escenarios permiten anticipar cómo podría comportarse el dólar en distintos contextos, lo que es útil para análisis de riesgo, cobertura financiera y toma de decisiones estratégicas.
    """)
import streamlit as st
import pandas as pd

# 🔹 Cargar predicciones futuras
df_pred = pd.read_csv("predicciones_2025_2029.csv", parse_dates=["Fecha"])
df_pred = df_pred.set_index("Fecha")

# 🔹 Selector de rango de fechas
st.markdown("### Generar informe de predicción")
fecha_inicio = st.date_input("Fecha inicial", value=pd.to_datetime("2025-09-10"))
fecha_fin = st.date_input("Fecha final", value=pd.to_datetime("2025-09-18"))

# 🔹 Convertir fechas
fecha_inicio = pd.to_datetime(fecha_inicio)
fecha_fin = pd.to_datetime(fecha_fin)

# 🔹 Filtrar datos
df_rango = df_pred.loc[(df_pred.index >= fecha_inicio) & (df_pred.index <= fecha_fin)]

if df_rango.empty:
    st.warning("⚠️ No hay predicciones disponibles en ese rango.")
else:
    # 🔹 Mostrar tabla resumen
    st.markdown("###  Informe de predicción")
    st.dataframe(df_rango.style.format({
        "DXY estimado": "{:.2f}",
        "yhat_lower": "{:.2f}",
        "yhat_upper": "{:.2f}",
        "Dispersión": "{:.2f}",
        "Confianza": "{:.2f}"
    }))

    # 🔹 Métricas agregadas
    st.markdown("###  Métricas del periodo seleccionado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Valor medio estimado", f"{df_rango['DXY estimado'].mean():.2f}")
    col2.metric("Confianza media", f"{df_rango['Confianza'].mean():.2f}")
    col3.metric("Dispersión media", f"{df_rango['Dispersión'].mean():.2f}")

    # 🔹 Explicación para audiencia no técnica
    st.markdown("###  ¿Cómo interpretar este informe?")
    st.markdown("""
    Este informe muestra las estimaciones del valor del dólar (DXY) para el periodo seleccionado.  
    - El **valor medio estimado** representa la tendencia central del dólar en ese intervalo.  
    - La **confianza media** indica la estabilidad del modelo: valores altos sugieren predicciones más fiables.  
    - La **dispersión media** refleja la amplitud del rango de incertidumbre: cuanto menor, más precisa es la estimación.

    Este tipo de informe puede utilizarse para planificación financiera, análisis de riesgo o validación de escenarios futuros.
    """)
import streamlit as st
import pandas as pd

# 🔹 Cargar predicciones futuras
df_pred = pd.read_csv("predicciones_2025_2029.csv", parse_dates=["Fecha"])
df_pred = df_pred.set_index("Fecha")

# 🔹 Calcular variación diaria del valor estimado
df_pred["Variación estimada"] = df_pred["DXY estimado"].diff()

# 🔹 Definir umbrales de alerta
umbral_dispersion = df_pred["Dispersión"].mean() + df_pred["Dispersión"].std()
umbral_variacion = df_pred["Variación estimada"].std() * 2

# 🔹 Detectar anomalías
df_alertas = df_pred[
    (df_pred["Dispersión"] > umbral_dispersion) |
    (df_pred["Variación estimada"].abs() > umbral_variacion)
]

# 🔹 Mostrar resumen
st.markdown("### Días con alerta en predicción del dólar")
st.markdown(f"""
Se han detectado **{len(df_alertas)} días** en los que el modelo muestra señales de baja fiabilidad o comportamiento anómalo.  
Estos días pueden requerir revisión adicional por parte del equipo analítico.
""")

# 🔹 Función de estilo para resaltar en rojo
def resaltar_alerta(fila):
    color = "background-color: #ffcccc"  # Rojo claro
    return [color] * len(fila)

# 🔹 Mostrar tabla con estilo
st.dataframe(
    df_alertas[[
        "DXY estimado", "yhat_lower", "yhat_upper", "Dispersión", "Confianza", "Variación estimada"
    ]].style
    .format({
        "DXY estimado": "{:.2f}",
        "yhat_lower": "{:.2f}",
        "yhat_upper": "{:.2f}",
        "Dispersión": "{:.2f}",
        "Confianza": "{:.2f}",
        "Variación estimada": "{:+.2f}"
    })
    .apply(resaltar_alerta, axis=1)
)

# 🔹 Explicación para audiencia no técnica
st.markdown("###  ¿Qué significa una alerta?")
st.markdown("""
Una alerta se genera cuando el modelo muestra alguno de estos comportamientos:
- **Alta dispersión**: el rango entre la predicción inferior y superior es demasiado amplio, lo que indica baja confianza.
- **Variación abrupta**: el valor estimado del dólar cambia bruscamente respecto al día anterior, lo que puede reflejar sensibilidad excesiva o ruido en los datos.

Las filas resaltadas en rojo indican días que requieren especial atención.  
No significa que el modelo esté fallando, pero sí que conviene revisar el contexto económico de esos días antes de tomar decisiones basadas en esas predicciones.
""")
import streamlit as st

st.markdown("##  Metodología del modelo de predicción USD/EUR")

st.markdown("""
Este modelo ha sido desarrollado por el departamento de datos del **Grupo Procurval** con el objetivo de anticipar el comportamiento del dólar frente al euro (USD/EUR) utilizando el índice DXY como referencia.

###  Datos utilizados
- **Histórico del índice DXY** desde 2010 hasta la actualidad
- **Variables macroeconómicas** como:
  - Índice de volatilidad (**VIX**)
  - Inflación en EE.UU. (**Inflacion_USA**)
  - Tasa de interés de la Reserva Federal (**Tasa_FED**)
- Datos extraídos de fuentes oficiales como **FRED**, **Yahoo Finance** y bases internas del grupo

### Técnicas aplicadas
- **Modelado temporal con Prophet** para estimar el valor futuro del dólar
- **Clasificación con Random Forest** para evaluar decisiones operativas
- **Simulación de escenarios** (positivo, negativo, neutro) para análisis estratégico
- **Evaluación con métricas financieras** como MAE, RMSE, MAPE y dispersión

### Limitaciones del modelo
- Las predicciones se basan en condiciones macroeconómicas estimadas; eventos inesperados (geopolíticos, financieros) pueden alterar los resultados
- El modelo no sustituye el juicio experto, sino que lo complementa
- La confianza del modelo varía según la dispersión del intervalo de predicción

###  Recomendaciones de uso
- Utilizar el modelo como **herramienta de apoyo** para decisiones tácticas y estratégicas
- Revisar los días con **alerta** antes de ejecutar decisiones sensibles
- Consultar los escenarios alternativos para evaluar riesgos y oportunidades

---

## Cierre del proyecto

Este proyecto ha sido diseñado para ofrecer una solución predictiva robusta, transparente y adaptable al entorno actual.  
El modelo permite anticipar el comportamiento del dólar con fiabilidad y flexibilidad.

**Cristina Puertas**
**Departamento de Data – Grupo Procourval**
            cpuertas@gpsc.es
---

###  Gracias por utilizar esta app

Esta aplicación está en constante evolución. Si tienes sugerencias, mejoras o nuevas variables que se necesiten incorporar dispone de un repositorio en gitHub para gestionar incidencias y propuestas de mejora.
https://github.com/cpuertas-gpsc/Cambio-divisas
            **Grupo Procourval – Departamento de Datos**
""")
