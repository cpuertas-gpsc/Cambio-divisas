# Sistema de Predicción Operativa DXY  
**Felipe Castellano Construcciones & Grupo Procurval**
![Felipe Castellano Construcciones]("logo completo.jpg")
---

## Descripción del proyecto

Este sistema combina modelos de series temporales (Prophet) y clasificación supervisada (Random Forest) para anticipar decisiones operativas sobre el índice del dólar (DXY). Está diseñado como herramienta de apoyo para departamento financiero, permitiendo evaluar si conviene cambiar o no cambiar posición en función de variables macroeconómicas y señales de mercado.

---


## Componentes del sistema

- `modelo_rf_julio.pkl`: Modelo Random Forest entrenado con datos de julio 2025  
- `imputador.pkl`: Imputador para limpiar datos faltantes antes de la predicción  
- `features_rf.pkl`: Lista de columnas esperadas por el modelo  
- `comparativa_julio_2025.xlsx`: Tabla con predicciones vs. datos reales de la FED  
- `logo_completo.jpg`: Imagen corporativa principal  
- `logo_grupo.jpg`: Imagen del grupo Procurval  
- `logo_mail.jpg`: Imagen de firma institucional

---

## Variables utilizadas

- `DXY`: Índice del dólar  
- `VIX`: Volatilidad del mercado  
- `Inflacion_USA`: IPC mensual  
- `Tasa_FED`: Tasa de interés oficial  
- `dia_semana`, `mes`, `año`: Variables temporales  
- `yhat`, `dispersión`: Predicción y dispersión de Prophet  
- `confianza_prophet`: Inversa de la dispersión (mayor = más fiable)

---

## Cómo usar en Streamlit

1. Cargar el modelo y el imputador con `joblib.load()`  
2. Cargar nuevos datos económicos en formato DataFrame  
3. Aplicar el imputador y asegurar que las columnas coincidan con `features_rf.pkl`  
4. Generar predicción (`cambiar` o `no_cambiar`)  
5. Mostrar recomendación operativa junto con nivel de confianza  
6. Visualizar la tabla `comparativa_julio_2025.xlsx` para evaluar rendimiento histórico  
7. Integrar los logotipos en el encabezado y pie de la app

---

## Métricas de rendimiento (julio 2025)

- % de acierto: 80.95 %  
- Fiabilidad media: 0.13  
- Acierto en días de alta confianza: 100 %

---

## Requisitos técnicos

- Python 3.10+  
- Librerías: `prophet`, `scikit-learn`, `pandas`, `joblib`, `streamlit`, `imblearn`, `matplotlib`, `seaborn`

---

## Próximos pasos

- Integrar nuevos indicadores macroeconómicos (PMI, empleo, EUR/USD)  
- Entrenar modelo secuencial con memoria temporal  
- Desplegar versión mensual con reentrenamiento automático  
- Añadir lógica de abstención en días de baja confianza  
- Incorporar visualizaciones interactivas y filtros por fecha en Streamlit

---

## Contacto

![Firma institucional](logo_mail.jpg)

**Cristina Puertas**  
Analista de datos – Departamento de Datos  
cpuertas@gpsc.es
