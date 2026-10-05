import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Despliegue de proyecto final", layout="wide")

# --- TÍTULO Y SUBTÍTULO REQUERIDOS ---
st.title("Despliegue de proyecto final")
st.subheader("Presentado por: Luna Tatiana Madroñero Jaimes y Juan Jose Restrepo salamanca")

st.markdown("""
Esta aplicación web integra todo el pipeline de análisis:
Carga de datos -> Filtrado de características -> Normalización -> Predicción mediante el modelo Boosting Optimizado.
""")

# --- CARGA DE RECURSOS (Modelo y Escalador) ---
# Se definen rutas relativas para que funcione directamente al clonar en GitHub
SCALER_PATH = "min_max_scaler.joblib"
MODEL_PATH = "optimized_boosting_model.joblib"

@st.cache_resource
def load_resources():
    scaler, model = None, None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    return scaler, model

scaler, model = load_resources()

if scaler is None or model is None:
    st.warning("⚠️ No se encontraron los archivos `min_max_scaler.joblib` u `optimized_boosting_model.joblib` en el directorio actual. Por favor, asegúrate de subirlos a tu repositorio de GitHub.")

# --- SECCIÓN DE CARGA DE ARCHIVOS ---
st.subheader("📁 Carga de Datos Estudiantiles")
uploaded_file = st.file_uploader
