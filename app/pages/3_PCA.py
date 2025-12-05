"""
Página 3: Análisis de Componentes Principales (PCA).

OBJETIVO DE LA PÁGINA:
----------------------
- Tomar el DataFrame preprocesado (clean_df) desde st.session_state.
- Permitir al usuario:
    1) Seleccionar las variables que entrarán al PCA (por defecto todas las numéricas).
    2) Elegir el número de componentes a calcular.
    3) Ejecutar el PCA y guardar resultados en st.session_state.
    4) Visualizar:
        - Tabla de varianza explicada y varianza acumulada.
        - Scree plot (gráfica de varianza explicada por componente).
        - Gráficos de scores (PC1 vs PC2, etc.).
        - Biplot básico (scores + vectores de loadings).

ESTADO COMPARTIDO UTILIZADO:
----------------------------
- Entrada principal:
    - clean_df : DataFrame preprocesado (creado en Página 2).
- Resultados de PCA (a escribir aquí):
    - pca_model : objeto PCA de scikit-learn.
    - pca_scores : DataFrame de scores (muestras x componentes).
    - pca_loadings : DataFrame de loadings (variables x componentes).
    - pca_explained_variance : DataFrame con varianza explicada.

FUNCIONES AUXILIARES EXTERNAS:
------------------------------
from scripts.pca_utils import (
    select_numeric_columns,
    run_pca,
)

Herramientas para gráficas:
---------------------------
- Se recomienda usar Plotly (plotly.express) para:
    - Scree plot.
    - Scores scatter plot (PCx vs PCy).
    - Biplot (opcionalmente, combinar con loadings).
- También se puede usar seaborn/matplotlib, pero Plotly es preferible por
  ser interactivo.

REQUISITOS DE INTERFAZ:
-----------------------
1) Encabezado con explicación breve de PCA y de lo que se muestra.
2) Panel de configuración:
    - Multiselect de columnas numéricas a incluir.
    - Slider o number_input para n_components (1..min(n_vars, 10)).
    - Botón "Calcular PCA".
3) Zona de resultados:
    - Mostrar tabla de varianza explicada (pca_explained_variance).
    - Scree plot.
    - Selector para elegir PC en eje X y Y de los scores.
    - Scatter plot de scores.
    - Biplot opcional (scores + vectores de loadings).
4) Manejo de errores:
    - Si no hay clean_df: mostrar aviso y pedir ir a Preprocesamiento.
    - Si PCA no ha sido calculado aún: mostrar mensaje apropiado.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Asegurar que scripts/ esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.pca_utils import select_numeric_columns, run_pca

def render_header() -> None:
    """Mostrar título y descripción breve de PCA."""
    # Codex: usar st.title / st.markdown con texto en español.
    pass


def render_pca_config_panel(df: pd.DataFrame) -> None:
    """
    Panel para seleccionar variables y número de componentes.

    Debe:
    - Detectar columnas numéricas.
    - Ofrecer un multiselect de columnas numéricas a incluir en el PCA.
    - Ofrecer un slider/number_input para n_components (1..min(n_vars, 10)).
    - Guardar configuración en st.session_state["pca_config"], por ejemplo:
        {
            "columns": [...],
            "n_components": 3
        }
    """
    pass


def render_run_pca_button(df: pd.DataFrame) -> None:
    """
    Botón para ejecutar PCA según la configuración actual.

    Flujo:
    - Leer configuración de st.session_state["pca_config"].
    - Usar run_pca(df, n_components, columns) de pca_utils.
    - Guardar en session_state:
        - "pca_model"
        - "pca_scores"
        - "pca_loadings"
        - "pca_explained_variance"
    - Mostrar st.success si todo sale bien.
    - Mostrar st.error si ocurre alguna excepción.
    """
    pass


def render_explained_variance_section() -> None:
    """
    Mostrar tabla y scree plot de varianza explicada.

    Requiere:
    - st.session_state["pca_explained_variance"] con columnas por componente.
    - Usar st.table o st.dataframe para la tabla.
    - Usar plotly.express para el scree plot (Componente vs Proporción_varianza).
    """
    pass


def render_scores_plots() -> None:
    """
    Mostrar gráficos de scores (PCx vs PCy).

    Requiere:
    - st.session_state["pca_scores"] con columnas PC1, PC2, ...
    - Selectbox para elegir PC en eje X y PC en eje Y.
    - Opcional: selectbox para colorear por una columna categórica de clean_df
      (por ejemplo, si hay una columna 'Muestra', 'Grupo', etc.).
    - Usar plotly.express.scatter.
    """
    pass


def render_biplot() -> None:
    """
    Mostrar un biplot básico (scores + vectores de loadings).

    Idea:
    - Usar pca_scores para las muestras (PCx, PCy).
    - Usar pca_loadings para dibujar vectores desde el origen.
    - Puede hacerse con plotly, escalando los loadings para que se vean bien.
    - Mostrar nombres de variables como etiquetas de las flechas.
    - Es aceptable que Codex implemente una versión sencilla.
    """
    pass


def main() -> None:
    """Entrypoint de la página PCA."""

    # Verificar que existen datos preprocesados
    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.error(
            "No hay datos preprocesados. "
            "Por favor, vaya a la página 'Preprocesamiento' y aplique el pipeline primero."
        )
        return

    # Inicializar contenedores de PCA
    st.session_state.setdefault("pca_config", {})
    st.session_state.setdefault("pca_model", None)
    st.session_state.setdefault("pca_scores", None)
    st.session_state.setdefault("pca_loadings", None)
    st.session_state.setdefault("pca_explained_variance", None)

    render_header()
    render_pca_config_panel(clean_df)
    render_run_pca_button(clean_df)
    render_explained_variance_section()
    render_scores_plots()
    render_biplot()


if __name__ == "__main__":
    main()
