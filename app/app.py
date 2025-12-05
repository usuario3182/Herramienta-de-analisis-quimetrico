"""app.py

Punto de entrada principal de la aplicación Streamlit multipágina
para el análisis quimiométrico.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# -----------------------------------------------------------------------------
# Ajuste de ruta para poder importar desde scripts/ si hiciera falta
# -----------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def _init_global_state() -> None:
    """Inicializar llaves globales en st.session_state si no existen todavía."""

    st.session_state.setdefault("plot_color_palette", "deep")
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("data_source", None)
    st.session_state.setdefault("clean_df", None)
    st.session_state.setdefault("pca_scores", None)
    st.session_state.setdefault("pca_loadings", None)
    st.session_state.setdefault("pca_explained_variance", None)
    st.session_state.setdefault("cluster_labels", None)
    st.session_state.setdefault("cluster_metrics", None)


def _render_pipeline_status() -> None:
    """Pequeño resumen del estado actual del flujo."""

    raw_df = st.session_state.get("raw_df")
    clean_df = st.session_state.get("clean_df")
    pca_scores = st.session_state.get("pca_scores")
    cluster_labels = st.session_state.get("cluster_labels")

    st.subheader("Estado rápido del análisis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Datos cargados", "Sí" if raw_df is not None else "No")
    c2.metric("Preprocesamiento", "OK" if clean_df is not None else "Pendiente")
    c3.metric("PCA", "Calculado" if pca_scores is not None else "No calculado")
    c4.metric("Clustering", "Con etiquetas" if cluster_labels is not None else "No aplicado")

    st.caption(
        "Sigue el flujo sugerido: Inicio → Preprocesamiento → PCA → Clustering → Resultados."
    )


def main() -> None:
    st.set_page_config(
        page_title="Análisis quimiométrico",
        page_icon="🧪",
        layout="wide",
    )

    _init_global_state()

    st.title("🧪 Análisis quimiométrico – Panel principal")

    st.markdown(
        """
        Esta aplicación guía un flujo típico de **análisis quimiométrico**:

        1. **Inicio / Carga de datos** – importa tu dataset o usa el ejemplo.
        2. **Preprocesamiento** – selección de variables, tratamiento de valores faltantes
           y escalado.
        3. **PCA** – reducción de dimensionalidad y exploración de componentes.
        4. **Clustering** – agrupamiento de muestras en el espacio de las PCs.
        5. **Resultados y exportación** – descarga de tablas y figuras para reportes.

        Usa el menú lateral de Streamlit o los accesos rápidos de abajo para navegar.
        """
    )

    _render_pipeline_status()

    st.markdown("---")
    st.subheader("Navegación rápida")

    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/0_Ayuda_Interpretacion.py", label="0. Ayuda e interpretación")
        st.page_link("pages/1_Home.py", label="1. Inicio / Carga de datos")
        st.page_link("pages/2_Preprocesamiento.py", label="2. Preprocesamiento")
    with col2:
        st.page_link("pages/3_PCA.py", label="3. PCA")
        st.page_link("pages/4_Clustering.py", label="4. Clustering")
        st.page_link(
            "pages/5_Resultados_Exportacion.py",
            label="5. Resultados y exportación",
        )

    st.markdown(
        """
        > Consejo: puedes volver a esta pantalla en cualquier momento usando el menú
        > lateral de Streamlit (página **app.py / Home**).
        """
    )


if __name__ == "__main__":
    main()
