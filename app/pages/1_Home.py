"""Página de inicio para la aplicación de análisis quimiométrico.

Lee dataset de ejemplo o archivos subidos por la persona usuaria y prepara el
``session_state`` compartido. Los pasos para implementar están descritos a lo
largo del archivo como comentarios.
"""

from __future__ import annotations
from typing import List
import streamlit as st

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.io_utils import (
    describe_dataframe,
    load_example_dataset,
    load_uploaded_dataset,
)


def _reset_analysis_state(preserved_keys: List[str] | None = None) -> None:
    """Limpiar resultados en cascada cuando se carga un nuevo dataset."""

    preserved_keys = preserved_keys or []
    keys_to_clear = [
        "schema",
        "clean_df",
        "pca_model",
        "pca_scores",
        "pca_loadings",
        "pca_explained_variance",
        "cluster_model",
        "cluster_labels",
        "cluster_metrics",
    ]
    for key in keys_to_clear:
        if key not in preserved_keys:
            st.session_state.pop(key, None)


def init_session_state() -> None:
    """Inicializar llaves necesarias en ``st.session_state``.

    - Garantiza que exista la paleta de colores global.
    - Prepara contenedores para ``raw_df`` y ``data_source`` sin sobrescribir
      valores ya cargados.
    """

    st.session_state.setdefault("plot_color_palette", "deep")
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("data_source", None)


def render_header() -> None:
    """Mostrar título y descripción general de la aplicación."""

    st.set_page_config(page_title="Inicio - Análisis Quimiométrico", page_icon="🧪")
    st.title("Análisis Quimiométrico: Aplicación Interactiva")
    st.markdown(
        """
        Esta aplicación multipágina guía el flujo típico de un análisis
        quimiométrico: desde la carga y preprocesamiento de datos, pasando por
        la reducción de dimensionalidad con PCA, hasta la generación de
        agrupamientos (clustering) y la exportación de resultados.
        """
    )


def render_dataset_selector() -> None:
    """UI para elegir el dataset de ejemplo o subir un archivo propio."""

    st.header("Carga de datos")
    st.markdown(
        "Seleccione entre cargar el dataset de ejemplo incluido o subir su propio archivo (CSV o Excel)."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Usar dataset de ejemplo")
        st.caption("El archivo se encuentra en la carpeta data/ejemplo_quimiometria.xls")
        if st.button("Cargar datos de ejemplo", type="primary"):
            try:
                df_example = load_example_dataset()
                st.session_state["raw_df"] = df_example
                st.session_state["data_source"] = "ejemplo"
                _reset_analysis_state(preserved_keys=["raw_df", "data_source", "plot_color_palette"])
                st.success("Datos cargados correctamente desde el dataset de ejemplo.")
            except Exception as exc:  # pragma: no cover - feedback para el usuario
                st.error(f"No se pudieron cargar los datos de ejemplo: {exc}")

    with col2:
        st.subheader("Subir archivo propio")
        uploaded_file = st.file_uploader(
            "Seleccione un archivo CSV o Excel",
            type=["csv", "xls", "xlsx", "xlsm"],
            help="Los datos deben estar en formato tabular. Las hojas de Excel solo leerán la primera hoja.",
        )
        if uploaded_file is not None:
            try:
                df_uploaded = load_uploaded_dataset(uploaded_file)
                st.session_state["raw_df"] = df_uploaded
                st.session_state["data_source"] = uploaded_file.name
                _reset_analysis_state(preserved_keys=["raw_df", "data_source", "plot_color_palette"])
                st.success(f"Datos cargados correctamente desde '{uploaded_file.name}'.")
            except Exception as exc:  # pragma: no cover - feedback para el usuario
                st.error(f"No se pudieron cargar los datos: {exc}")


def render_dataset_preview() -> None:
    """Mostrar vista previa y resumen del dataset cargado."""

    st.divider()

    if st.session_state.get("raw_df") is None:
        st.info(
            "Todavía no hay datos cargados. Use el dataset de ejemplo o suba un archivo para continuar."
        )
        return

    st.subheader("Vista rápida de los datos")
    st.caption("Se muestran las primeras filas para confirmar la estructura de la tabla.")
    st.dataframe(st.session_state["raw_df"].head(), use_container_width=True)

    try:
        summary_df = describe_dataframe(st.session_state["raw_df"])
        st.table(summary_df)
    except Exception as exc:  # pragma: no cover - defensivo
        st.warning(f"No se pudo generar el resumen de datos: {exc}")


def render_navigation() -> None:
    """Mostrar enlaces a las siguientes páginas una vez que hay datos."""

    if st.session_state.get("raw_df") is None:
        return

    st.info(
        "Cuando termine de revisar, continúe con el preprocesamiento para definir tipos de variables y limpieza."
    )
    navigation_cols = st.columns(2)
    with navigation_cols[0]:
        st.page_link("pages/2_Preprocesamiento.py", label="Ir a Preprocesamiento", icon="⚙️")
    with navigation_cols[1]:
        st.page_link("pages/0_Ayuda_Interpretacion.py", label="Ver ayuda e interpretación", icon="📖")


def main() -> None:
    init_session_state()
    render_header()
    render_dataset_selector()
    render_dataset_preview()
    render_navigation()


if __name__ == "__main__":
    main()
