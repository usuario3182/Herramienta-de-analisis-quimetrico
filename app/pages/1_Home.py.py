"""Página de inicio para la aplicación de análisis quimiométrico.

Permite cargar el dataset de ejemplo o subir un archivo propio y prepara el
estado de la aplicación para las siguientes etapas de preprocesamiento y
análisis.
"""

from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from scripts.io_utils import load_example_dataset, load_uploaded_file


def _reset_analysis_state(preserved_keys: List[str] | None = None) -> None:
    """Clear downstream cached results when a new dataset is loaded."""

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


def _ensure_palette() -> None:
    """Ensure the plot color palette is initialized once."""

    if "plot_color_palette" not in st.session_state:
        st.session_state["plot_color_palette"] = "deep"


def _store_dataset(df: pd.DataFrame, source_label: str) -> None:
    """Persist the raw dataframe in session_state and show feedback."""

    st.session_state["raw_df"] = df
    _reset_analysis_state(preserved_keys=["raw_df", "plot_color_palette"])
    st.success(f"Datos cargados correctamente desde {source_label}.")
    st.dataframe(df.head(), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Inicio - Análisis Quimiométrico", page_icon="🧪")
    _ensure_palette()

    st.title("🧪 Análisis Quimiométrico: Aplicación Interactiva")
    st.markdown(
        """
        Esta aplicación multipágina guía el flujo típico de un análisis
        quimiométrico: desde la carga y preprocesamiento de datos, pasando por
        la reducción de dimensionalidad con PCA, hasta la generación de
        agrupamientos (clustering) y la exportación de resultados.
        """
    )

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
                _store_dataset(df_example, "el dataset de ejemplo")
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
                df_uploaded = load_uploaded_file(uploaded_file)
                _store_dataset(df_uploaded, f"el archivo '{uploaded_file.name}'")
            except Exception as exc:  # pragma: no cover - feedback para el usuario
                st.error(f"No se pudieron cargar los datos: {exc}")

    st.divider()

    if "raw_df" in st.session_state:
        st.subheader("Vista rápida de los datos")
        st.caption("Se muestran las primeras filas para confirmar la estructura de la tabla.")
        st.dataframe(st.session_state["raw_df"].head(), use_container_width=True)

        st.info(
            "Cuando termine de revisar, continúe con el preprocesamiento para definir tipos de variables y limpieza."
        )

        navigation_cols = st.columns(2)
        with navigation_cols[0]:
            st.page_link(
                "pages/2_Preprocesamiento.py.py",
                label="Ir a Preprocesamiento",
                icon="⚙️",
            )
        with navigation_cols[1]:
            st.page_link("pages/0_Ayuda_Interpretacion.py.py", label="Ver ayuda e interpretación", icon="📖")
    else:
        st.info(
            "Todavía no hay datos cargados. Use el dataset de ejemplo o suba un archivo para continuar con el flujo."
        )


if __name__ == "__main__":
    main()
