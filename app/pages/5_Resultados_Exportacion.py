"""
Página 5: Resultados y exportación.

OBJETIVO DE LA PÁGINA
---------------------
- Recopilar los resultados generados en las páginas anteriores:
    * Datos originales y preprocesados.
    * Resultados de PCA (scores, loadings, varianza explicada).
    * Resultados de clustering (labels y métricas).
- Permitir a la persona usuaria:
    1) Visualizar un resumen rápido del estado del análisis.
    2) Descargar tablas en formato CSV (o JSON en el caso de métricas).
    3) Guardar figuras clave en formato HTML interactivo (Plotly).
    4) Ajustar la paleta de colores global para las gráficas de la app.

ESTADO COMPARTIDO UTILIZADO
---------------------------
Lectura de:
- raw_df
- clean_df
- pca_scores
- pca_loadings
- pca_explained_variance
- cluster_labels
- cluster_metrics
- plot_color_palette

Escritura de:
- plot_color_palette (si la persona usuaria cambia la paleta).
"""

from __future__ import annotations

import io
import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Asegurar que la raíz del proyecto esté en el path (por si se necesitan utils)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convertir un DataFrame a bytes CSV con codificación UTF-8."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=True)
    return buffer.getvalue().encode("utf-8")


def _fig_to_image_bytes(fig, fmt: str = "png") -> bytes:
    """
    Convertir una figura de Plotly a bytes de imagen.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    fmt : {"png", "svg"}

    Returns
    -------
    bytes
    """
    return fig.to_image(format=fmt)


def _dict_to_json_bytes(data: Dict) -> bytes:
    """Convertir un diccionario a bytes JSON."""
    buffer = io.StringIO()
    pd.json_normalize(data).to_json(buffer, orient="records", force_ascii=False)
    return buffer.getvalue().encode("utf-8")





# ---------------------------------------------------------------------------
# Secciones de la UI
# ---------------------------------------------------------------------------


def render_header() -> None:
    """Encabezado y explicación general de la página de resultados."""
    st.set_page_config(page_title="Resultados y exportación")
    st.title("Resultados y exportación")

    st.markdown(
        """
        En esta página puedes **revisar y descargar** los resultados generados en el
        flujo de análisis quimiométrico:

        - Datos originales y preprocesados.
        - Scores y loadings del PCA.
        - Varianza explicada por componente.
        - Etiquetas y métricas de clustering.
        - Figuras clave en formato HTML interactivo.

        Usa esta sección al final del análisis para documentar resultados,
        compartirlos o continuar el trabajo en otras herramientas.
        """
    )


def render_pipeline_status() -> None:
    """Mostrar un resumen rápido del estado del pipeline."""

    st.subheader("Estado del análisis")

    raw_df = st.session_state.get("raw_df")
    clean_df = st.session_state.get("clean_df")
    pca_scores = st.session_state.get("pca_scores")
    cluster_labels = st.session_state.get("cluster_labels")

    cols = st.columns(4)
    cols[0].metric(
        "Datos cargados",
        "Sí" if raw_df is not None else "No",
    )
    cols[1].metric(
        "Preprocesamiento",
        "OK" if clean_df is not None else "Pendiente",
    )
    cols[2].metric(
        "PCA",
        "Calculado" if pca_scores is not None else "No calculado",
    )
    cols[3].metric(
        "Clustering",
        "Con etiquetas" if cluster_labels is not None else "No aplicado",
    )

    st.caption(
        "Consejo: si alguna etapa está pendiente, revisa las páginas anteriores "
        "antes de descargar resultados."
    )


def render_color_palette_selector() -> None:
    """Selector global de paleta de colores para las gráficas."""
    st.subheader("Paleta de colores para gráficas")

    default_palette = st.session_state.get("plot_color_palette", "deep")
    palettes = ["deep", "muted", "pastel", "bright", "dark", "colorblind", "viridis"]
    choice = st.selectbox(
        "Seleccione la paleta de colores base",
        palettes,
        index=palettes.index(default_palette) if default_palette in palettes else 0,
    )
    st.session_state["plot_color_palette"] = choice
    st.caption(
        "Esta paleta se puede reutilizar en las páginas de PCA y clustering "
        "para mantener una estética consistente."
    )


def render_data_exports() -> None:
    """Sección para descargar tablas de resultados en CSV."""

    st.subheader("Descarga de datos y resultados tabulares")

    raw_df: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    clean_df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    pca_scores: Optional[pd.DataFrame] = st.session_state.get("pca_scores")
    pca_loadings: Optional[pd.DataFrame] = st.session_state.get("pca_loadings")
    explained_df: Optional[pd.DataFrame] = st.session_state.get("pca_explained_variance")
    cluster_labels = st.session_state.get("cluster_labels")
    cluster_metrics: Optional[Dict] = st.session_state.get("cluster_metrics")

    # --- Datos de entrada / preprocesados ---
    if raw_df is not None or clean_df is not None:
        st.markdown("**Datos de entrada y preprocesamiento**")
        cols = st.columns(2)

        if raw_df is not None:
            with cols[0]:
                st.caption("Datos originales (tal como se cargaron).")
                st.download_button(
                    "⬇️ Descargar datos originales (CSV)",
                    data=_df_to_csv_bytes(raw_df),
                    file_name="datos_originales.csv",
                    mime="text/csv",
                )

        if clean_df is not None:
            with cols[1]:
                st.caption("Datos después del preprocesamiento.")
                st.download_button(
                    "⬇️ Descargar datos preprocesados (CSV)",
                    data=_df_to_csv_bytes(clean_df),
                    file_name="datos_preprocesados.csv",
                    mime="text/csv",
                )

        st.divider()

    # --- Resultados de PCA ---
    if pca_scores is not None or pca_loadings is not None or explained_df is not None:
        st.markdown("**Resultados de PCA**")
        cols = st.columns(3)

        if pca_scores is not None:
            export_scores = pca_scores.copy()
            if cluster_labels is not None and len(cluster_labels) == len(export_scores):
                export_scores["cluster"] = cluster_labels
            with cols[0]:
                st.caption("Scores de PCA por muestra (incluye clúster si existe).")
                st.download_button(
                    "⬇️ Descargar scores de PCA (CSV)",
                    data=_df_to_csv_bytes(export_scores),
                    file_name="pca_scores.csv",
                    mime="text/csv",
                )

        if pca_loadings is not None:
            with cols[1]:
                st.caption("Loadings de PCA (contribución de cada variable).")
                st.download_button(
                    "⬇️ Descargar loadings de PCA (CSV)",
                    data=_df_to_csv_bytes(pca_loadings),
                    file_name="pca_loadings.csv",
                    mime="text/csv",
                )

        if explained_df is not None:
            with cols[2]:
                st.caption("Varianza explicada por componente.")
                st.download_button(
                    "⬇️ Descargar tabla de varianza explicada (CSV)",
                    data=_df_to_csv_bytes(explained_df),
                    file_name="pca_varianza_explicada.csv",
                    mime="text/csv",
                )
                
            # --- Archivo integrado de resultados (scores + labels) ---
        if pca_scores is not None:
            st.markdown("**Archivo integrado de resultados**")

            results_df = pca_scores.copy()

            # Adjuntar etiquetas de clúster si existen
            if cluster_labels is not None and len(cluster_labels) == len(results_df):
                results_df["cluster"] = np.asarray(cluster_labels)

            st.caption(
                "Este archivo incluye los scores de PCA por muestra y, si están "
                "disponibles, las etiquetas de clúster."
            )
            st.download_button(
                "⬇️ Descargar resultados (scores + clusters) en CSV",
                data=_df_to_csv_bytes(results_df),
                file_name="resultados_pca_clustering.csv",
                mime="text/csv",
            )


        st.divider()

    # --- Resultados de clustering ---
    if cluster_labels is not None or cluster_metrics is not None:
        st.markdown("**Resultados de clustering**")
        cols = st.columns(2)

        if cluster_labels is not None:
            labels_df = pd.DataFrame({"cluster": np.asarray(cluster_labels)}, index=clean_df.index if clean_df is not None else None)
            with cols[0]:
                st.caption("Etiquetas de clúster por muestra.")
                st.download_button(
                    "⬇️ Descargar etiquetas de clúster (CSV)",
                    data=_df_to_csv_bytes(labels_df),
                    file_name="clustering_labels.csv",
                    mime="text/csv",
                )

        if cluster_metrics:
            with cols[1]:
                st.caption("Métricas globales de clustering (ej. silhouette, inertia).")
                st.json(cluster_metrics)
                st.download_button(
                    "⬇️ Descargar métricas de clustering (JSON)",
                    data=_dict_to_json_bytes(cluster_metrics),
                    file_name="clustering_metrics.json",
                    mime="application/json",
                )


def render_figures_section() -> None:
    """
    Mostrar figuras clave sin manejar exportación desde Python.
    El usuario puede exportar cada gráfica usando el ícono de cámara
    integrado en Plotly (PNG / SVG / JPG).
    """

    st.subheader("Figuras y visualizaciones")

    explained_df: Optional[pd.DataFrame] = st.session_state.get("pca_explained_variance")
    pca_scores: Optional[pd.DataFrame] = st.session_state.get("pca_scores")
    cluster_labels = st.session_state.get("cluster_labels")

    # ========================
    # Scree plot
    # ========================
    if explained_df is not None and not explained_df.empty:
        st.markdown("**Scree plot (PCA)**")

        fig_scree = px.bar(
            explained_df,
            x="Componente",
            y="Proporción_varianza",
            title="Varianza explicada por componente",
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        st.caption(
            "Para exportar esta figura en PNG o SVG, use el botón de cámara "
            "que aparece en la esquina superior derecha del gráfico."
        )

    # ========================
    # Scores PC1 vs PC2
    # ========================
    if pca_scores is not None and not pca_scores.empty:

        st.markdown("**Scores de PCA (PC1 vs PC2)**")

        if "PC1" in pca_scores.columns and "PC2" in pca_scores.columns:
            plot_df = pca_scores.copy()

            color_arg = None
            if cluster_labels is not None and len(cluster_labels) == len(plot_df):
                plot_df["cluster"] = cluster_labels
                color_arg = "cluster"

            fig_scores = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color=color_arg,
                title="Scores PC1 vs PC2",
                height=700
            )
            st.plotly_chart(fig_scores, use_container_width=True)

            st.caption(
                "Para exportar esta figura en PNG o SVG, haga clic en el ícono "
                "de cámara dentro del gráfico. Plotly genera automáticamente la imagen."
            )

        else:
            st.info(
                "No se encontraron columnas 'PC1' y 'PC2' en los scores de PCA. "
                "Verifique la configuración de PCA en la página correspondiente."
            )

    if explained_df is None and pca_scores is None:
        st.info(
            "Aún no se han generado resultados de PCA suficientes para crear figuras. "
            "Revise la página de PCA si desea visualizarlas aquí."
        )

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    render_header()
    render_pipeline_status()
    render_color_palette_selector()
    render_data_exports()
    render_figures_section()


if __name__ == "__main__":
    main()
