"""Página de PCA para configurar, calcular y visualizar resultados."""
from __future__ import annotations

from typing import List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from scripts.pca_utils import run_pca
# plaeta de colores
from scripts.io_utils import get_discrete_palette
import plotly.graph_objects as go

def render_header() -> None:
    """Encabezado explicativo de la página."""

    st.title("Análisis de Componentes Principales")
    st.markdown(
        """
        El PCA reduce la dimensionalidad agrupando la variabilidad en un número
        menor de componentes. En esta página puede elegir qué variables numéricas
        incluir, definir cuántos componentes calcular y revisar tablas y gráficos
        de varianza explicada, scores y biplots.
        """
    )


def render_pca_config_panel(df):
    """Panel lateral o sección para configurar variables y componentes."""

    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        st.error("No hay columnas numéricas disponibles para PCA.")
        return

    previous_config = st.session_state.get("pca_config", {})
    default_columns = previous_config.get("columns", numeric_columns)

    selected_columns = st.multiselect(
        "Variables numéricas para PCA",
        options=numeric_columns,
        default=default_columns,
    )

    max_components = min(len(selected_columns) if selected_columns else len(numeric_columns), 10)
    if max_components < 1:
        max_components = min(len(numeric_columns), 10)

    current_n = st.session_state.get("pca_config", {}).get("n_components", max_components)
    n_components = st.number_input(
        "Número de componentes",
        min_value=1,
        max_value=max_components,
        value=current_n if 1 <= current_n <= max_components else max_components,
        step=1,
    )

    st.session_state["pca_config"] = {
        "columns": selected_columns if selected_columns else numeric_columns,
        "n_components": int(n_components),
    }


def render_run_pca_button(df) -> None:
    """Ejecuta PCA y guarda resultados en el estado de sesión."""

    if st.button("Calcular PCA", type="primary"):
        config = st.session_state.get("pca_config", {})
        columns = config.get("columns")
        n_components = config.get("n_components")
        if not columns:
            st.error("Seleccione al menos una variable numérica para el PCA.")
            return
        try:
            model, scores_df, loadings_df, explained_df = run_pca(
                df, n_components=n_components, columns=columns
            )
            st.session_state["pca_model"] = model
            st.session_state["pca_scores"] = scores_df
            st.session_state["pca_loadings"] = loadings_df
            st.session_state["pca_explained_variance"] = explained_df
            st.success("PCA calculado correctamente.")
        except ValueError as exc:  # pragma: no cover - feedback al usuario
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - feedback al usuario
            st.error(f"Ocurrió un error al calcular el PCA: {exc}")


def render_explained_variance_section() -> None:
    """Muestra tabla y gráfico de varianza explicada."""

    explained_df = st.session_state.get("pca_explained_variance")
    if explained_df is None:
        st.info("Aún no se ha calculado el PCA.")
        return

    st.subheader("Varianza explicada")
    st.dataframe(explained_df, use_container_width=True)
    
    qualitative_palettes = {
        "Plotly": px.colors.qualitative.Plotly,
        "D3": px.colors.qualitative.D3,
        "Pastel": px.colors.qualitative.Pastel,
        "Dark24": px.colors.qualitative.Dark24,
        "Set2": px.colors.qualitative.Set2,
    }
    palette_name = st.session_state.get("plot_color_palette", "deep")
    
    
    discrete_seq = qualitative_palettes.get(palette_name, px.colors.qualitative.Plotly)
    fig = px.line(
        explained_df,
        x="Componente",
        y="Proporción_varianza",
        markers=True,
        title="Gráfico de sedimentación (Scree plot)",
        color_discrete_sequence= discrete_seq
    )
    fig.update_layout(xaxis_title="Componente", yaxis_title="Proporción de varianza")
    st.plotly_chart(fig, use_container_width=True)


def _get_available_pcs() -> List[str]:
    scores = st.session_state.get("pca_scores")
    if scores is None:
        return []
    return list(scores.columns)


def render_scores_plots() -> None:
    """
    Mostrar gráficos de scores (PCx vs PCy) con opción de colorear por variable.
    Usa la paleta global almacenada en st.session_state["plot_color_palette"].
    """
    
    palette_name = st.session_state.get("plot_color_palette", "deep")
    discrete_colors = get_discrete_palette(palette_name)
    
    scores_df: pd.DataFrame | None = st.session_state.get("pca_scores")
    if scores_df is None or scores_df.empty:
        st.info("Aún no se han calculado scores de PCA.")
        return

    pc_cols = [c for c in scores_df.columns if c.startswith("PC")]
    if len(pc_cols) < 2:
        st.warning("Se requieren al menos dos componentes principales para este gráfico.")
        return

    st.subheader("Gráfico de scores de PCA")

    col1, col2, col3 = st.columns(3)

    with col1:
        x_pc = st.selectbox(
            "Componente eje X",
            pc_cols,
            index=0,
            key="scores_x_pc",
        )

    with col2:
        y_default = 1 if len(pc_cols) > 1 else 0
        y_pc = st.selectbox(
            "Componente eje Y",
            pc_cols,
            index=y_default,
            key="scores_y_pc",
        )

    # --- Opciones de color ---
    clean_df: pd.DataFrame | None = st.session_state.get("clean_df")
    color_options = ["Sin color"]
    valid_color_cols: list[str] = []

    if clean_df is not None:
        # Solo columnas con el mismo número de filas que scores_df
        for col in clean_df.columns:
            if len(clean_df[col]) == len(scores_df):
                valid_color_cols.append(col)
        color_options += valid_color_cols

    with col3:
        color_choice = st.selectbox(
            "Color por variable",
            color_options,
            key="scores_color_by",
        )

    # Construimos el data_frame final que usará Plotly
    plot_df = scores_df.copy()
    color_arg: str | None = None

    if color_choice != "Sin color" and color_choice in valid_color_cols and clean_df is not None:
        # Creamos una columna interna para el color
        color_arg = "color_var"
        plot_df[color_arg] = clean_df[color_choice].values

    # --- Paleta global para Plotly ---
    palette_name = st.session_state.get("plot_color_palette", "Plotly")

    qualitative_palettes = {
        "Plotly": px.colors.qualitative.Plotly,
        "D3": px.colors.qualitative.D3,
        "Pastel": px.colors.qualitative.Pastel,
        "Dark24": px.colors.qualitative.Dark24,
        "Set2": px.colors.qualitative.Set2,
    }
    discrete_seq = qualitative_palettes.get(palette_name, px.colors.qualitative.Plotly)

    fig = px.scatter(
        plot_df,
        x=x_pc,
        y=y_pc,
        color=color_arg,
        title=f"Scores de PCA: {x_pc} vs {y_pc}",
        color_discrete_sequence=discrete_seq,
    )
    fig.update_layout(template="plotly_dark", height=500)

    st.plotly_chart(fig, use_container_width=True)


def render_biplot() -> None:
    """Genera un biplot sencillo con scores y loadings."""

    palette_name = st.session_state.get("plot_color_palette", "deep")
    discrete_colors = get_discrete_palette(palette_name)
    
    scores_df = st.session_state.get("pca_scores")
    loadings_df = st.session_state.get("pca_loadings")
    if scores_df is None or loadings_df is None:
        st.info("Calcule el PCA para visualizar el biplot.")
        return

    pcs = _get_available_pcs()
    if len(pcs) < 2:
        st.warning("Se requieren al menos dos componentes para el biplot.")
        return

    col1, col2 = st.columns(2)
    discrete_colors = get_discrete_palette(palette_name)

    pc_x = col1.selectbox("Componente X (biplot)", pcs, index=0)
    pc_y = col2.selectbox("Componente Y (biplot)", pcs, index=1 if len(pcs) > 1 else 0)

    # Scatter de scores usa la paleta completa
    fig = px.scatter(
        scores_df,
        x=pc_x,
        y=pc_y,
        title="Biplot PCA",
        color_discrete_sequence=discrete_colors,
    )

    # Color para vectores
    vector_color = discrete_colors[0] if discrete_colors else "red"

    # Escalado de vectores
    loadings_subset = loadings_df[[pc_x, pc_y]]
    max_score = max(scores_df[pc_x].abs().max(), scores_df[pc_y].abs().max())
    scale = max_score * 0.8 if max_score > 0 else 1

    # Añadir vectores
    for var_name, (loading_x, loading_y) in loadings_subset.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, loading_x * scale],
                y=[0, loading_y * scale],
                mode="lines+text",
                text=["", var_name],
                textposition="top center",
                line=dict(color=vector_color, width=2),
                showlegend=False,
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    fig.update_layout(xaxis_title=pc_x, yaxis_title=pc_y)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    render_header()

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.warning("Primero aplique el preprocesamiento en la página anterior.")
        return

    st.session_state.setdefault("pca_config", {})
    st.session_state.setdefault("pca_model", None)
    st.session_state.setdefault("pca_scores", None)
    st.session_state.setdefault("pca_loadings", None)
    st.session_state.setdefault("pca_explained_variance", None)

    render_pca_config_panel(clean_df)
    render_run_pca_button(clean_df)
    render_explained_variance_section()
    render_scores_plots()
    render_biplot()


if __name__ == "__main__":
    main()
