"""Página de PCA para configurar, calcular y visualizar resultados."""
from __future__ import annotations

from typing import List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.pca_utils import run_pca


def render_header() -> None:
    """Encabezado explicativo de la página."""

    st.title("📊 Análisis de Componentes Principales")
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

    effective_columns = selected_columns or numeric_columns
    max_components = min(len(effective_columns), 10)
    current_n = previous_config.get("n_components", max_components)
    n_components = st.slider(
        "Número de componentes",
        min_value=1,
        max_value=max_components,
        value=current_n if 1 <= current_n <= max_components else max_components,
        step=1,
    )

    st.session_state["pca_config"] = {
        "columns": effective_columns,
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

    fig = px.line(
        explained_df,
        x="Componente",
        y="Proporción_varianza",
        markers=True,
        title="Gráfico de sedimentación (Scree plot)",
    )
    fig.update_layout(xaxis_title="Componente", yaxis_title="Proporción de varianza")
    st.plotly_chart(fig, use_container_width=True)


def _get_available_pcs() -> List[str]:
    scores = st.session_state.get("pca_scores")
    if scores is None:
        return []
    return list(scores.columns)


def render_scores_plots() -> None:
    """Permite explorar los scores del PCA en 2D."""

    scores_df = st.session_state.get("pca_scores")
    clean_df = st.session_state.get("clean_df")
    if scores_df is None:
        st.info("Calcule el PCA para visualizar los scores.")
        return

    pcs = _get_available_pcs()
    if len(pcs) < 2:
        st.warning("Se necesitan al menos dos componentes para el gráfico de scores.")
        return

    col1, col2 = st.columns(2)
    pc_x = col1.selectbox("Componente en eje X", pcs, index=0)
    pc_y = col2.selectbox("Componente en eje Y", pcs, index=1 if len(pcs) > 1 else 0)

    color_options = ["Sin color"]
    if clean_df is not None:
        color_options.extend(list(clean_df.columns))

    color_by = st.selectbox("Color por variable", options=color_options)
    color_arg = None if color_by == "Sin color" else color_by

    fig = px.scatter(
        scores_df,
        x=pc_x,
        y=pc_y,
        color=color_arg,
        title="Scores PCA",
        labels={pc_x: pc_x, pc_y: pc_y},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_biplot() -> None:
    """Genera un biplot sencillo con scores y loadings."""

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
    pc_x = col1.selectbox("Componente X (biplot)", pcs, index=0)
    pc_y = col2.selectbox("Componente Y (biplot)", pcs, index=1 if len(pcs) > 1 else 0)

    fig = px.scatter(scores_df, x=pc_x, y=pc_y, title="Biplot PCA")

    loadings_subset = loadings_df[[pc_x, pc_y]]
    max_score = max(scores_df[pc_x].abs().max(), scores_df[pc_y].abs().max())
    scale = max_score * 0.6 if max_score > 0 else 1

    for var_name, (loading_x, loading_y) in loadings_subset.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, loading_x * scale],
                y=[0, loading_y * scale],
                mode="lines+text",
                text=["", var_name],
                textposition="top center",
                line=dict(color="red"),
                showlegend=False,
            )
        )

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
