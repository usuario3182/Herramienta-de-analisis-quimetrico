"""
Página 4: Clustering.
 
OBJETIVO DE LA PÁGINA
---------------------
- Aplicar métodos de clustering (K-Means y jerárquico) sobre:
    * el DataFrame preprocesado (clean_df), o
    * los scores de PCA (pca_scores), según configuración.
- Permitir al usuario:
    1) Elegir la fuente de datos para el clustering:
         - "Variables preprocesadas"  (clean_df)
         - "Scores de PCA"            (pca_scores)
    2) Seleccionar las columnas numéricas a utilizar.
    3) Elegir el algoritmo y sus parámetros:
         - K-Means: número de clústeres (k).
         - Jerárquico: número de clústeres, tipo de linkage/distancia.
    4) Ejecutar el clustering y guardar labels en st.session_state.
    5) Visualizar:
         - Métricas básicas (por ejemplo, silhouette).
         - Gráfico de dispersión coloreado por clúster:
             * Si hay PCA: PC1 vs PC2 con color = clúster.
             * Si no hay PCA, alguna combinación de variables seleccionadas.
         - Dendrograma aproximado para el clustering jerárquico
           (opcional o a través de un helper sencillo).

ESTADO COMPARTIDO UTILIZADO
---------------------------
Lectura de:
- clean_df                 (creado en Página 2)
- pca_scores               (creado en Página 3, opcional)
Escritura de:
- cluster_model            (modelo entrenado, KMeans o AgglomerativeClustering)
- cluster_labels           (Series/array con etiquetas por muestra)
- cluster_metrics          (dict con métricas de evaluación del clúster)
- cluster_config           (configuración elegida por la persona usuaria)

FUNCIONES AUXILIARES EXTERNAS
-----------------------------
from scripts.clustering_utils import (
    select_numeric_features,
    run_kmeans,
    run_hierarchical,
)

REQUISITOS DE INTERFAZ
----------------------
1) Encabezado:
    - Explicar brevemente qué es clustering y qué aporta en el contexto
      quimiométrico.
2) Panel de configuración:
    - Selector de fuente de datos: "clean_df" vs "pca_scores".
    - Multiselect de columnas numéricas (según la fuente).
    - Selector de algoritmo:
         - "K-Means"
         - "Clustering jerárquico"
    - Parámetros:
         - Número de clústeres (k).
         - Linkage/affinity para jerárquico.
    - Botón "Ejecutar clustering".
3) Zona de resultados:
    - Mostrar métricas.
    - Gráfico de dispersión coloreado por clúster.
    - Opcional: dendrograma simplificado para jerárquico.

IMPORTANTE PARA CODEX
---------------------
- NO modificar las páginas 1_Home.py, 2_Preprocesamiento.py, 3_PCA.py
  ni otros módulos que ya están funcionando.
- Solo escribir/editar código dentro de las funciones marcadas con TODO
  en este archivo.
- Todo el texto visible en la UI debe estar en español.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.io_utils import get_discrete_palette

palette_name = st.session_state.get("plot_color_palette", "deep")
discrete_colors = get_discrete_palette(palette_name)

# Asegurar que scripts/ esté en el path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from scripts.clustering_utils import (
    select_numeric_features,
    run_hierarchical,
    run_kmeans,
)


# === TODO: implementar funciones de UI para clustering debajo de esta línea ===


def render_header() -> None:
    """Mostrar título y explicación general de clustering.

    TODO: implementar contenido de encabezado (st.title, st.markdown) en español.
    """
    st.title("Clustering de muestras")
    st.markdown(
        "Agrupa observaciones similares para revelar patrones o segmentos en los datos. "
        "Aquí puedes usar las variables preprocesadas o los scores de PCA para formar "
        "clústeres y evaluar su calidad con métricas básicas."
    )
def render_cluster_projection_and_summary() -> None:
    """
    Mostrar:
    - Dispersión de clústers en el espacio de PCs (PC1 vs PC2, etc.).
    - Tabla de centroides en PC-space.
    - Tabla de medias por clúster en el espacio original (clean_df).
    """

    pca_scores: pd.DataFrame | None = st.session_state.get("pca_scores")
    clean_df: pd.DataFrame | None = st.session_state.get("clean_df")
    cluster_labels = st.session_state.get("cluster_labels")

    if pca_scores is None or cluster_labels is None:
        st.info(
            "Para ver la proyección de clústers en el espacio de PCs, "
            "primero calcule el PCA y luego ejecute un método de clustering."
        )
        return

    from scripts.io_utils import get_discrete_palette
    palette_name = st.session_state.get("plot_color_palette", "deep")
    discrete_colors = get_discrete_palette(palette_name)

    labels_series = pd.Series(np.asarray(cluster_labels), index=pca_scores.index, name="cluster")
    scores_with_cluster = pca_scores.copy()
    scores_with_cluster["cluster"] = labels_series.astype(str)

    pc_cols = [c for c in scores_with_cluster.columns if c.startswith("PC")]
    if len(pc_cols) < 2:
        st.warning("Se requieren al menos dos componentes principales para la proyección de clústers.")
        return

    st.subheader("Proyección de clústers en el espacio de componentes principales")

    # Selección de PCs para la proyección
    col1, col2 = st.columns(2)
    pc_x = col1.selectbox("Componente X (resumen)", pc_cols, index=0)
    pc_y = col2.selectbox(
        "Componente Y (resumen)",
        pc_cols,
        index=1 if len(pc_cols) > 1 else 0,
    )

    # Centroides en espacio de PCs
    centroids_pcs = scores_with_cluster.groupby("cluster")[pc_cols].mean()
    centroids_pcs["n_muestras"] = scores_with_cluster.groupby("cluster").size()

    st.markdown("**Centroides en el espacio de componentes principales**")
    st.dataframe(centroids_pcs.style.format(precision=4), use_container_width=True)

    # Scatter de PCs con centroides
    import plotly.graph_objects as go

    fig = px.scatter(
        scores_with_cluster,
        x=pc_x,
        y=pc_y,
        color="cluster",
        title=f"Clústeres y centroides en PCA ({pc_x} vs {pc_y})",
        color_discrete_sequence=discrete_colors,
        height=700
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))

    # Añadir centroides como marcadores grandes
    for cluster_id, row in centroids_pcs.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[pc_x]],
                y=[row[pc_y]],
                mode="markers+text",
                marker=dict(size=14, symbol="x"),
                text=[f"C{cluster_id}"],
                textposition="top center",
                name=f"Centroide {cluster_id}",
                showlegend=False,
            )
        )

    st.plotly_chart(fig, use_container_width=True)

    # Medias por clúster en el espacio original
    if clean_df is not None:
        st.markdown("**Medias de variables originales por clúster**")

        df_original = clean_df.copy()
        # alineamos índices por seguridad
        df_original = df_original.loc[scores_with_cluster.index]
        df_original["cluster"] = labels_series

        means_original = (
            df_original
            .groupby("cluster")
            .mean(numeric_only=True)
            .sort_index()
        )
        means_original["n_muestras"] = df_original.groupby("cluster").size()

        st.dataframe(means_original.style.format(precision=4), use_container_width=True)
    else:
        st.info(
            "No se encontraron datos preprocesados (`clean_df`) para calcular "
            "las medias de variables originales por clúster."
        )



def get_available_sources() -> Dict[str, pd.DataFrame]:
    """
    Devolver un diccionario con las fuentes de datos disponibles para clustering.

    Claves sugeridas:
        - "clean_df": st.session_state["clean_df"]
        - "pca_scores": st.session_state["pca_scores"]

    Solo deben incluirse las fuentes que existan en session_state.

    TODO: implementar esta función.
    """
    sources: Dict[str, pd.DataFrame] = {}
    if "clean_df" in st.session_state and st.session_state["clean_df"] is not None:
        sources["clean_df"] = st.session_state["clean_df"]
    if "pca_scores" in st.session_state and st.session_state["pca_scores"] is not None:
        sources["pca_scores"] = st.session_state["pca_scores"]
    return sources


def render_clustering_config_panel(sources: Dict[str, pd.DataFrame]) -> None:
    """
    Panel para configurar el clustering.

    Ahora prioriza:
    - Primero: clustering en espacio de componentes principales (pca_scores).
    - Luego: clustering sobre todas las variables preprocesadas (clean_df).
    """

    # Etiquetas más claras para la UI
    source_labels = {
        "pca_scores": "Componentes principales (PCA)",
        "clean_df": "Variables originales preprocesadas",
    }

    # Ordenar fuentes: primero pca_scores si existe, luego clean_df
    source_keys: list[str] = []
    if "pca_scores" in sources:
        source_keys.append("pca_scores")
    if "clean_df" in sources:
        source_keys.append("clean_df")

    # Si hay otras fuentes en el dict, agrégalas al final
    for key in sources.keys():
        if key not in source_keys:
            source_keys.append(key)

    if not source_keys:
        st.error("No hay fuentes de datos disponibles para clustering.")
        return

    # Por defecto usar PCA si está disponible, si no, la primera fuente
    if "pca_scores" in source_keys:
        default_index = source_keys.index("pca_scores")
    else:
        default_index = 0

    display_labels = [source_labels.get(key, key) for key in source_keys]

    st.subheader("Configuración de clustering")

    st.caption(
        """
        Primero se recomienda hacer el **clustering en el espacio de componentes principales (PCA)**, 
        y opcionalmente comparar con el clustering usando **todas las variables preprocesadas**.
        """
    )

    selected_index = st.selectbox(
        "Fuente de datos para clustering",
        options=range(len(source_keys)),
        index=default_index,
        format_func=lambda i: display_labels[i],
    )
    selected_source = source_keys[selected_index]

    df_source = sources[selected_source]

    # Selección de columnas numéricas
    # Si la fuente es pca_scores, normalmente las columnas son PC1, PC2, ...
    if selected_source == "pca_scores":
        numeric_cols = [c for c in df_source.columns if c.startswith("PC")]
        if not numeric_cols:  # fallback por si no siguen el patrón
            numeric_cols = df_source.select_dtypes(include="number").columns.tolist()
    else:
        numeric_cols = df_source.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        selected_columns = st.multiselect(
            "Columnas numéricas a utilizar para el clustering",
            options=numeric_cols,
            default=numeric_cols,
        )
    else:
        st.warning("No hay columnas numéricas disponibles en la fuente seleccionada.")
        selected_columns = []

    # Selección de algoritmo
    algo_label = st.selectbox("Algoritmo de clustering", ["K-Means", "Clustering jerárquico"])
    algorithm = "kmeans" if algo_label == "K-Means" else "hierarchical"

    # Número de clústeres
    n_clusters = st.slider("Número de clústeres (k)", min_value=2, max_value=10, value=3)

    # Parámetros extra para clustering jerárquico
    linkage = "ward"
    affinity = "euclidean"
    if algorithm == "hierarchical":
        linkage = st.selectbox("Método de enlace (linkage)", ["ward", "complete", "average", "single"])
        affinity_options = ["euclidean", "manhattan", "cosine"]
        affinity = st.selectbox("Distancia / affinity", affinity_options, index=0)
        if linkage == "ward":
            affinity = "euclidean"
            st.caption("Con linkage 'ward' se fuerza la distancia euclidean (requisito de sklearn).")

    # Guardar configuración en session_state
    st.session_state["cluster_config"] = {
        "source": selected_source,         # "pca_scores" o "clean_df"
        "columns": selected_columns,       # lista de columnas usadas
        "algorithm": algorithm,            # "kmeans" o "hierarchical"
        "n_clusters": n_clusters,
        "linkage": linkage,
        "affinity": affinity,
    }



def render_run_clustering_button(sources: Dict[str, pd.DataFrame]) -> None:
    """
    Botón para ejecutar el clustering según la configuración actual.

    Flujo sugerido:
    - Leer config = st.session_state.get("cluster_config", {}).
    - Validar que:
        * exista una fuente seleccionada,
        * haya al menos una columna,
        * n_clusters sea >= 2.
    - Obtener el DataFrame de la fuente (clean_df o pca_scores).
    - Llamar a:
        - run_kmeans(...) o
        - run_hierarchical(...)
      según config["algorithm"].
    - Guardar en session_state:
        - "cluster_model"
        - "cluster_labels"
        - "cluster_metrics"
    - Mostrar st.success si todo sale bien.
    - Capturar ValueError u otras excepciones y mostrarlas con st.error().

    TODO: implementar esta función.
    """
    if st.button("Ejecutar clustering", type="primary"):
        config = st.session_state.get("cluster_config", {})
        try:
            if "source" not in config or config["source"] not in sources:
                raise ValueError("Seleccione una fuente de datos válida.")
            if not config.get("columns"):
                raise ValueError("Debe seleccionar al menos una columna para clustering.")

            n_clusters = config.get("n_clusters")
            if n_clusters is None or n_clusters < 2:
                raise ValueError("El número de clústeres debe ser al menos 2.")

            df_source = sources[config["source"]]
            if config.get("algorithm") == "hierarchical":
                model, labels, metrics = run_hierarchical(
                    df_source,
                    n_clusters,
                    config.get("columns"),
                    linkage=config.get("linkage", "ward"),
                    affinity=config.get("affinity", "euclidean"),
                )
            else:
                model, labels, metrics = run_kmeans(
                    df_source,
                    n_clusters,
                    config.get("columns"),
                    random_state=0,
                )

            st.session_state["cluster_model"] = model
            st.session_state["cluster_labels"] = labels
            st.session_state["cluster_metrics"] = metrics

            st.success("Clustering ejecutado correctamente.")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error al ejecutar el clustering: {e}")


def render_cluster_plots(sources: Dict[str, pd.DataFrame]) -> None:
    """
    Mostrar gráficos de resultados de clustering.

    - Usa la misma fuente de datos que se usó para el clustering.
    - Si la fuente es "pca_scores" y existen columnas PC1, PC2, ...:
        * Hace scatter en el espacio de PCs con color por etiqueta de clúster.
    - Si la fuente es otra (clean_df, etc.), usa variables numéricas
      seleccionables como ejes X e Y y colorea por clúster.
    """

    labels = st.session_state.get("cluster_labels")
    metrics = st.session_state.get("cluster_metrics")
    config = st.session_state.get("cluster_config", {})

    if labels is None:
        st.info("Ejecute el clustering para ver los resultados y las gráficas.")
        return

    source_key = config.get("source")
    if not source_key or source_key not in sources:
        st.warning("No se encontró la fuente de datos utilizada para el clustering.")
        return

    df_source = sources[source_key].copy()

    # Tomamos la paleta actual desde session_state
    from scripts.io_utils import get_discrete_palette  # import local para evitar problemas de orden
    palette_name = st.session_state.get("plot_color_palette", "deep")
    discrete_colors = get_discrete_palette(palette_name)

    # Aseguramos que las etiquetas sean una serie alineada con el índice
    labels_series = pd.Series(np.asarray(labels), index=df_source.index, name="cluster")
    df_source["cluster"] = labels_series.astype(str)

    st.subheader("Visualización de clústeres")

    # ------------------------------------------------------------------
    # Caso 1: fuente = pca_scores → espacio de componentes principales
    # ------------------------------------------------------------------
    if source_key == "pca_scores":
        # Buscar columnas que empiecen con "PC"
        pc_cols = [c for c in df_source.columns if c.startswith("PC")]
        if len(pc_cols) < 2:
            st.warning(
                "Se requieren al menos dos componentes principales (PCs) para graficar. "
                "Verifique la configuración de PCA."
            )
        else:
            col1, col2 = st.columns(2)
            pc_x = col1.selectbox("Componente X (PCA)", pc_cols, index=0)
            pc_y = col2.selectbox(
                "Componente Y (PCA)",
                pc_cols,
                index=1 if len(pc_cols) > 1 else 0,
            )

            fig = px.scatter(
                df_source,
                x=pc_x,
                y=pc_y,
                color="cluster",
                title=f"Clústeres proyectados en espacio PCA ({pc_x} vs {pc_y})",
                color_discrete_sequence=discrete_colors,
                height=700
            )
            fig.update_traces(marker=dict(size=8, opacity=0.85))
            fig.update_layout(legend_title="Clúster")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Caso 2: fuente = clean_df (u otra) → espacio de variables originales
    # ------------------------------------------------------------------
    else:
        numeric_cols = df_source.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Se necesitan al menos dos columnas numéricas para graficar.")
        else:
            col1, col2 = st.columns(2)
            x_axis = col1.selectbox("Eje X", options=numeric_cols, index=0)
            y_default = 1 if len(numeric_cols) > 1 else 0
            y_axis = col2.selectbox("Eje Y", options=numeric_cols, index=y_default)

            fig = px.scatter(
                df_source,
                x=x_axis,
                y=y_axis,
                color="cluster",
                title="Distribución de clústeres en variables originales",
                color_discrete_sequence=discrete_colors,
            )
            fig.update_traces(marker=dict(size=8, opacity=0.85))
            fig.update_layout(legend_title="Clúster")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Métricas del clustering
    # ------------------------------------------------------------------
    if metrics:
        st.subheader("Métricas del clustering")
        metric_items = list(metrics.items())
        cols = st.columns(len(metric_items))
        for (name, value), col in zip(metric_items, cols):
            display_value = "N/A" if value is None else f"{float(value):.4f}"
            col.metric(name, display_value)


def main() -> None:
    """Entrypoint de la página Clustering."""

    # Verificar que, al menos, exista clean_df
    if "clean_df" not in st.session_state:
        st.error(
            "No hay datos preprocesados disponibles. "
            "Por favor, vaya a la página 'Preprocesamiento' y aplique el pipeline primero."
        )
        return

    # Inicializar contenedores de estado
    st.session_state.setdefault("cluster_config", {})
    st.session_state.setdefault("cluster_model", None)
    st.session_state.setdefault("cluster_labels", None)
    st.session_state.setdefault("cluster_metrics", None)

    # Flujo de la página
    render_header()
    sources = get_available_sources()
    if not sources:
        st.warning(
            "No hay fuentes de datos disponibles para clustering. "
            "Asegúrese de haber preprocesado los datos y/o calculado el PCA."
        )
        return
    
    
    render_clustering_config_panel(sources)
    render_run_clustering_button(sources)
    render_cluster_plots(sources)
    render_cluster_projection_and_summary()


if __name__ == "__main__":
    main()
