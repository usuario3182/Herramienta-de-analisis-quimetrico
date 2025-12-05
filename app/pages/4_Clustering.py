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
    pass


def get_available_sources() -> Dict[str, pd.DataFrame]:
    """
    Devolver un diccionario con las fuentes de datos disponibles para clustering.

    Claves sugeridas:
        - "clean_df": st.session_state["clean_df"]
        - "pca_scores": st.session_state["pca_scores"]

    Solo deben incluirse las fuentes que existan en session_state.

    TODO: implementar esta función.
    """
    pass


def render_clustering_config_panel(sources: Dict[str, pd.DataFrame]) -> None:
    """
    Panel lateral/central para configurar el clustering.

    Flujo sugerido:
    - Selectbox para elegir la fuente de datos ("clean_df" vs "pca_scores").
    - A partir de la fuente elegida, obtener columnas numéricas disponibles.
    - Multiselect de columnas a utilizar.
    - Selectbox para algoritmo ("K-Means", "Clustering jerárquico").
    - Number_input o slider para número de clústeres (k).
    - Si el algoritmo es jerárquico:
        * Selectbox para linkage (ward, complete, average, single).
        * Selectbox para affinity/distancia (euclidean, manhattan, etc.),
          respetando las restricciones de sklearn (ward solo con euclidean).
    - Guardar la configuración en st.session_state["cluster_config"], por ejemplo:
        {
            "source": "clean_df" o "pca_scores",
            "columns": [...],
            "algorithm": "kmeans" o "hierarchical",
            "n_clusters": k,
            "linkage": "...",
            "affinity": "..."
        }

    TODO: implementar esta función.
    """
    pass


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
    pass


def render_cluster_plots(sources: Dict[str, pd.DataFrame]) -> None:
    """
    Mostrar gráficos de resultados de clustering.

    Ideas:
    - Usar la misma fuente de datos que se usó para el clustering.
    - Si la fuente es "pca_scores" y existen columnas PC1 y PC2:
        * Hacer scatter de PC1 vs PC2 con color por etiqueta de clúster.
    - Si no hay PCA, escoger dos columnas numéricas de la fuente
      (con selectboxes) para ejes X e Y y colorear por clúster.
    - Usar plotly.express.scatter para los gráficos.

    También puede mostrarse una tabla con las métricas de clúster.

    TODO: implementar esta función.
    """
    pass


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


if __name__ == "__main__":
    main()
