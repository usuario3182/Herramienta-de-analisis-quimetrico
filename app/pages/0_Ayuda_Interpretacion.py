"""Página 0: Ayuda e interpretación de resultados.

Explica brevemente:
- El flujo general de la app.
- Cómo interpretar PCA.
- Cómo interpretar los clústers.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def render_header() -> None:
    st.set_page_config(page_title="Ayuda e interpretación", page_icon="📖")
    st.title("📖 Ayuda e interpretación de resultados")

    st.markdown(
        """
        Esta página reúne **explicaciones breves** sobre qué hace cada módulo de la
        aplicación y cómo interpretar las salidas principales (tablas, gráficos y
        métricas).

        Puedes usarla como referencia rápida mientras trabajas en las otras páginas.
        """
    )


def render_flow_overview() -> None:
    st.header("1. Flujo general de la aplicación")

    st.markdown(
        """
        1. **Inicio / Carga de datos**  
           - Carga un archivo CSV/Excel o utiliza el dataset de ejemplo.  
           - El dataset cargado se guarda como `raw_df` en el estado de sesión.

        2. **Preprocesamiento**  
           - Selección de variables a utilizar.  
           - Definición de tipo de cada variable (numérica, categórica, fecha).  
           - Imputación de valores faltantes (media, mediana, moda, eliminar filas, etc.).  
           - Opcionalmente, **escalado** (estandarizar o Min–Max).  
           - El resultado se guarda como `clean_df`.

        3. **PCA**  
           - Selección de variables numéricas para PCA.  
           - Elección del número de componentes.  
           - Se generan:
                - `pca_scores`: coordenadas de cada muestra en el espacio de PCs.  
                - `pca_loadings`: contribución de cada variable a cada componente.  
                - `pca_explained_variance`: tabla de varianza explicada.

        4. **Clustering**  
           - Aplicación de K-Means o clustering jerárquico sobre `clean_df` o `pca_scores`.  
           - Se obtienen:
                - `cluster_labels`: etiqueta de clúster por muestra.  
                - `cluster_metrics`: métricas globales (ej. silhouette, inertia).

        5. **Resultados y exportación**  
           - Descarga de datos originales, preprocesados, scores, loadings, varianza,  
             etiquetas de clúster y métricas.  
           - Descarga de figuras clave (scree plot, scores PC1 vs PC2, etc.).
        """
    )


def render_pca_help() -> None:
    st.header("2. Interpretación de PCA")

    explained_df: Optional[pd.DataFrame] = st.session_state.get("pca_explained_variance")
    scores_df: Optional[pd.DataFrame] = st.session_state.get("pca_scores")
    loadings_df: Optional[pd.DataFrame] = st.session_state.get("pca_loadings")

    st.subheader("2.1 Varianza explicada")

    st.markdown(
        """
        - Cada componente principal (PC1, PC2, ...) explica un porcentaje de la
          varianza total de los datos.  
        - El **scree plot** muestra cómo se reparte la varianza entre componentes.  
        - Regla práctica:
            - Elige un número de componentes tal que la **varianza acumulada**
              supere un umbral razonable (por ejemplo, 70–90 %).
        """
    )

    if explained_df is not None:
        st.markdown("Vista rápida de la tabla de varianza explicada (si existe en memoria):")
        st.dataframe(explained_df.head(), use_container_width=True)

    st.subheader("2.2 Scores de PCA")

    st.markdown(
        """
        - Los **scores** son las coordenadas de cada muestra en el espacio de PCs.  
        - Los gráficos `PC1 vs PC2`, `PC1 vs PC3`, etc. permiten:
            - Detectar grupos naturales de muestras.
            - Identificar outliers (muestras alejadas del resto).
        - Si coloreas por una variable categórica o por clúster, puedes ver si
          las agrupaciones tienen sentido químico o experimental.
        """
    )

    if scores_df is not None:
        st.caption("Dimensión actual de los scores de PCA:")
        st.write(f"{scores_df.shape[0]} muestras × {scores_df.shape[1]} componentes")

    st.subheader("2.3 Loadings (contribución de variables)")

    st.markdown(
        """
        - Los **loadings** indican cuánto contribuye cada variable original a cada
          componente principal.  
        - Una carga (loading) alta en PC1 significa que esa variable tiene mucho
          peso en esa dirección de variación.  
        - En un biplot:
            - Las flechas largas indican variables importantes.
            - Ángulos pequeños entre flechas ≈ variables correlacionadas.
        """
    )

    if loadings_df is not None:
        st.caption("Ejemplo de tabla de loadings:")
        st.dataframe(loadings_df.head(), use_container_width=True)


def render_clustering_help() -> None:
    st.header("3. Interpretación de clustering")

    cluster_labels = st.session_state.get("cluster_labels")
    cluster_metrics = st.session_state.get("cluster_metrics")

    st.markdown(
        """
        El objetivo del clustering es **agrupar muestras similares** según sus
        características (ya sea en el espacio original o en el de las PCs).

        - **K-Means**:
            - Minimiza la distancia de cada punto a su centroide de clúster.
            - Necesita que especifiques `k` (número de clústers).

        - **Clustering jerárquico**:
            - Construye un árbol de fusiones (dendrograma).
            - Permite explorar distintos números de clústers cortando el dendrograma.
        """
    )

    st.subheader("3.1 Métrica silhouette")

    st.markdown(
        """
        - La **silhouette** mide qué tan bien separado está cada clúster.  
        - Valores típicos:
            - Cerca de 1 → clústers muy compactos y bien separados.
            - Cerca de 0 → clústers solapados.
            - Negativos → algunas muestras podrían estar en el clúster equivocado.
        """
    )

    if cluster_metrics and "silhouette" in cluster_metrics:
        st.write(f"Silhouette actual (si existe): **{cluster_metrics['silhouette']:.3f}**")

    st.subheader("3.2 Inercia (solo K-Means)")

    st.markdown(
        """
        - La **inercia** es la suma de distancias cuadráticas de las muestras a su
          centroide de clúster.  
        - Se utiliza para el método del **codo**:
            - Calculas la inercia para varios valores de `k`.
            - Buscas un punto donde la mejora comienza a ser marginal (el “codo”).
        """
    )

    if cluster_metrics and "inertia" in cluster_metrics:
        st.write(f"Inercia actual (si existe): **{cluster_metrics['inertia']:.3g}**")

    if cluster_labels is not None:
        st.caption(
            f"Número de muestras con etiqueta de clúster: {len(cluster_labels)} "
            f"(guardadas en `cluster_labels`)."
        )


def render_best_practices() -> None:
    st.header("4. Buenas prácticas y recomendaciones")

    st.markdown(
        """
        - Antes de aplicar PCA o clustering:
            - Revisa **distribuciones**, **outliers** y **valores faltantes**.
            - Asegúrate de que las variables numéricas estén en escalas comparables
              (escalado o estandarización).

        - Después de PCA:
            - No te quedes solo con el porcentaje de varianza.  
              Analiza qué variables explican cada componente (loadings).
            - Verifica si los patrones encontrados tienen sentido químico /
              experimental.

        - Después de clustering:
            - Revisa la silhouette y la inercia, pero también la **interpretación
              química** de los clústers.
            - Usa los resúmenes por clúster (medias, centroides en PCs) para
              describir cada grupo: "Clúster 1: alto en C18, bajo en C12…", etc.

        - Documenta siempre:
            - Qué preprocesamiento aplicaste (imputación, escalado).
            - Cuántos componentes usaste en PCA y por qué.
            - Qué algoritmo de clustering usaste y con qué parámetros.
        """
    )


def main() -> None:
    render_header()
    render_flow_overview()
    st.markdown("---")
    render_pca_help()
    st.markdown("---")
    render_clustering_help()
    st.markdown("---")
    render_best_practices()


if __name__ == "__main__":
    main()
