"""
clustering_utils.py

Funciones auxiliares para aplicar algoritmos de clustering dentro de la
aplicación de análisis quimiométrico.

RESPONSABILIDAD DEL MÓDULO
--------------------------
- Preparar la matriz de datos a partir de un DataFrame (selección de columnas).
- Ejecutar algoritmos de clustering (K-Means y jerárquico/agglomerative).
- Calcular métricas simples de calidad de clúster (por ejemplo, silhouette).
- Devolver:
    * El modelo entrenado.
    * Las etiquetas de clúster (labels) como array/Series.
    * Información adicional útil (centroides, linkage, métricas).

SE USA PRINCIPALMENTE EN:
-------------------------
- app/pages/4_Clustering.py

CONVENCIONES:
-------------
- Entrada principal: DataFrame limpio (clean_df) o scores de PCA
  (pca_scores), según defina la página 4.
- Por simplicidad, trabajaremos solo con columnas numéricas.
- Los nombres de las funciones deben ser estables para que la página 4
  pueda importarlas sin romperse.

IMPORTANTE PARA CODEX:
----------------------
- NO agregar código de Streamlit aquí.
- Mantener mensajes de error en español (para mostrarlos tal cual en la UI).
- SOLO completar las funciones marcadas con TODO. El resto del archivo
  no debe modificarse salvo que sea estrictamente necesario para compilar.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

 
# === TODO: implementar funciones auxiliares de clustering debajo de esta línea ===


def select_numeric_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Seleccionar columnas numéricas a utilizar para clustering.

    - Si columns es None o lista vacía, se toman todas las columnas numéricas.
    - Si columns se proporciona, se valida que existan y se filtran solo las
      columnas numéricas.
    - Si no queda ninguna columna numérica, se lanza ValueError.

    TODO: implementar cuerpo de la función.
    """
    raise NotImplementedError


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int,
    columns: List[str] | None = None,
    random_state: int = 0,
) -> Tuple[KMeans, np.ndarray, Dict[str, float]]:
    """
    Ejecutar K-Means sobre las columnas seleccionadas.

    Devuelve:
    - modelo KMeans entrenado.
    - labels: arreglo de etiquetas de clúster (shape = [n_muestras]).
    - metrics: diccionario con métricas (por ejemplo:
          {
              "silhouette": ...,
              "inertia": ...
          }
      )

    TODO: implementar cuerpo de la función.
    """
    raise NotImplementedError


def run_hierarchical(
    df: pd.DataFrame,
    n_clusters: int,
    columns: List[str] | None = None,
    linkage: str = "ward",
    affinity: str = "euclidean",
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, float]]:
    """
    Ejecutar clustering jerárquico/agglomerative.

    Devuelve:
    - modelo AgglomerativeClustering entrenado.
    - labels: arreglo de etiquetas de clúster (shape = [n_muestras]).
    - metrics: diccionario con métricas (por ejemplo:
          {
              "silhouette": ...
          }
      )

    NOTA: Para linkage="ward" sklearn ignora "affinity" y usa euclidean.

    TODO: implementar cuerpo de la función.
    """
    raise NotImplementedError

