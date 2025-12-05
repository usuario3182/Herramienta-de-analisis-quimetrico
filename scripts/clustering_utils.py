"""
clustering_utils.py

Funciones auxiliares para aplicar algoritmos de clustering dentro de la
aplicación de análisis quimiométrico.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


# -------------------------------------------------------------------
# Helpers internos (no exportados) para evitar duplicar lógica
# -------------------------------------------------------------------


def _select_numeric_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Seleccionar columnas numéricas a utilizar para clustering.

    - Si columns es None o lista vacía, se toman todas las columnas numéricas.
    - Si columns se proporciona, se valida que existan y se filtran solo las
      columnas numéricas.
    - Si no queda ninguna columna numérica, se lanza ValueError.
    """
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"Las siguientes columnas no existen en el DataFrame: {', '.join(missing)}"
            )
        numeric_df = df[columns].select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError(
                "No hay columnas numéricas para clustering dentro de la selección."
            )
        return numeric_df

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No hay columnas numéricas para clustering.")
    return numeric_df


def _validate_n_clusters(n_clusters: int, n_samples: int) -> None:
    """Validar rango de número de clústeres."""
    if n_samples < 2:
        raise ValueError("Se requieren al menos 2 muestras para aplicar clustering.")
    if n_clusters < 2 or n_clusters > n_samples:
        raise ValueError(
            "El número de clústeres debe estar entre 2 y el número de muestras."
        )


def _compute_silhouette_safe(X: pd.DataFrame, labels: np.ndarray) -> float | None:
    """
    Calcular silhouette de forma segura.

    Devuelve:
    - float si el cálculo es posible.
    - None si no es posible (por ejemplo, solo un clúster o una sola muestra).
    """
    # silhouette_score requiere al menos 2 muestras y al menos 2 clústeres distintos
    if len(X) < 2:
        return None
    if len(set(labels)) < 2:
        return None
    return float(silhouette_score(X, labels))


# -------------------------------------------------------------------
# API pública usada por la página de clustering
# -------------------------------------------------------------------


def select_numeric_features(
    df: pd.DataFrame,
    columns: List[str] | None = None,
) -> pd.DataFrame:
    """
    Versión pública del selector de columnas numéricas.

    Se expone por si la página de clustering necesita reutilizar esta lógica.
    """
    return _select_numeric_features(df, columns)


def run_kmeans(
    df: pd.DataFrame,
    n_clusters: int,
    columns: List[str] | None = None,
    random_state: int = 0,
) -> Tuple[KMeans, np.ndarray, Dict[str, float | None]]:
    """
    Ejecutar K-Means sobre las columnas seleccionadas.

    Devuelve:
    - modelo KMeans entrenado.
    - labels: arreglo de etiquetas de clúster (shape = [n_muestras]).
    - metrics: diccionario con métricas, por ejemplo:
          {
              "silhouette": ...,
              "inertia": ...
          }
    """
    X = _select_numeric_features(df, columns)

    _validate_n_clusters(n_clusters, len(X))

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    model.fit(X)

    labels: np.ndarray = model.labels_
    inertia: float = float(model.inertia_)
    silhouette = _compute_silhouette_safe(X, labels)

    metrics: Dict[str, float | None] = {
        "silhouette": silhouette,
        "inertia": inertia,
    }
    return model, labels, metrics


def run_hierarchical(
    df: pd.DataFrame,
    n_clusters: int,
    columns: List[str] | None = None,
    linkage: Literal["ward", "complete", "average", "single"] = "ward",
    affinity: str = "euclidean",
) -> Tuple[AgglomerativeClustering, np.ndarray, Dict[str, float | None]]:
    """
    Ejecutar clustering jerárquico/agglomerative.

    Devuelve:
    - modelo AgglomerativeClustering entrenado.
    - labels: arreglo de etiquetas de clúster (shape = [n_muestras]).
    - metrics: diccionario con métricas, por ejemplo:
          {
              "silhouette": ...
          }

    NOTA: Para linkage="ward" sklearn ignora "affinity" y usa distancia euclidiana.
    """
    X = _select_numeric_features(df, columns)

    _validate_n_clusters(n_clusters, len(X))

    # En versiones recientes de sklearn, el parámetro se llama `metric`.
    effective_metric = "euclidean" if linkage == "ward" else affinity

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=effective_metric,
    )
    model.fit(X)

    labels: np.ndarray = model.labels_
    silhouette = _compute_silhouette_safe(X, labels)

    metrics: Dict[str, float | None] = {
        "silhouette": silhouette,
    }
    return model, labels, metrics
