"""Utilidades para cálculo y visualización de PCA en la app de quimiometría."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA


def select_numeric_columns(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Selecciona columnas numéricas respetando la lista indicada.

    Args:
        df: DataFrame de entrada.
        columns: Lista de columnas elegidas por el usuario. Si es None o vacía,
            se seleccionan todas las columnas numéricas disponibles.

    Raises:
        ValueError: si no hay columnas numéricas disponibles o alguna columna
            solicitada no existe.
    """

    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Las columnas solicitadas no existen en los datos: {', '.join(missing)}")
        candidate_df = df[list(columns)]
    else:
        candidate_df = df.select_dtypes(include=["number"]).copy()

    numeric_cols = [col for col in candidate_df.columns if is_numeric_dtype(candidate_df[col])]
    if not numeric_cols:
        raise ValueError("No hay columnas numéricas disponibles para realizar PCA.")

    return candidate_df[numeric_cols]


def build_explained_variance_table(model: PCA) -> pd.DataFrame:
    """Construye una tabla con la varianza explicada del modelo PCA."""

    components = [f"PC{i}" for i in range(1, len(model.explained_variance_) + 1)]
    df = pd.DataFrame(
        {
            "Componente": components,
            "Varianza": model.explained_variance_,
            "Proporción_varianza": model.explained_variance_ratio_,
        }
    )
    df["Varianza_acumulada"] = df["Proporción_varianza"].cumsum()
    df["Proporción_varianza"] = df["Proporción_varianza"].round(4)
    df["Varianza_acumulada"] = df["Varianza_acumulada"].round(4)
    return df


def run_pca(
    df: pd.DataFrame, n_components: int, columns: Optional[Sequence[str]] = None
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta PCA y devuelve modelo, scores, loadings y varianza explicada."""

    X = select_numeric_columns(df, columns)
    if n_components is None:
        raise ValueError("Debe especificar el número de componentes a calcular.")
    if n_components < 1 or n_components > X.shape[1]:
        raise ValueError(
            f"El número de componentes debe estar entre 1 y {X.shape[1]}, se recibió {n_components}."
        )

    model = PCA(n_components=n_components)
    scores = model.fit_transform(X)

    component_names = [f"PC{i}" for i in range(1, n_components + 1)]
    scores_df = pd.DataFrame(scores, index=df.index, columns=component_names)
    loadings_df = pd.DataFrame(model.components_.T, index=X.columns, columns=component_names)
    explained_df = build_explained_variance_table(model)

    return model, scores_df, loadings_df, explained_df
