"""Funciones de preprocesamiento para la app de análisis quimiométrico."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler


Schema = Dict[str, str]
ImputationConfig = Dict[str, Dict[str, object]]
ScalingConfig = Dict[str, bool]


def infer_variable_types(df: pd.DataFrame) -> Schema:
    """Inferir tipos de variables basados en el dtype de pandas."""

    inferred: Schema = {}
    for column in df.columns:
        series = df[column]
        if is_datetime64_any_dtype(series):
            inferred[column] = "date"
        elif is_numeric_dtype(series):
            inferred[column] = "numeric"
        else:
            inferred[column] = "categorical"
    return inferred


def convert_dtypes(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Convertir columnas según el esquema proporcionado."""

    converted = df.copy()
    for column, col_type in schema.items():
        if column not in converted.columns:
            raise ValueError(f"La columna '{column}' no existe en los datos.")
        try:
            if col_type == "numeric":
                converted[column] = pd.to_numeric(converted[column], errors="raise")
            elif col_type == "date":
                converted[column] = pd.to_datetime(converted[column], errors="raise")
            elif col_type == "categorical":
                converted[column] = converted[column].astype("category")
            else:
                raise ValueError(f"Tipo no soportado para la columna '{column}': {col_type}")
        except Exception as exc:  # pragma: no cover - feedback específico
            raise ValueError(
                f"Error al convertir la columna '{column}' al tipo {col_type}: {exc}"
            ) from exc
    return converted


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calcular el resumen de valores faltantes por columna."""

    total_rows = len(df)
    summary_rows: Iterable[Tuple[str, int, float]] = (
        (col, int(df[col].isna().sum()), float(df[col].isna().mean()) * 100)
        for col in df.columns
    )
    summary_df = pd.DataFrame(summary_rows, columns=["columna", "faltantes", "%"])
    summary_df["%"] = summary_df["%"].round(2)
    summary_df["filas_totales"] = total_rows
    return summary_df


def apply_imputation(
    df: pd.DataFrame, imputation_config: ImputationConfig, schema: Schema
) -> pd.DataFrame:
    """Aplicar imputación según configuración por columna."""

    imputed = df.copy()
    for column, config in imputation_config.items():
        if column not in imputed.columns:
            raise ValueError(f"La columna '{column}' no existe en los datos.")
        strategy = config.get("strategy", "none") if config else "none"
        fill_value = config.get("fill_value") if config else None

        if strategy == "none":
            continue
        if strategy == "drop_rows":
            imputed = imputed[imputed[column].notna()]
            continue

        col_type = schema.get(column)
        series = imputed[column]

        if strategy in {"mean", "median"}:
            if not is_numeric_dtype(series):
                raise ValueError(
                    f"La estrategia '{strategy}' solo es válida para datos numéricos en la columna '{column}'."
                )
            value = series.mean() if strategy == "mean" else series.median()
            imputed[column] = series.fillna(value)
        elif strategy == "mode":
            mode_series = series.mode(dropna=True)
            if mode_series.empty:
                raise ValueError(
                    f"No se pudo calcular la moda para la columna '{column}' porque no hay valores válidos."
                )
            imputed[column] = series.fillna(mode_series.iloc[0])
        elif strategy == "constant":
            if fill_value is None:
                raise ValueError(
                    f"Debe proporcionar un valor constante para imputar la columna '{column}'."
                )
            imputed[column] = series.fillna(fill_value)
        else:
            raise ValueError(
                f"Estrategia de imputación no reconocida para la columna '{column}': {strategy}"
            )
    return imputed


def apply_scaling(df: pd.DataFrame, scaling_config: ScalingConfig, schema: Schema) -> pd.DataFrame:
    """Estandarizar columnas numéricas según la configuración."""

    scaled = df.copy()
    numeric_to_scale = [
        col for col, flag in scaling_config.items() if flag and schema.get(col) == "numeric"
    ]
    if not numeric_to_scale:
        return scaled

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(scaled[numeric_to_scale])
    scaled[numeric_to_scale] = scaled_values
    return scaled


def preprocess_data(
    df: pd.DataFrame,
    schema: Schema | None = None,
    imputation_config: ImputationConfig | None = None,
    scaling_config: ScalingConfig | None = None,
) -> pd.DataFrame:
    """Ejecutar el pipeline completo de preprocesamiento."""

    if df is None or df.empty:
        raise ValueError("No hay datos para preprocesar.")

    schema = schema or infer_variable_types(df)
    inferred_schema = infer_variable_types(df)
    for column, inferred_type in inferred_schema.items():
        schema.setdefault(column, inferred_type)

    imputation_config = imputation_config or {}
    scaling_config = scaling_config or {}

    converted = convert_dtypes(df, schema)
    imputed = apply_imputation(converted, imputation_config, schema)
    scaled = apply_scaling(imputed, scaling_config, schema)
    return scaled
