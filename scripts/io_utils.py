"""Utility functions for loading datasets into the Streamlit app.

These helpers centralize the logic for reading local example files and user
uploads, providing clear errors that can be surfaced in the UI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


EXAMPLE_FILENAME = "ejemplo_quimiometria.xls"

# scripts/io_utils.py (o donde tú quieras)

import plotly.express as px

# Paletas discretas sugeridas por nombre
DISCRETE_PALETTES = {
    "deep": px.colors.qualitative.Plotly,
    "muted": px.colors.qualitative.Safe,
    "pastel": px.colors.qualitative.Pastel,
    "bright": px.colors.qualitative.Bold,
    "dark": px.colors.qualitative.Dark2,
    "colorblind": px.colors.qualitative.Safe,
    "viridis": px.colors.qualitative.Vivid,  # para series discretas
}

# Paletas continuas
CONTINUOUS_PALETTES = {
    "deep": px.colors.sequential.Blues,
    "muted": px.colors.sequential.GnBu,
    "pastel": px.colors.sequential.Sunset,
    "bright": px.colors.sequential.YlOrRd,
    "dark": px.colors.sequential.Magma,
    "colorblind": px.colors.sequential.Cividis,
    "viridis": px.colors.sequential.Viridis,
}

def get_discrete_palette(name: str | None) -> list[str] | None:
    """Devolver secuencia de colores discretos según el nombre guardado."""
    if not name:
        return None
    return DISCRETE_PALETTES.get(name)

def get_continuous_palette(name: str | None) -> list[str] | None:
    """Devolver escala continua según el nombre guardado."""
    if not name:
        return None
    return CONTINUOUS_PALETTES.get(name)


def load_example_dataset(base_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the bundled chemometrics example dataset.

    Parameters
    ----------
    base_path:
        Optional base directory. Defaults to the repository root (one level up
        from the scripts directory).

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with the example data.

    Raises
    ------
    FileNotFoundError
        If the example dataset is not found.
    ValueError
        If the file cannot be read as a tabular dataset.
    """

    root_dir = base_path or Path(__file__).resolve().parents[1]
    data_path = root_dir / "data" / EXAMPLE_FILENAME

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset de ejemplo en {data_path}.")

    try:
        df = pd.read_excel(data_path)
    except Exception as exc:  # pragma: no cover - defensive for unexpected read errors
        raise ValueError(f"No se pudo leer el dataset de ejemplo: {exc}") from exc

    if df.empty:
        raise ValueError("El dataset de ejemplo está vacío.")

    return df


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Load a user-provided file in CSV or Excel format.

    Parameters
    ----------
    uploaded_file:
        The file-like object returned by ``st.file_uploader``.

    Returns
    -------
    pd.DataFrame
        Dataframe built from the uploaded file.

    Raises
    ------
    ValueError
        If the file type is unsupported or cannot be parsed.
    """

    if uploaded_file is None:
        raise ValueError("No se proporcionó ningún archivo para cargar.")

    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith((".xls", ".xlsx", ".xlsm")):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("Formato no soportado. Cargue un archivo CSV o Excel.")
    except Exception as exc:  # pragma: no cover - defensive for unexpected read errors
        raise ValueError(f"No se pudo leer el archivo subido: {exc}") from exc

    if df.empty:
        raise ValueError("El archivo cargado no contiene datos.")

    return df


def describe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small descriptive summary of the dataframe.

    Parameters
    ----------
    df:
        Dataframe to describe.

    Returns
    -------
    pd.DataFrame
        Summary metrics with human-friendly labels.
    """

    if df is None:
        raise ValueError("No se proporcionó un dataframe para describir.")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64[ns]", "datetimetz"]).columns

    summary = {
        "Filas": len(df),
        "Columnas": df.shape[1],
        "Variables numéricas": len(numeric_cols),
        "Variables categóricas": len(categorical_cols),
        "Variables fecha": len(datetime_cols),
        "Valores faltantes": int(df.isna().sum().sum()),
    }

    return pd.DataFrame(
        {"Métrica": list(summary.keys()), "Valor": list(summary.values())}
    )


# Backwards compatibility with earlier name
load_uploaded_file = load_uploaded_dataset
