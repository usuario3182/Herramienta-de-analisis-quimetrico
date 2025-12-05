"""
pca_utils.py

Funciones auxiliares para realizar PCA dentro de la aplicación de análisis
quimiométrico.

RESPONSABILIDAD DEL MÓDULO:
---------------------------
Este módulo NO maneja Streamlit ni la interfaz. Solo:
- Prepara los datos para PCA (selección de columnas numéricas).
- Ejecuta PCA usando scikit-learn.
- Devuelve:
    * El modelo PCA ajustado.
    * Los scores (coordenadas de las muestras en el espacio de componentes).
    * Los loadings (contribución de cada variable original a cada componente).
    * Una tabla con varianza explicada y varianza acumulada.

SE USA PRINCIPALMENTE EN:
-------------------------
- app/pages/3_PCA.py

CONVENCIONES:
-------------
- Input principal: DataFrame limpio (clean_df) desde st.session_state.
- Solo se usan columnas numéricas para PCA.
- Los nombres de componentes serán "PC1", "PC2", ..., "PCk".

Funciones a implementar:

1) select_numeric_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame
   - Si columns es None o vacío: seleccionar todas las columnas numéricas de df.
   - Si columns se especifica: tomar solo esas columnas (y validar que existan).
   - Levantar ValueError si no hay columnas numéricas.

2) run_pca(
       df: pd.DataFrame,
       n_components: int,
       columns: list[str] | None = None
   ) -> tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]
   - Seleccionar columnas numéricas (usando select_numeric_columns).
   - Ajustar sklearn.decomposition.PCA con n_components.
   - Devolver:
        model: objeto PCA entrenado.
        scores_df: DataFrame (muestras x componentes) con columnas "PC1", ...
        loadings_df: DataFrame (variables x componentes) con columnas "PC1", ...
        explained_df: DataFrame con columnas:
            "Componente", "Varianza", "Proporción_varianza", "Varianza_acumulada"

3) build_explained_variance_table(model: PCA) -> pd.DataFrame
   - A partir del modelo PCA, crear la tabla de varianza explicada y acumulada.
   - Esta función puede usarse de forma independiente si es necesario.

Codex: Implementa estas funciones de forma clara y robusta. Serán usadas por
la página 3 (PCA) para mostrar tablas y gráficos interactivos.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.decomposition import PCA

def select_numeric_columns(df: pd.DataFrame, columns: List[str] | None = None) -> pd.DataFrame:
    """TODO: Seleccionar columnas numéricas y validar entrada."""
    raise NotImplementedError


def build_explained_variance_table(model: PCA) -> pd.DataFrame:
    """TODO: Construir tabla con varianza explicada y acumulada por componente."""
    raise NotImplementedError


def run_pca(
    df: pd.DataFrame,
    n_components: int,
    columns: List[str] | None = None,
) -> Tuple[PCA, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TODO: Ejecutar PCA y devolver modelo, scores, loadings y tabla de varianza."""
    raise NotImplementedError
