"""Página de preprocesamiento de datos para el flujo quimiométrico."""

from __future__ import annotations
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import pandas as pd
import plotly.express as px

from scripts.preprocessing import compute_missing_summary, infer_variable_types, preprocess_data

# Paleta de colores
from scripts.io_utils import get_discrete_palette, get_continuous_palette

palette_name = st.session_state.get("plot_color_palette", "deep")
discrete_colors = get_discrete_palette(palette_name)
continuous_colors = get_continuous_palette(palette_name)

palette_name = st.session_state.get("plot_color_palette", "deep")
discrete_colors = get_discrete_palette(palette_name)
continuous_colors = get_continuous_palette(palette_name)

VISIBLE_TYPE_OPTIONS = {
    "Numérica": "numeric",
    "Categórica": "categorical",
    "Fecha": "date",
}


def render_header() -> None:
    """Encabezado de la página con descripción breve."""

    st.set_page_config(page_title="Preprocesamiento")
    st.title("Preprocesamiento de datos")
    st.markdown(
        """
        Defina los tipos de variables, configure la imputación de valores faltantes y
        estandarice variables numéricas antes de aplicar PCA y clustering.
        """
    )


def render_variable_type_table(df: pd.DataFrame) -> None:
    """Tabla interactiva para seleccionar tipo y uso de cada columna."""

    st.subheader("Definición de tipos de variables")
    st.caption(
        "Seleccione el tipo adecuado para cada columna y marque cuáles se incluirán en el análisis."
    )

    # Inicializar esquema y flags de selección
    if "schema" not in st.session_state or not st.session_state.get("schema"):
        st.session_state["schema"] = infer_variable_types(df)

    if "selected_variables_flags" not in st.session_state:
        st.session_state["selected_variables_flags"] = {
            col: True for col in df.columns
        }

    # Botón seleccionar / deseleccionar todas
    flags = st.session_state["selected_variables_flags"]
    all_selected = all(flags.get(col, False) for col in df.columns)

    if st.button("Seleccionar / quitar todas las variables"):
        new_value = not all_selected  # si todas están seleccionadas, las quita; si no, las selecciona todas
        for col in df.columns:
            flags[col] = new_value
            # actualizar también el estado de los checkboxes
            st.session_state[f"include_{col}"] = new_value
        st.session_state["selected_variables_flags"] = flags


    inferred = infer_variable_types(df)

    for column in df.columns:
        non_null = df[column].dropna()
        example_value = non_null.iloc[0] if len(non_null) > 0 else ""
        current_internal = st.session_state["schema"].get(
            column, inferred.get(column, "numeric")
        )

        options = list(VISIBLE_TYPE_OPTIONS.keys())
        default_label = next(
            (label for label, internal in VISIBLE_TYPE_OPTIONS.items()
             if internal == current_internal),
            "Numérica",
        )

        include_flag_default = st.session_state["selected_variables_flags"].get(
            column, True
        )

        with st.container():
            cols = st.columns([3, 3, 2])
            cols[0].markdown(f"**{column}**")
            cols[0].caption(f"Ejemplo: {example_value}")

            selected_label = cols[1].selectbox(
                "Tipo de variable",
                options,
                index=options.index(default_label),
                key=f"schema_{column}",
            )
            st.session_state["schema"][column] = VISIBLE_TYPE_OPTIONS[selected_label]

            include_flag = cols[2].checkbox(
                "Incluir",
                value=include_flag_default,
                key=f"include_{column}",
            )
            st.session_state["selected_variables_flags"][column] = include_flag

    # Botón para consolidar selección
    if st.button("Listo selección de variables"):
        selected = [
            col
            for col, flag in st.session_state["selected_variables_flags"].items()
            if flag
        ]
        if not selected:
            st.error("Debe seleccionar al menos una variable para el análisis.")
            st.session_state["variables_confirmed"] = False
        else:
            st.session_state["selected_variables"] = selected
            st.session_state["variables_confirmed"] = True
            st.success(f"Se seleccionaron {len(selected)} variables para el análisis.")


def render_missing_and_imputation_controls() -> None:
    """Configurar imputación por variable sobre el working_df."""

    st.subheader("Valores faltantes e imputación")

    # Siempre trabajar sobre el DF de trabajo
    if "working_df" not in st.session_state:
        st.info("No hay datos para preprocesar todavía.")
        return

    df = st.session_state["working_df"]

    # Resumen de faltantes
    missing_summary = compute_missing_summary(df)
    st.dataframe(missing_summary, use_container_width=True)

    # Columnas que aún tienen NA
    cols_with_na = missing_summary.loc[missing_summary["faltantes"] > 0, "columna"].tolist()

    if not cols_with_na:
        st.info("Ninguna variable tiene datos nulos.")
        return

    # Selección de columna con NA
    selected_column = st.radio(
        "Seleccione una columna con valores faltantes",
        cols_with_na,
        key="imputation_column",
    )

    # Tipo de la columna según el schema
    col_type = st.session_state["schema"].get(selected_column, "numeric")

    # Opciones de imputación
    strategy_labels = []
    strategy_values = []

    strategy_labels.append("Eliminar filas con NA")
    strategy_values.append("drop_rows")

    if col_type == "numeric":
        strategy_labels.extend(["Media", "Mediana"])
        strategy_values.extend(["mean", "median"])

    # Para numéricas y categóricas, permitir moda
    if col_type in {"numeric", "categorical"}:
        strategy_labels.append("Moda")
        strategy_values.append("mode")

    strategy_labels.append("Valor constante")
    strategy_values.append("constant")

    strategy_label = st.selectbox(
        "Método de imputación",
        strategy_labels,
        key="imputation_strategy",
    )
    strategy = dict(zip(strategy_labels, strategy_values))[strategy_label]

    fill_value = None
    if strategy == "constant":
        fill_value = st.text_input(
            "Valor constante para imputación",
            key="imputation_constant",
        )

    # Botón para aplicar SOLO a la columna seleccionada
    if st.button("Aplicar imputación a esta variable"):
        try:
            working_df = st.session_state["working_df"].copy()
            series = working_df[selected_column]

            if strategy == "drop_rows":
                # Filtrar filas donde esta columna no es NA
                working_df = working_df[series.notna()].copy()

            elif strategy in {"mean", "median"}:
                if not is_numeric_dtype(series):
                    st.error(
                        f"La estrategia '{strategy_label}' solo es válida para variables numéricas."
                    )
                    return
                value = series.mean() if strategy == "mean" else series.median()
                working_df[selected_column] = series.fillna(value)

            elif strategy == "mode":
                mode_series = series.mode(dropna=True)
                if mode_series.empty:
                    st.error(
                        f"No se pudo calcular la moda para la columna '{selected_column}'."
                    )
                    return
                working_df[selected_column] = series.fillna(mode_series.iloc[0])

            elif strategy == "constant":
                if fill_value is None or fill_value == "":
                    st.error("Debe proporcionar un valor constante para la imputación.")
                    return
                # No forcemos tipo; dejamos que pandas intente hacer el cast
                working_df[selected_column] = series.fillna(fill_value)

            else:
                st.error(f"Estrategia de imputación no reconocida: {strategy}")
                return

            # Guardar DF actualizado en sesión
            st.session_state["working_df"] = working_df

            st.success(
                f"Imputación aplicada a la columna '{selected_column}'. "
                "Si ya no tiene valores faltantes, dejará de aparecer en la lista."
            )

        except Exception as exc:
            st.error(f"No se pudo aplicar la imputación: {exc}")


def render_quick_numeric_plots(df: pd.DataFrame) -> None:
    """
    Botones para visualizar:
    - Mapa de correlación (Plotly).
    - Boxplot por variable numérica.
    - Distribución (histograma) por variable numérica.
    """

    st.subheader("Exploración rápida de variables numéricas")

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        st.info("No hay variables numéricas para explorar.")
        return

    tab_corr, tab_box, tab_hist = st.tabs(
        ["Mapa de correlación (Plotly)", "Boxplot por variable", "Distribución / histograma"]
    )

    # ========== TAB 1: CORRELACIÓN ==========
    with tab_corr:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            color_continuous_scale=continuous_colors,
            aspect="auto",
            title="Matriz de correlación",
        )
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 2: BOXPLOT ==========
    with tab_box:
        col_to_plot = st.selectbox("Seleccione variable", numeric_df.columns, key="box_var")
        fig = px.box(numeric_df, y=col_to_plot, points="all", title=f"Boxplot de {col_to_plot}",
                     color_discrete_sequence=discrete_colors)
        st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 3: HISTOGRAMA ==========
    with tab_hist:
        col_to_plot = st.selectbox("Variable para histograma", numeric_df.columns, key="hist_var")
        fig = px.histogram(
            numeric_df,
            x=col_to_plot,
            nbins=30,
            marginal="rug",
            color_discrete_sequence=discrete_colors,
            title=f"Distribución de {col_to_plot}",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_apply_preprocessing_button(original_df: pd.DataFrame) -> None:
    """Botón para ejecutar el pipeline de preprocesamiento (dtypes + escalado)."""

    st.subheader("Aplicar cambios")

    # Opciones de escalado global para numéricas
    scaling_mode = st.radio(
        "Escalado para todas las variables numéricas seleccionadas:",
        (
            "Sin escalado",
            "Estandarizar (media 0, varianza 1)",
            "Escalado Min-Max [0,1]",
        ),
        index=0,
        key="scaling_mode_choice",
    )

    if st.button("Aplicar preprocesamiento", type="primary"):
        try:
            working_df = st.session_state.get("working_df", original_df)

            # Traducir opción de texto a modo interno
            if scaling_mode.startswith("Sin"):
                mode = "none"
            elif scaling_mode.startswith("Estandarizar"):
                mode = "standard"
            else:
                mode = "minmax"

            clean_df = preprocess_data(
                working_df,
                schema=st.session_state.get("schema"),
                imputation_config={},  # ya imputamos manualmente
                scaling_config={},     # dejamos que scaling_mode controle el escalado
                scaling_mode=mode,
            )
            st.session_state["clean_df"] = clean_df
            st.success("Preprocesamiento aplicado correctamente.")

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ocurrió un error inesperado al preprocesar los datos: {exc}")


def main() -> None:
    render_header()

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Primero cargue un dataset en la página de Inicio.")
        return

    # DF de trabajo sobre el que se irán aplicando imputaciones
    st.session_state.setdefault("working_df", raw_df.copy())

    st.session_state.setdefault("schema", infer_variable_types(raw_df))
    st.session_state.setdefault("imputation_config", {})
    st.session_state.setdefault("scaling_config", {})
    st.session_state.setdefault("clean_df", None)

    render_variable_type_table(raw_df)
    render_missing_and_imputation_controls()   # 👈 ahora sin df como argumento
    render_apply_preprocessing_button(raw_df)
    render_quick_numeric_plots(st.session_state["working_df"])


if __name__ == "__main__":
    main()
