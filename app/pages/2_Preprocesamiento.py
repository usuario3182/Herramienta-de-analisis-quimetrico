"""Página de preprocesamiento de datos para el flujo quimiométrico."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from scripts.preprocessing import compute_missing_summary, infer_variable_types, preprocess_data


VISIBLE_TYPE_OPTIONS = {
    "Numérica": "numeric",
    "Categórica": "categorical",
    "Fecha": "date",
}


def render_header() -> None:
    """Encabezado de la página con descripción breve."""

    st.set_page_config(page_title="Preprocesamiento", page_icon="⚙️")
    st.title("Preprocesamiento de datos")
    st.markdown(
        """
        Defina los tipos de variables, configure la imputación de valores faltantes y
        estandarice variables numéricas antes de aplicar PCA y clustering.
        """
    )


def render_variable_type_table(df: pd.DataFrame) -> None:
    """Tabla interactiva para seleccionar tipo de cada columna."""

    st.subheader("Definición de tipos de variables")
    st.caption("Seleccione el tipo adecuado para cada columna del dataset.")

    if "schema" not in st.session_state or not st.session_state.get("schema"):
        st.session_state["schema"] = infer_variable_types(df)

    inferred = infer_variable_types(df)

    for column in df.columns:
        non_null = df[column].dropna()
        example_value = non_null.iloc[0] if len(non_null) > 0 else ""
        current_internal = st.session_state["schema"].get(column, inferred.get(column, "numeric"))
        options = list(VISIBLE_TYPE_OPTIONS.keys())
        default_label = next(
            (label for label, internal in VISIBLE_TYPE_OPTIONS.items() if internal == current_internal),
            "Numérica",
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


def render_missing_and_imputation_controls(df: pd.DataFrame) -> None:
    """Configurar imputación y escalado por variable."""

    st.subheader("Valores faltantes e imputación")
    missing_summary = compute_missing_summary(df)
    st.dataframe(missing_summary, use_container_width=True)

    if "imputation_config" not in st.session_state:
        st.session_state["imputation_config"] = {}
    if "scaling_config" not in st.session_state:
        st.session_state["scaling_config"] = {}

    columns = list(df.columns)
    if not columns:
        st.info("No hay columnas para configurar.")
        return

    selected_column = st.selectbox("Seleccione una columna", columns)
    col_type = st.session_state["schema"].get(selected_column, infer_variable_types(df).get(selected_column))

    strategy_options = [
        ("No imputar", "none"),
        ("Eliminar filas con NA", "drop_rows"),
    ]
    if col_type == "numeric":
        strategy_options.extend([("Media", "mean"), ("Mediana", "median")])
    if col_type in {"categorical", "numeric"}:
        strategy_options.append(("Moda", "mode"))
    strategy_options.append(("Valor constante", "constant"))

    labels = [label for label, _ in strategy_options]
    values = [value for _, value in strategy_options]

    current_strategy = st.session_state["imputation_config"].get(selected_column, {}).get("strategy", "none")
    strategy_label_default = labels[values.index(current_strategy)] if current_strategy in values else labels[0]

    cols = st.columns([2, 2, 2])
    chosen_label = cols[0].selectbox(
        "Método de imputación",
        labels,
        index=labels.index(strategy_label_default),
    )
    chosen_strategy = dict(strategy_options)[chosen_label]

    fill_value = st.session_state["imputation_config"].get(selected_column, {}).get("fill_value")
    if chosen_strategy == "constant":
        fill_value = cols[1].text_input("Valor constante", value=fill_value or "")
    else:
        fill_value = None

    st.session_state["imputation_config"][selected_column] = {
        "strategy": chosen_strategy,
        "fill_value": fill_value,
    }

    if col_type == "numeric":
        current_scale = bool(st.session_state["scaling_config"].get(selected_column, False))
        scale_flag = cols[2].checkbox("Estandarizar variable", value=current_scale)
        st.session_state["scaling_config"][selected_column] = scale_flag

    plot_cols = st.columns(2)
    if plot_cols[0].button("Ver histograma"):
        if col_type == "numeric":
            fig, ax = plt.subplots()
            
            series = df[selected_column].dropna()
            df_plot = series.to_frame(name=selected_column)
            
            sns.histplot(data=df_plot, x=selected_column, kde=True, ax=ax)
            ax.set_title(f"Histograma de {selected_column}")
            st.pyplot(fig)
        else:
            st.warning("El histograma solo aplica a variables numéricas.")

    if plot_cols[1].button("Ver boxplot"):
        if col_type == "numeric":
            series = df[selected_column].dropna()
            df_plot = series.to_frame(name=selected_column)

            fig, ax = plt.subplots()
            sns.boxplot(data=df_plot, x=selected_column, ax=ax)
            ax.set_title(f"Boxplot de {selected_column}")
            st.pyplot(fig)
        else:
            st.warning("El boxplot solo aplica a variables numéricas.")


def render_apply_preprocessing_button(df: pd.DataFrame) -> None:
    """Botón para ejecutar el pipeline de preprocesamiento."""

    st.subheader("Aplicar cambios")
    if st.button("Aplicar preprocesamiento", type="primary"):
        try:
            clean_df = preprocess_data(
                df,
                schema=st.session_state.get("schema"),
                imputation_config=st.session_state.get("imputation_config"),
                scaling_config=st.session_state.get("scaling_config"),
            )
            st.session_state["clean_df"] = clean_df
            st.success("Preprocesamiento aplicado correctamente.")
        except ValueError as exc:  # pragma: no cover - feedback para el usuario
            st.error(str(exc))


def render_correlation_heatmap() -> None:
    """Mostrar botón para calcular y dibujar el mapa de correlación."""

    clean_df = st.session_state.get("clean_df")
    if clean_df is None:
        st.info("Aplique el preprocesamiento para habilitar el mapa de correlación.")
        return

    if st.button("Mostrar mapa de correlación"):
        numeric_df = clean_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            st.warning("No hay variables numéricas para calcular correlación.")
            return
        corr = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Mapa de correlación (variables numéricas)")
        st.pyplot(fig)


def main() -> None:
    render_header()

    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Primero cargue un dataset en la página de Inicio.")
        return

    st.session_state.setdefault("schema", infer_variable_types(raw_df))
    st.session_state.setdefault("imputation_config", {})
    st.session_state.setdefault("scaling_config", {})
    st.session_state.setdefault("clean_df", None)

    render_variable_type_table(raw_df)
    render_missing_and_imputation_controls(raw_df)
    render_apply_preprocessing_button(raw_df)
    render_correlation_heatmap()


if __name__ == "__main__":
    main()
