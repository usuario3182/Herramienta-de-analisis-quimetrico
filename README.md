# Herramienta-de-analisis-quimetrico

# Instrucciones para replicar:
## Clonar el repositorio
`git clone https://github.com/usuario/Herramienta-de-analisis-quimetrico.git`

`cd Herramienta-de-analisis-quimetrico`

## Crear entorno virtual

`python -m venv .venv`

Linux/Mac

`source .venv/bin/activate`   

Windows

`.\.venv\Scripts\activate`    

## Instalar dependencias: 
`pip install -r requirements.txt`

## Ejecutar la aplicación: 
`streamlit run app/app.py`

# Aplicación modular en Streamlit para preprocesamiento, PCA, clustering y exportación de resultados
Este repositorio contiene una plataforma interactiva diseñada para realizar análisis quimiométrico de forma intuitiva y reproducible.
Su arquitectura modular permite que el usuario siga un flujo guiado desde la carga del dataset hasta la interpretación de resultados finales.
La herramienta está pensada para aplicaciones como:
* Análisis de aceites y grasas (FAMEs, concentraciones, perfiles químicos)
* Análisis multivariado en laboratorio

# Funcionalidades principales
Preprocesamiento de datos
* Selección manual de variables a incluir en el análisis
* Conversión automática e interactiva de tipos de variable
* Imputación por variable: media, mediana, moda, eliminación de filas, constante
* Soporte opcional para One-Hot-Encoding
* Escalado global:
* Sin escalado
* Estandarización (z-score)
* Min–Max [0, 1]
* Exploración gráfica interactiva:
* Matriz de correlación (Plotly, adaptada a tema oscuro)
* Boxplots
* Histogramas

# PCA — Análisis de Componentes Principales
Cálculo completo de PCA
* Scree plot
* Scores PC1 vs PC2 u otras combinaciones
* Biplot con vectores de loadings y escalado dinámico
* Varianza explicada y tabla de loadings
* Exportación de visualizaciones
* Uso de paleta de colores seleccionada por el usuario


# Clustering
Métodos disponibles:
* K-Means
* Clustering jerárquico (HCA) con selección de linkage y affinity

Clustering sobre:
* Variables preprocesadas
* Componentes principales (recomendado)

Se generan gráficas de:
* Dendrograma generado con plotly.figure_factory
* Métricas: silhouette, inertia
* Gráfica de clústeres en espacio PC (PC1–PC2 por defecto)
* Resumen por clúster:
* Centroides en PC-space
* Medias de variables originales


# Exportación de resultados
Descarga de:
* Datos originales
* Datos preprocesados
* Scores y loadings de PCA
* Etiquetas de clustering

Métricas
* Generación dinámica de archivos .csv y .json

# Estructura del repositorio:
Herramienta-de-analisis-quimetrico/

```
Herramienta-de-analisis-quimetrico/
- app/
    - app.py
    - pages/
        - 0_Ayuda_Interpretacion.py
        - 1_Home.py
        - 2_Preprocesamiento.py
        - 3_PCA.py
        - 4_Clustering.py
        - 5_Resultados_Exportacion.py
- scripts/
    - preprocessing.py
    - pca_utils.py
    - clustering_utils.py
    - io_utils.py
- requirements.txt
- README.md
```

