# Predicción Meteorológica Australia - Proyecto Machine Learning

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Descripción General

Este proyecto de **machine learning** utiliza **Kedro 0.19.12** para desarrollar modelos predictivos sobre datos meteorológicos de Australia. El sistema implementa múltiples algoritmos de regresión y clasificación para predicción del clima, con un enfoque en ingeniería de datos robusta y evaluación comparativa de modelos.

### Características Principales

- **13 pipelines especializados** para procesamiento de datos y modelado
- **8 modelos de regresión** (Linear, Ridge, Lasso, SVR, Random Forest, Gradient Boosting, KNN, Gaussian NB)
- **7 modelos de clasificación** (Naive Bayes, Random Forest, SVM, Decision Tree, Logistic Regression, KNN, Gradient Boosting)
- **Sistema de Model Registry** avanzado con versionado y comparación
- **Integración con PostgreSQL** y containerización Docker
- **Evaluación automática** con métricas y visualizaciones

## Arquitectura del Proyecto

### Pipelines Implementados

1. **Gestión de Datos**:
   - `data_management` - Gestión unificada de datos
   - `data_to_postgres` - Carga a base de datos
   - `data_transform` - Transformaciones de datos

2. **Selección de Características**:
   - `feature_selection_classification` - Selección para clasificación
   - `feature_selection_regression` - Selección para regresión

3. **Modelos de Regresión**:
   - `regression_models` - Implementación de modelos
   - `regression_evaluation` - Evaluación específica

4. **Modelos de Clasificación**:
   - `classification_models` - Implementación de modelos
   - `classification_evaluation` - Evaluación específica

5. **Evaluación y Comparación**:
   - `model_evaluation` - Evaluación general
   - `model_comparison` - Comparación entre modelos
   - `model_training` - Entrenamiento automatizado

6. **Registry de Modelos**:
   - `model_registry` - Gestión avanzada y versionado

### Estructura de Datos

```
data/
├── 01_raw/           # Datos originales (weatherAUS.csv)
├── 02_intermediate/  # Datos procesados intermedios
├── 03_primary/       # Datos principales limpios
├── 04_feature/       # Características engineered
├── 05_model_input/   # Datos listos para modelado
├── 06_models/        # Modelos entrenados (.pkl)
├── 07_model_output/  # Predicciones de modelos
└── 08_reporting/     # Métricas y visualizaciones
```

## Configuración e Instalación

### Requisitos Previos
- Python >= 3.9
- Docker y Docker Compose
- PostgreSQL (containerizado)

### Configuración del Ambiente Virtual

Antes de instalar las dependencias, es necesario crear un ambiente virtual de Python:

```bash
# Crear ambiente virtual
python -m venv venv

# Activar ambiente virtual
# En Windows:
venv\Scripts\activate
# En Linux/MacOS:
source venv/bin/activate
```

### Instalación de Dependencias

Una vez activado el ambiente virtual, instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

### Configuración con Docker

1. **Levantar los servicios**:
```bash
docker-compose up -d
```

2. **Verificar instalación** (si hay errores):
```bash
docker exec -it kedro_app bash -c "pip install 'kedro-datasets[pandas,sql]' --upgrade"
```

3. **Instalar dependencias adicionales**:
```bash
docker exec -it kedro_app bash -c "pip install scikit-learn matplotlib seaborn joblib pyarrow fastparquet"
```

## Ejecución del Proyecto

### Ejecutar Pipeline Completo

```bash
kedro run
```

### Ejecutar Pipelines Específicos

```bash
# Solo modelos de regresión
kedro run --pipeline regression_models

# Solo evaluación de clasificación
kedro run --pipeline classification_evaluation

# Comparación de modelos
kedro run --pipeline model_comparison
```

### Visualización con Kedro-Viz

```bash
kedro viz
```

## Trabajo con Notebooks

> **Nota**: Al usar `kedro jupyter` o `kedro ipython` tienes acceso a las variables: `context`, `session`, `catalog`, y `pipelines`.

### Jupyter Notebook
```bash
kedro jupyter notebook
```

### JupyterLab
```bash
kedro jupyter lab
```

### IPython
```bash
kedro ipython
```

### Ignorar Outputs de Notebooks en Git

Para eliminar automáticamente las salidas de las celdas antes de hacer commit:

```bash
nbstripout --install
```

> **Nota**: Las salidas se mantienen localmente, solo se excluyen del repositorio.

## Testing

Ejecuta las pruebas del proyecto:

```bash
pytest
```

Configura el umbral de cobertura en `pyproject.toml` bajo la sección `[tool.coverage.report]`.

## Gestión de Dependencias

- **Ver dependencias**: Revisa `requirements.txt`
- **Actualizar**: Modifica `requirements.txt` y ejecuta `pip install -r requirements.txt`
- **Información adicional**: [Documentación Kedro - Dependencias](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Monitoreo de Datos

### Revisar Resultados

```bash
# Listar reportes generados
docker exec -it kedro_app bash -c "ls -la /app/data/08_reporting/"

# Ver métricas específicas
docker exec -it kedro_app bash -c "cat /app/data/08_reporting/linear_regression_metrics_summary.json"
```

## Modelos Disponibles

### Regresión (Target: `risk_mm`)
- **Linear Regression** - Regresión lineal básica
- **Ridge Regression** - Regresión con regularización L2
- **Lasso Regression** - Regresión con regularización L1
- **Support Vector Regression (SVR)** - Máquinas de soporte vectorial
- **Random Forest** - Ensamble de árboles
- **Gradient Boosting** - Boosting con gradiente
- **K-Nearest Neighbors** - Vecinos más cercanos
- **Gaussian Naive Bayes** - Bayes ingenuo gaussiano

### Clasificación
- **Naive Bayes** - Clasificador bayesiano
- **Random Forest** - Ensamble de árboles para clasificación
- **Support Vector Machine (SVM)** - Máquinas de soporte vectorial
- **Decision Tree** - Árbol de decisión
- **Logistic Regression** - Regresión logística
- **K-Nearest Neighbors** - Vecinos más cercanos
- **Gradient Boosting** - Boosting con gradiente

## Empaquetado del Proyecto

Para más información sobre construcción de documentación y empaquetado: [Documentación Kedro - Empaquetado](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)

## Reglas y Mejores Prácticas

Para obtener el mejor rendimiento del proyecto:

* **No elimines** líneas del archivo `.gitignore` proporcionado
* **Asegura la reproducibilidad** siguiendo convenciones de ingeniería de datos
* **No hagas commit de datos** al repositorio
* **No hagas commit de credenciales** o configuración local. Mantén toda la configuración sensible en `conf/local/`
* **Utiliza el Model Registry** para gestionar versiones de modelos
* **Ejecuta evaluaciones** comparativas antes de deployar modelos

## Estructura del Código

El proyecto sigue la estructura estándar de Kedro con **92 archivos Python** organizados en:

- `src/weathe_aus_without_notebook/pipelines/` - Implementación de pipelines
- `src/weathe_aus_without_notebook/model_registry/` - Sistema de registry de modelos
- `conf/` - Configuraciones del proyecto
- `tests/` - Pruebas unitarias
- `notebooks/` - Notebooks de exploración
