## Comando iniciar docker

docker-compose up -d

## Apagar docker 
docker-compose down

## Para ver el estado de los contenedores:
cmddocker-compose ps

## Para ver los logs:
cmddocker-compose logs

## para ver estructura del projecto
python project_structure.py


Usage examples:

# Run  solo  data management unificada:
kedro run data_management

# Run todos modelos de regresion con data unificada:
kedro run full_regression

# Run todos modelos de clasificacion con data unificada:
kedro run full_classification

# Run todo (complete MLOps):
kedro run complete_mlops

# Run quick test (single model):
kedro run quick_test_regression
kedro run quick_test_classification

# Run specific model with unified data:
kedro run data_management,linear_regression,regression_evaluation

## hay ciertos coamnods desde ahora que tendremos que entrar al contenedor
# Entrar al contenedor:
docker-compose exec kedro bash

# Dentro del contenedor:
kedro run data_postgres

------------------------------------------------------------------------------------------------------------------------------
# Comandos de Ejecución Kedro - Pipelines

## 🔥 MODELOS DE REGRESIÓN

```bash
# Regresión Lineal
kedro run linear_regression

# Random Forest Regresión  
kedro run random_forest_regression

# KNN Regressor
kedro run knn_regressor

# Lasso Regression
kedro run lasso

# Ridge Regression
kedro run ridge

# Support Vector Regression
kedro run svr

# Bayes Regression
kedro run bayes_regression

# Gradient Boosting Regression
kedro run gradient_boosting_regression
```

## 🎯 MODELOS DE CLASIFICACIÓN

```bash
# Bayes Classification
kedro run --pipeline bayes_classification

# Decision Tree Classifier
kedro run --pipeline decision_tree

# Gradient Boosting Classifier
kedro run --pipeline gradient_boosting_classification

# KNN Classifier
kedro run --pipeline knn_classification

# Logistic Regression Classifier
kedro run --pipeline logistic_regression

# Random Forest Classifier
kedro run --pipeline random_forest_classification

# SVM Classifier
kedro run --pipeline svm_classification
```

## 📊 PIPELINES DE DATOS

```bash
# Gestión de datos unificada (nueva versión)
kedro run data_management

# Datos a PostgreSQL (legacy)
kedro run data_postgres

# Transformación de datos (legacy)
kedro run data_transform

# Selección de features para clasificación
kedro run feature_selection_classification

# Selección de features para regresión
kedro run feature_selection_regression
```

## 🔄 PIPELINES DE EVALUACIÓN

```bash
# Evaluación de regresión
kedro run regression_evaluation

# Evaluación de clasificación
kedro run classification_evaluation

# Comparación de modelos
kedro run model_comparison

# Evaluación de modelos
kedro run model_evaluation

# Entrenamiento de modelos
kedro run model_training
```

## 🚀 PIPELINES COMBINADOS (MLOps)

```bash
# Toda la regresión (datos + todos los modelos + evaluación)
kedro run full_regression

# Toda la clasificación (datos + todos los modelos + evaluación)
kedro run full_classification

# Pipeline MLOps completo (TODO)
kedro run complete_mlops

# Pipeline por defecto (ejecuta complete_mlops)
kedro run
```

## ⚡ PIPELINES DE PRUEBA RÁPIDA

```bash
# Test rápido de regresión (datos + linear regression + evaluación)
kedro run quick_test_regression

# Test rápido de clasificación (datos + logistic regression + evaluación)
kedro run quick_test_classification
```

## 🛠️ COMANDOS ÚTILES DE DEBUG

```bash
# Ver todos los pipelines registrados
kedro registry list

# Ejecutar hasta un nodo específico
kedro run --to-nodes=nombre_del_nodo

# Ejecutar desde un nodo específico
kedro run --from-nodes=nombre_del_nodo

# Ver la estructura del pipeline
kedro viz

# Ver ayuda de comandos
kedro run --help

# Ejecutar con logs detallados
kedro run nombre_pipeline --log-level=DEBUG
```

## 🎯 COMANDOS ESPECÍFICOS POR TAREA

### Solo Modelos de Regresión:
```bash
for model in linear_regression random_forest_regression knn_regressor lasso ridge svr bayes_regression gradient_boosting_regression; do
    kedro run $model
done
```

### Solo Modelos de Clasificación:
```bash
for model in bayes_classification decision_tree gradient_boosting_classification knn_classification logistic_regression random_forest_classification svm_classification; do
    kedro run $model
done
```

## 📝 NOTAS IMPORTANTES

- **Pipeline por defecto**: `kedro run` ejecuta `complete_mlops`
- **Nombres exactos**: Usar exactamente los nombres mostrados (sin `_pipeline` al final)
- **Dependencias**: Algunos pipelines requieren que se ejecuten los de datos primero
- **Logs**: Los logs detallados ayudan a debuggear problemas
- **Orden recomendado**: 
  1. `data_management` 
  2. Modelo específico
  3. Evaluación específica

## 🔧 COMANDOS DE MANTENIMIENTO

```bash
# Limpiar artifacts
kedro clean

# Reinstalar dependencias
kedro install

# Verificar configuración
kedro info

# Ver estructura del proyecto
kedro catalog list
```