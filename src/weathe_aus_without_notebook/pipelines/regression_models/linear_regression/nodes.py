"""
Nodos para el pipeline de regresión lineal.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import json
import os
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

def train_linear_regression(modeling_data_regression: pd.DataFrame, parameters: Dict) -> None:
    """
    Entrena un modelo de regresión lineal.
    
    Args:
        modeling_data_regression: DataFrame con datos de modelado
        parameters: Parámetros del modelo
        
    Returns:
        None (guarda resultados directamente en archivos)
    """
    # Crear directorios necesarios
    os.makedirs("data/06_models", exist_ok=True)
    os.makedirs("data/08_reporting", exist_ok=True)
    
    # Extraer parámetros
    target_column = parameters.get("split", {}).get("target_column", "risk_mm")
    test_size = parameters.get("split", {}).get("test_size", 0.2)
    random_state = parameters.get("split", {}).get("random_state", 42)
    fit_intercept = parameters.get("train", {}).get("fit_intercept", True)
    cv_folds = parameters.get("evaluate", {}).get("cv_folds", 5)
    
    # Registrar información básica
    logger.info(f"Datos de modelado: {len(modeling_data_regression)} filas y {modeling_data_regression.columns.size} columnas")
    
    # Verificar si la columna objetivo existe
    if target_column not in modeling_data_regression.columns:
        available_columns = ", ".join(modeling_data_regression.columns)
        raise ValueError(f"La columna objetivo '{target_column}' no está en el DataFrame. Columnas disponibles: {available_columns}")
    
    # Separar características y objetivo
    X = modeling_data_regression.drop(columns=[target_column])
    y = modeling_data_regression[target_column]
    
    # Dividir en entrenamiento y prueba
    logger.info(f"Dividiendo datos en conjuntos de entrenamiento ({1-test_size:.0%}) y prueba ({test_size:.0%})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    logger.info(f"Conjunto de prueba: {X_test.shape[0]} muestras")
    
    # Crear y entrenar el modelo
    model = LinearRegression(fit_intercept=fit_intercept)
    logger.info("Entrenando modelo de regresión lineal...")
    model.fit(X_train, y_train)
    
    # Mostrar coeficientes
    coefficients = dict(zip(X_train.columns, model.coef_))
    intercept = model.intercept_
    
    logger.info(f"Intercepto: {intercept:.4f}")
    logger.info("Coeficientes principales:")
    for feature, coef in sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
        logger.info(f"  {feature}: {coef:.4f}")
    
    # Guardar modelo usando joblib
    model_path = "data/06_models/linear_regression_model.joblib"
    dump(model, model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # ----- FASE DE EVALUACIÓN -----
    logger.info("Evaluando modelo en conjunto de prueba...")
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calcular validación cruzada
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Registrar métricas
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"CV RMSE ({cv_folds} folds): {cv_rmse:.4f}")
    
    # Crear diccionario de métricas
    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "cv_rmse": float(cv_rmse)
    }
    
    # Añadir datos de predicciones para visualización
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    
    # ----- FASE DE VISUALIZACIÓN -----
    
    # Crear un resumen en formato JSON
    metrics_summary = {k: round(v, 4) for k, v in metrics.items()}
    
    # Extraer los datos
    actual = results_df['actual']
    predicted = results_df['predicted']
    
    # Crear el gráfico de dispersión de valores reales vs predicciones
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5)
    
    # Añadir línea de perfecta predicción
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Añadir títulos y etiquetas
    plt.title(f'Regresión Lineal: Valores Reales vs Predicciones\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)
    
    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig('data/08_reporting/linear_regression_scatter.png')
    plt.close()
    
    # Crear histograma de residuos
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    
    # Añadir títulos y etiquetas
    plt.title('Distribución de Residuos - Regresión Lineal')
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    
    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig('data/08_reporting/linear_regression_residuals.png')
    plt.close()
    
    # Guardar resumen de métricas en un archivo JSON
    metrics_file = 'data/08_reporting/linear_regression_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    logger.info(f"Visualizaciones y métricas guardadas en data/08_reporting/")
    logger.info(f"Resumen de métricas guardado en {metrics_file}")
    
    # Guardar una copia de las características más importantes
    importance_df = pd.DataFrame({
        'feature': coefficients.keys(),
        'importance': np.abs(list(coefficients.values()))
    }).sort_values('importance', ascending=False).head(20)
    
    importance_file = 'data/08_reporting/linear_regression_feature_importance.csv'
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"Importancia de características guardada en {importance_file}")
    
    return None