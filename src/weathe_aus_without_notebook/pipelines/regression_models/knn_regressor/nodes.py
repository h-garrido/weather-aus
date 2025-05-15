"""
Nodos para el pipeline de KNN Regressor.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import json
import os

logger = logging.getLogger(__name__)

def train_knn_regressor(modeling_data_regression: pd.DataFrame, parameters: Dict) -> None:
    """
    Entrena un modelo de KNN Regressor.
    
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
    n_neighbors = parameters.get("train", {}).get("n_neighbors", 5)
    weights = parameters.get("train", {}).get("weights", "uniform")
    algorithm = parameters.get("train", {}).get("algorithm", "auto")
    leaf_size = parameters.get("train", {}).get("leaf_size", 30)
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
    
    # Escalar las características (muy importante para KNN)
    logger.info("Escalando características...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar el scaler para uso futuro
    scaler_path = "data/06_models/knn_regressor_scaler.joblib"
    dump(scaler, scaler_path)
    logger.info(f"Scaler guardado en {scaler_path}")
    
    # Crear y entrenar el modelo
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=-1  # Usar todos los núcleos disponibles
    )
    
    logger.info(f"Entrenando modelo KNN Regressor con {n_neighbors} vecinos...")
    model.fit(X_train_scaled, y_train)
    
    # Guardar modelo usando joblib
    model_path = "data/06_models/knn_regressor_model.joblib"
    dump(model, model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    # ----- FASE DE EVALUACIÓN -----
    logger.info("Evaluando modelo en conjunto de prueba...")
    
    # Predecir en el conjunto de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calcular validación cruzada (con pipeline de escalado)
    from sklearn.pipeline import Pipeline
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=-1
        ))
    ])
    
    cv_scores = cross_val_score(knn_pipeline, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
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
        "cv_rmse": float(cv_rmse),
        "model_params": {
            "n_neighbors": n_neighbors,
            "weights": weights,
            "algorithm": algorithm,
            "leaf_size": leaf_size
        }
    }
    
    # Añadir datos de predicciones para visualización
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred
    })
    
    # ----- FASE DE VISUALIZACIÓN -----
    
    # Crear un resumen en formato JSON
    metrics_summary = {k: v if isinstance(v, dict) else round(v, 4) for k, v in metrics.items()}
    
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
    plt.title(f'KNN Regressor: Valores Reales vs Predicciones\nR² = {metrics["r2"]:.4f}, RMSE = {metrics["rmse"]:.4f}')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.grid(True)
    
    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig('data/08_reporting/knn_regressor_scatter.png')
    plt.close()
    
    # Crear histograma de residuos
    residuals = actual - predicted
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    
    # Añadir títulos y etiquetas
    plt.title('Distribución de Residuos - KNN Regressor')
    plt.xlabel('Residuos')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    
    # Guardar el gráfico
    plt.tight_layout()
    plt.savefig('data/08_reporting/knn_regressor_residuals.png')
    plt.close()
    
    # Analizar el efecto del número de vecinos (opcional)
    k_values = range(1, 21)
    train_scores = []
    test_scores = []
    
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        train_scores.append(knn.score(X_train_scaled, y_train))
        test_scores.append(knn.score(X_test_scaled, y_test))
    
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, train_scores, 'o-', label='Entrenamiento')
    plt.plot(k_values, test_scores, 'o-', label='Prueba')
    plt.xlabel('Número de vecinos (k)')
    plt.ylabel('R² Score')
    plt.title('Efecto del número de vecinos en el rendimiento del modelo')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/08_reporting/knn_regressor_k_analysis.png')
    plt.close()
    
    # Guardar resumen de métricas en un archivo JSON
    metrics_file = 'data/08_reporting/knn_regressor_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    logger.info(f"Visualizaciones y métricas guardadas en data/08_reporting/")
    logger.info(f"Resumen de métricas guardado en {metrics_file}")
    
    return None