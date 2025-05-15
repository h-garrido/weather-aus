"""
Este módulo contiene nodos para el modelo Support Vector Regression (SVR).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import logging

logger = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba, y escala las características.
    
    Args:
        data: Datos que contienen características y objetivo
        parameters: Parámetros para dividir los datos
        
    Returns:
        X_train_scaled: Características de entrenamiento escaladas
        X_test_scaled: Características de prueba escaladas
        y_train: Objetivo de entrenamiento
        y_test: Objetivo de prueba
        scaler: Objeto StandardScaler ajustado
        X_train: Características de entrenamiento originales (sin escalar)
        X_test: Características de prueba originales (sin escalar)
    """
    split_params = parameters["split"]
    target_column = split_params["target_column"]
    test_size = split_params["test_size"]
    random_state = split_params["random_state"]
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Guardar los datos originales no escalados
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    
    # Escalar las características (importante para SVR)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info(f"Data split: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    logger.info("Características escaladas con StandardScaler")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train_original, X_test_original


def train_svr(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> SVR:
    """
    Entrena un modelo Support Vector Regression.
    
    Args:
        X_train_scaled: Características de entrenamiento escaladas
        y_train: Valores objetivo de entrenamiento
        parameters: Parámetros para el modelo
        
    Returns:
        model: Modelo SVR entrenado
    """
    logger.info("Entrenando Support Vector Regression...")
    train_params = parameters["train"]
    
    model = SVR(
        kernel=train_params["kernel"],
        C=train_params["C"],
        epsilon=train_params["epsilon"],
        gamma=train_params["gamma"],
        tol=train_params["tol"],
        cache_size=train_params["cache_size"],
        max_iter=train_params["max_iter"]
    )
    
    model.fit(X_train_scaled, y_train)
    
    logger.info(f"Support Vector Regression entrenado exitosamente con kernel={train_params['kernel']}")
    
    # Información adicional sobre el modelo
    n_support_vectors = model.n_support_.sum()
    logger.info(f"Número de vectores de soporte utilizados: {n_support_vectors}")
    
    return model


def evaluate_svr(
    model: SVR,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa el modelo Support Vector Regression.
    
    Args:
        model: Modelo SVR entrenado
        X_train_scaled: Características de entrenamiento escaladas
        X_test_scaled: Características de prueba escaladas
        y_train: Valores objetivo de entrenamiento
        y_test: Valores objetivo de prueba
        parameters: Parámetros para la evaluación
        
    Returns:
        metrics: Diccionario que contiene métricas de evaluación
    """
    logger.info("Evaluando Support Vector Regression...")
    evaluate_params = parameters["evaluate"]
    cv_folds = evaluate_params["cv_folds"]
    
    # Realiza predicciones en los datos de prueba
    y_pred = model.predict(X_test_scaled)
    
    # Calcula métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Puntajes de validación cruzada
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=parameters["split"]["random_state"])
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Información sobre el modelo
    n_support_vectors = model.n_support_.sum()
    train_params = parameters["train"]
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_rmse": cv_rmse,
        "n_support_vectors": int(n_support_vectors),
        "kernel": train_params["kernel"],
        "C": train_params["C"],
        "epsilon": train_params["epsilon"],
        "gamma": train_params["gamma"]
    }
    
    logger.info(f"Métricas de prueba: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    logger.info(f"RMSE de validación cruzada: {cv_rmse:.4f}")
    logger.info(f"Número de vectores de soporte: {n_support_vectors}")
    logger.info(f"Hiperparámetros: kernel={train_params['kernel']}, C={train_params['C']}, epsilon={train_params['epsilon']}, gamma={train_params['gamma']}")
    
    return metrics


def predict_svr(
    model: SVR,
    X_test_scaled: pd.DataFrame,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Realiza predicciones con el modelo Support Vector Regression.
    
    Args:
        model: Modelo SVR entrenado
        X_test_scaled: Características de prueba escaladas
        X_test: Características de prueba originales (para conservar los datos originales)
        
    Returns:
        predictions_df: DataFrame con las predicciones y características originales
    """
    logger.info("Realizando predicciones con Support Vector Regression...")
    y_pred = model.predict(X_test_scaled)
    
    # Crea un DataFrame con las predicciones y las características originales
    predictions_df = X_test.copy()
    predictions_df['prediction'] = y_pred
    
    # Calcular estadísticas de predicción
    min_pred = y_pred.min()
    max_pred = y_pred.max()
    mean_pred = y_pred.mean()
    std_pred = y_pred.std()
    
    logger.info(f"Predicciones completadas para {len(X_test)} instancias.")
    logger.info(f"Estadísticas de predicción: min={min_pred:.4f}, max={max_pred:.4f}, media={mean_pred:.4f}, desv={std_pred:.4f}")
    
    return predictions_df


def create_svr_pipeline_artifacts(
    model: SVR,
    scaler: StandardScaler,
    X_train: pd.DataFrame
) -> Dict[str, Any]:
    """
    Crea artefactos adicionales para el pipeline SVR.
    
    Args:
        model: Modelo SVR entrenado
        scaler: Scaler usado para normalizar los datos
        X_train: Características de entrenamiento originales
        
    Returns:
        artifacts: Diccionario con artefactos adicionales
    """
    logger.info("Creando artefactos adicionales para el pipeline SVR...")
    
    # Podríamos calcular importancia de características para SVR mediante
    # permutation importance, pero es computacionalmente costoso
    
    artifacts = {
        "model_info": {
            "n_features": X_train.shape[1],
            "n_support_vectors": int(model.n_support_.sum()),
            "feature_names": X_train.columns.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist()
        }
    }
    
    logger.info(f"Artefactos creados: información del modelo con {artifacts['model_info']['n_features']} características")
    return artifacts