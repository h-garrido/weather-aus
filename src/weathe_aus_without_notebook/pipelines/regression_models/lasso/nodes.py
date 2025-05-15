"""
Este módulo contiene nodos para el modelo Lasso Regression.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import logging

logger = logging.getLogger(__name__)


def split_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Args:
        data: Datos que contienen características y objetivo
        parameters: Parámetros para dividir los datos
        
    Returns:
        X_train: Características de entrenamiento
        X_test: Características de prueba
        y_train: Objetivo de entrenamiento
        y_test: Objetivo de prueba
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
    
    logger.info(f"Data split: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_lasso_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> Lasso:
    """
    Entrena un modelo Lasso Regression.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Valores objetivo de entrenamiento
        parameters: Parámetros para el modelo
        
    Returns:
        model: Modelo Lasso entrenado
    """
    logger.info("Entrenando Lasso Regression...")
    train_params = parameters["train"]
    
    model = Lasso(
        alpha=train_params["alpha"],
        fit_intercept=train_params["fit_intercept"],
        max_iter=train_params["max_iter"],
        tol=train_params["tol"],
        selection=train_params["selection"],
        random_state=parameters["split"]["random_state"]
    )
    
    model.fit(X_train, y_train)
    
    # Verificar convergencia
    if model.n_iter_ >= train_params["max_iter"]:
        logger.warning(f"¡El modelo Lasso no convergió! Número de iteraciones: {model.n_iter_}")
    else:
        logger.info(f"Lasso Regression convergió en {model.n_iter_} iteraciones.")
    
    # Contar características seleccionadas (coeficientes no nulos)
    n_features = X_train.shape[1]
    n_selected_features = np.sum(model.coef_ != 0)
    logger.info(f"Características seleccionadas: {n_selected_features}/{n_features} ({n_selected_features/n_features:.2%})")
    
    return model


def evaluate_lasso_regressor(
    model: Lasso,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evalúa el modelo Lasso Regression.
    
    Args:
        model: Modelo Lasso entrenado
        X_train: Características de entrenamiento
        X_test: Características de prueba
        y_train: Valores objetivo de entrenamiento
        y_test: Valores objetivo de prueba
        parameters: Parámetros para la evaluación
        
    Returns:
        metrics: Diccionario que contiene métricas de evaluación
    """
    logger.info("Evaluando Lasso Regression...")
    evaluate_params = parameters["evaluate"]
    cv_folds = evaluate_params["cv_folds"]
    
    # Realiza predicciones en los datos de prueba
    y_pred = model.predict(X_test)
    
    # Calcula métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Puntajes de validación cruzada
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=parameters["split"]["random_state"])
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # Coeficientes del modelo
    coefficients = dict(zip(X_train.columns, model.coef_))
    intercept = model.intercept_
    
    # Filtrar coeficientes no nulos
    non_zero_coefficients = {k: v for k, v in coefficients.items() if v != 0}
    n_selected_features = len(non_zero_coefficients)
    
    # Ordenar coeficientes por su valor absoluto
    sorted_coefficients = {k: v for k, v in sorted(
        non_zero_coefficients.items(), 
        key=lambda item: abs(item[1]), 
        reverse=True
    )}
    
    # Obtener los top N coeficientes no nulos
    top_n = min(10, len(sorted_coefficients))
    top_coefficients = dict(list(sorted_coefficients.items())[:top_n])
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_rmse": cv_rmse,
        "intercept": float(intercept),
        "n_selected_features": n_selected_features,
        "top_non_zero_coefficients": top_coefficients
    }
    
    logger.info(f"Métricas de prueba: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    logger.info(f"RMSE de validación cruzada: {cv_rmse:.4f}")
    logger.info(f"Intercepto: {intercept:.4f}")
    logger.info(f"Características seleccionadas: {n_selected_features}/{len(coefficients)}")
    logger.info(f"Top {top_n} coeficientes no nulos: {top_coefficients}")
    
    return metrics


def predict_lasso_regressor(
    model: Lasso,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Realiza predicciones con el modelo Lasso Regression.
    
    Args:
        model: Modelo Lasso entrenado
        X_test: Nuevas características para predecir
        
    Returns:
        predictions_df: DataFrame con las predicciones y características
    """
    logger.info("Realizando predicciones con Lasso Regression...")
    y_pred = model.predict(X_test)
    
    # Crea un DataFrame con las predicciones y las características originales
    predictions_df = X_test.copy()
    predictions_df['prediction'] = y_pred
    
    # Identificar características con coeficientes no nulos en el modelo
    selected_features = [feature for feature, coef in zip(X_test.columns, model.coef_) if coef != 0]
    
    logger.info(f"Predicciones completadas para {len(X_test)} instancias.")
    logger.info(f"Modelo utiliza {len(selected_features)} características: {selected_features[:5]}...")
    
    return predictions_df