"""
Este módulo contiene nodos para el modelo Gaussian Naive Bayes adaptado para regresión.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import logging

logger = logging.getLogger(__name__)

class GaussianNBRegressor(BaseEstimator, RegressorMixin):
    """
    Un modelo Gaussian Naive Bayes adaptado para problemas de regresión.
    
    Esta implementación discretiza la variable objetivo continua en bins
    y luego utiliza un clasificador GaussianNB estándar. Para la predicción,
    devuelve el valor esperado basado en las probabilidades de clase.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Inicializa el Gaussian Naive Bayes Regressor.
        
        Args:
            n_bins: Número de bins para discretizar la variable objetivo
        """
        self.n_bins = n_bins
        self.model = GaussianNB()
        self.bins = None
        self.bin_centers = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNBRegressor':
        """
        Ajusta el Gaussian Naive Bayes Regressor.
        
        Args:
            X: Características de entrenamiento
            y: Valores objetivo de entrenamiento
            
        Returns:
            self: El regresor ajustado
        """
        # Discretiza la variable objetivo
        self.bins = np.linspace(np.min(y), np.max(y), self.n_bins + 1)
        y_binned = np.digitize(y, self.bins[:-1])
        
        # Calcula el centro de cada bin para la predicción
        self.bin_centers = np.array([(self.bins[i] + self.bins[i+1]) / 2 
                                     for i in range(len(self.bins) - 1)])
        
        # Ajusta el clasificador GaussianNB
        self.model.fit(X, y_binned)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predice usando el Gaussian Naive Bayes Regressor.
        
        Args:
            X: Características para predecir
            
        Returns:
            y_pred: Valores objetivo predichos
        """
        # Obtiene las probabilidades de clase
        probs = self.model.predict_proba(X)
        
        # Calcula el valor esperado usando las probabilidades de clase
        y_pred = np.sum(probs * self.bin_centers.reshape(1, -1), axis=1)
        return y_pred


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


def train_gaussian_nb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any]
) -> GaussianNBRegressor:
    """
    Entrena un regresor Gaussian Naive Bayes.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Valores objetivo de entrenamiento
        parameters: Parámetros para el modelo
        
    Returns:
        model: Modelo GaussianNBRegressor entrenado
    """
    logger.info("Entrenando Gaussian Naive Bayes Regressor...")
    train_params = parameters["train"]
    
    model = GaussianNBRegressor(n_bins=train_params["n_bins"])
    model.fit(X_train.values, y_train.values)
    
    logger.info("Gaussian Naive Bayes Regressor entrenado exitosamente.")
    return model


def evaluate_gaussian_nb_regressor(
    model: GaussianNBRegressor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, float]:
    """
    Evalúa el regresor Gaussian Naive Bayes.
    
    Args:
        model: Modelo GaussianNBRegressor entrenado
        X_train: Características de entrenamiento
        X_test: Características de prueba
        y_train: Valores objetivo de entrenamiento
        y_test: Valores objetivo de prueba
        parameters: Parámetros para la evaluación
        
    Returns:
        metrics: Diccionario que contiene métricas de evaluación
    """
    logger.info("Evaluando Gaussian Naive Bayes Regressor...")
    evaluate_params = parameters["evaluate"]
    cv_folds = evaluate_params["cv_folds"]
    
    # Realiza predicciones en los datos de prueba
    y_pred = model.predict(X_test.values)
    
    # Calcula métricas
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Puntajes de validación cruzada
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=parameters["split"]["random_state"])
    cv_scores = cross_val_score(model, X_train.values, y_train.values, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_rmse": cv_rmse
    }
    
    logger.info(f"Métricas de prueba: MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
    logger.info(f"RMSE de validación cruzada: {cv_rmse:.4f}")
    
    return metrics


def predict_gaussian_nb_regressor(
    model: GaussianNBRegressor,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Realiza predicciones con el regresor Gaussian Naive Bayes.
    
    Args:
        model: Modelo GaussianNBRegressor entrenado
        X_test: Nuevas características para predecir
        
    Returns:
        predictions_df: DataFrame con las predicciones y características
    """
    logger.info("Realizando predicciones con Gaussian Naive Bayes Regressor...")
    y_pred = model.predict(X_test.values)
    
    # Crea un DataFrame con las predicciones y las características originales
    predictions_df = X_test.copy()
    predictions_df['prediction'] = y_pred
    
    return predictions_df