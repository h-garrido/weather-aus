"""
Este módulo contiene la definición del pipeline para el modelo Ridge Regression.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    split_data,
    train_ridge_regressor,
    evaluate_ridge_regressor,
    predict_ridge_regressor
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para el modelo de Ridge Regression.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para Ridge Regression
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data_regression", "params:ridge_regressor"],
                outputs=["X_train_ridge", "X_test_ridge", "y_train_ridge", "y_test_ridge"],
                name="split_data_ridge_node",
            ),
            node(
                func=train_ridge_regressor,
                inputs=["X_train_ridge", "y_train_ridge", "params:ridge_regressor"],
                outputs="ridge_regressor_model",
                name="train_ridge_regressor_node",
            ),
            node(
                func=evaluate_ridge_regressor,
                inputs=[
                    "ridge_regressor_model", 
                    "X_train_ridge", 
                    "X_test_ridge", 
                    "y_train_ridge", 
                    "y_test_ridge", 
                    "params:ridge_regressor"
                ],
                outputs="ridge_regressor_metrics",
                name="evaluate_ridge_regressor_node",
            ),
            node(
                func=predict_ridge_regressor,
                inputs=["ridge_regressor_model", "X_test_ridge"],
                outputs="ridge_regressor_predictions",
                name="predict_ridge_regressor_node",
            ),
        ]
    )