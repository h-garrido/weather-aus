"""
Este módulo contiene la definición del pipeline para el modelo Gradient Boosting Regressor.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    split_data,
    train_gradient_boosting_regressor,
    evaluate_gradient_boosting_regressor,
    predict_gradient_boosting_regressor
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para el modelo de regresión Gradient Boosting.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para regresión con Gradient Boosting
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data_regression", "params:gradient_boosting_regressor"],
                outputs=["X_train_gradient_boosting", "X_test_gradient_boosting", "y_train_gradient_boosting", "y_test_gradient_boosting"],
                name="split_data_gradient_boosting_node",
            ),
            node(
                func=train_gradient_boosting_regressor,
                inputs=["X_train_gradient_boosting", "y_train_gradient_boosting", "params:gradient_boosting_regressor"],
                outputs="gradient_boosting_regressor_model",
                name="train_gradient_boosting_regressor_node",
            ),
            node(
                func=evaluate_gradient_boosting_regressor,
                inputs=[
                    "gradient_boosting_regressor_model", 
                    "X_train_gradient_boosting", 
                    "X_test_gradient_boosting", 
                    "y_train_gradient_boosting", 
                    "y_test_gradient_boosting", 
                    "params:gradient_boosting_regressor"
                ],
                outputs="gradient_boosting_regressor_metrics",
                name="evaluate_gradient_boosting_regressor_node",
            ),
            node(
                func=predict_gradient_boosting_regressor,
                inputs=["gradient_boosting_regressor_model", "X_test_gradient_boosting"],
                outputs="gradient_boosting_regressor_predictions",
                name="predict_gradient_boosting_regressor_node",
            ),
        ]
    )