"""
Este módulo contiene la definición del pipeline para el modelo Lasso Regression.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    split_data,
    train_lasso_regressor,
    evaluate_lasso_regressor,
    predict_lasso_regressor
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para el modelo de Lasso Regression.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para Lasso Regression
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data_regression", "params:lasso_regressor"],
                outputs=["X_train_lasso", "X_test_lasso", "y_train_lasso", "y_test_lasso"],
                name="split_data_lasso_node",
            ),
            node(
                func=train_lasso_regressor,
                inputs=["X_train_lasso", "y_train_lasso", "params:lasso_regressor"],
                outputs="lasso_regressor_model",
                name="train_lasso_regressor_node",
            ),
            node(
                func=evaluate_lasso_regressor,
                inputs=[
                    "lasso_regressor_model", 
                    "X_train_lasso", 
                    "X_test_lasso", 
                    "y_train_lasso", 
                    "y_test_lasso", 
                    "params:lasso_regressor"
                ],
                outputs="lasso_regressor_metrics",
                name="evaluate_lasso_regressor_node",
            ),
            node(
                func=predict_lasso_regressor,
                inputs=["lasso_regressor_model", "X_test_lasso"],
                outputs="lasso_regressor_predictions",
                name="predict_lasso_regressor_node",
            ),
        ]
    )