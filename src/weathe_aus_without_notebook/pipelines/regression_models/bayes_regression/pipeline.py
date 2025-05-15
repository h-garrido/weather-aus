"""
Este módulo contiene la definición del pipeline para el modelo Gaussian Naive Bayes adaptado para regresión.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    split_data,
    train_gaussian_nb_regressor,
    evaluate_gaussian_nb_regressor,
    predict_gaussian_nb_regressor
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para el modelo de regresión Gaussian Naive Bayes.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para regresión con Gaussian Naive Bayes
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data_regression", "params:gaussian_nb_regressor"],
                outputs=["X_train_gaussian_nb", "X_test_gaussian_nb", "y_train_gaussian_nb", "y_test_gaussian_nb"],
                name="split_data_gaussian_nb_node",
            ),
            node(
                func=train_gaussian_nb_regressor,
                inputs=["X_train_gaussian_nb", "y_train_gaussian_nb", "params:gaussian_nb_regressor"],
                outputs="gaussian_nb_regressor_model",
                name="train_gaussian_nb_regressor_node",
            ),
            node(
                func=evaluate_gaussian_nb_regressor,
                inputs=[
                    "gaussian_nb_regressor_model", 
                    "X_train_gaussian_nb", 
                    "X_test_gaussian_nb", 
                    "y_train_gaussian_nb", 
                    "y_test_gaussian_nb", 
                    "params:gaussian_nb_regressor"
                ],
                outputs="gaussian_nb_regressor_metrics",
                name="evaluate_gaussian_nb_regressor_node",
            ),
            node(
                func=predict_gaussian_nb_regressor,
                inputs=["gaussian_nb_regressor_model", "X_test_gaussian_nb"],
                outputs="gaussian_nb_regressor_predictions",
                name="predict_gaussian_nb_regressor_node",
            ),
        ]
    )