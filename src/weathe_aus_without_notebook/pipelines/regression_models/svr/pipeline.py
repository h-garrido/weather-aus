"""
Este módulo contiene la definición del pipeline para el modelo Support Vector Regression.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    split_data,
    train_svr,
    evaluate_svr,
    predict_svr,
    create_svr_pipeline_artifacts
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para el modelo Support Vector Regression.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para Support Vector Regression
    """
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["modeling_data_regression", "params:svr_regressor"],
                outputs=["X_train_svr_scaled", "X_test_svr_scaled", "y_train_svr", "y_test_svr", "svr_scaler", "X_train_svr", "X_test_svr"],
                name="split_data_svr_node",
            ),
            node(
                func=train_svr,
                inputs=["X_train_svr_scaled", "y_train_svr", "params:svr_regressor"],
                outputs="svr_model",
                name="train_svr_node",
            ),
            node(
                func=evaluate_svr,
                inputs=[
                    "svr_model", 
                    "X_train_svr_scaled", 
                    "X_test_svr_scaled", 
                    "y_train_svr", 
                    "y_test_svr", 
                    "params:svr_regressor"
                ],
                outputs="svr_metrics",
                name="evaluate_svr_node",
            ),
            node(
                func=predict_svr,
                inputs=["svr_model", "X_test_svr_scaled", "X_test_svr"],
                outputs="svr_predictions",
                name="predict_svr_node",
            ),
            node(
                func=create_svr_pipeline_artifacts,
                inputs=["svr_model", "svr_scaler", "X_train_svr"],
                outputs="svr_artifacts",
                name="create_svr_artifacts_node",
            ),
        ]
    )