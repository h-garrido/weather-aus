"""
Pipeline para KNN Regressor.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_knn_regressor

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_knn_regressor,
                inputs=["modeling_data_regression", "params:knn_regressor"],
                outputs=None,
                name="train_knn_regressor_node",
            ),
        ]
    )