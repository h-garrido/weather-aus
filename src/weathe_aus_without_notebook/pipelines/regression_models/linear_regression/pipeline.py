"""
Pipeline para regresión lineal.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_linear_regression

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_linear_regression,
                inputs=["modeling_data_regression", "params:linear_regression"],
                outputs=None,
                name="train_linear_regression_node",
            ),
        ]
    )