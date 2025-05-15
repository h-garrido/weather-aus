"""
Pipeline para Random Forest Regressor.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_random_forest

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_random_forest,
                inputs=["modeling_data_regression", "params:random_forest"],
                outputs=None,
                name="train_random_forest_node",
            ),
        ]
    )