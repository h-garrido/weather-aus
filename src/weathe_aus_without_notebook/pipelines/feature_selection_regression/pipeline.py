from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_feature_interactions_regression, select_regression_features

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_feature_interactions_regression,
                inputs="transformed_data",
                outputs="data_with_interactions_regression",
                name="create_feature_interactions_regression_node",
            ),
            node(
                func=select_regression_features,
                inputs="data_with_interactions_regression",
                outputs="modeling_data_regression",
                name="select_regression_features_node",
            ),
        ]
    )