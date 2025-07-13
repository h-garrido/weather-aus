from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_logistic_regression_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Logistic Regression Classification.
    
    Returns:
        Pipeline de Logistic Regression
    """
    return pipeline([
        node(
            func=train_logistic_regression_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:logistic_regression_params"
            ],
            outputs="logistic_regression_model",
            name="train_logistic_regression_node"
        )
    ])