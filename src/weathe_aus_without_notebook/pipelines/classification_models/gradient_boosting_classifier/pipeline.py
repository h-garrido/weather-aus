from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_gradient_boosting_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Gradient Boosting Classification.
    
    Returns:
        Pipeline de Gradient Boosting
    """
    return pipeline([
        node(
            func=train_gradient_boosting_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:gradient_boosting_params"
            ],
            outputs="gradient_boosting_classification_model",
            name="train_gradient_boosting_node"
        )
    ])