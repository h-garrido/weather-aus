from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_random_forest_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Random Forest Classification.
    
    Returns:
        Pipeline de Random Forest
    """
    return pipeline([
        node(
            func=train_random_forest_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:random_forest_params"
            ],
            outputs="random_forest_classification_model",
            name="classification_random_forest"
        )
    ])