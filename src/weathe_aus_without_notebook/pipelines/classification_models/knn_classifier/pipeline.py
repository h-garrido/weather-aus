from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_knn_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de KNN Classification.
    
    Returns:
        Pipeline de KNN
    """
    return pipeline([
        node(
            func=train_knn_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:knn_params"
            ],
            outputs="knn_model",
            name="train_knn_node"
        )
    ])