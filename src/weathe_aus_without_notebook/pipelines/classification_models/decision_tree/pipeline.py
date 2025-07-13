from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_decision_tree_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de Decision Tree Classification.
    
    Returns:
        Pipeline de Decision Tree
    """
    return pipeline([
        node(
            func=train_decision_tree_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:decision_tree_params"
            ],
            outputs="decision_tree_model",
            name="train_decision_tree_node"
        )
    ])