from kedro.pipeline import Pipeline, pipeline, node
from .nodes import train_svm_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de SVM Classification.
    
    Returns:
        Pipeline de SVM
    """
    return pipeline([
        node(
            func=train_svm_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:svm_params"
            ],
            outputs="svm_classification_model",
            name="train_svm_node"
        )
    ])