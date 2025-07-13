from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_bayes_classification

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea pipeline para Naive Bayes Classifier.
    """
    return pipeline([
        node(
            func=train_bayes_classification,
            inputs=[
                "transformed_data",
                "selected_features_classification", 
                "params:bayes_classification_params"
            ],
            outputs="bayes_classification_model",
            name="train_bayes_classification_node",
        ),
    ])