"""
Model Training Pipeline
Defines the structure and flow of the model training pipeline.
"""
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    train_classification_model,
    train_regression_model, 
    evaluate_models_on_test
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the model training pipeline.
    
    Returns:
        Pipeline: Model training pipeline with classification and regression models
    """
    return pipeline([
        # Train classification model (Rain prediction)
        node(
            func=train_classification_model,
            inputs=[
                "X_train_classification", 
                "y_train_classification",
                "parameters"
            ],
            outputs="model_training_classification_artifacts", 
            name="train_classification_model_node",
        ),
        
        # Train regression model (Temperature prediction)  
        node(
            func=train_regression_model,
            inputs=[
                "X_train_regression",
                "y_train_regression", 
                "parameters"
            ],
            outputs="model_training_regression_artifacts",     
            name="train_regression_model_node",
        ),
        
        # Evaluate both models on test data
         node(
            func=evaluate_models_on_test,
            inputs=[
                "model_training_classification_artifacts",     
                "model_training_regression_artifacts",         
                "X_test_classification",
                "X_test_regression", 
                "y_test_classification",
                "y_test_regression"
            ],
            outputs="model_training_evaluation_results",       
            name="evaluate_models_node",
        ),
    ])