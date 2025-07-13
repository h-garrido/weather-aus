from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    evaluate_single_classifier,
    compare_classification_models,
    generate_classification_visualizations,
    generate_classification_report,
    create_classification_models_dict,
    prepare_test_data_from_transformed
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the classification evaluation pipeline.
    
    This pipeline:
    1. 🔍 Evaluates individual classification models
    2. 📊 Compares multiple models
    3. 📈 Generates visualizations
    4. 📝 Creates comprehensive evaluation report
    5. 📝 Creates comprehensive evaluation report
    
    Returns:
        Pipeline: Classification evaluation pipeline
    """
    return pipeline(
        [
            # ==========================================
            # 0. CREATE MODELS DICTIONARY 
            # ==========================================
            # Nuevo nodo para preparar datos
            node(
                func=prepare_test_data_from_transformed,
                inputs=["transformed_data", "parameters"],
                outputs=["X_test_for_eval", "y_test_for_eval"],
                name="prepare_test_data_node"
            ),

            node(
                func=create_classification_models_dict,
                inputs=[
                    "logistic_regression_model",
                    "random_forest_classification_model",
                    "svm_classification_model",
                    "decision_tree_model",
                    "bayes_classification_model",
                    "knn_model",
                    "gradient_boosting_classification_model"
                ],
                outputs="classification_models_dict",
                name="create_classification_models_dict_node",
                tags=["classification", "consolidation", "models"]
            ),
            # ==========================================
            # 1. COMPARE ALL CLASSIFICATION MODELS
            # ==========================================
            node(
                func=compare_classification_models,
                inputs=[
                    "classification_models_dict",  # All trained classification models
                    #"X_test_classification",       # Test features (from unified data management)
                    #"y_test_classification",       # Test targets (from unified data management)
                    "X_test_for_eval",
                    "y_test_for_eval",
                    "parameters"
                ],
                outputs="classification_comparison_report",
                name="compare_classification_models_node",
                tags=["classification", "evaluation", "comparison"]
            ),
            
            # ==========================================
            # 2. GENERATE VISUALIZATIONS
            # ==========================================
            node(
                func=generate_classification_visualizations,
                inputs=[
                    "classification_comparison_report",
                    "parameters"
                ],
                outputs="classification_visualizations",
                name="generate_classification_visualizations_node",
                tags=["classification", "visualization", "reporting"]
            ),
            
            # ==========================================
            # 3. GENERATE COMPREHENSIVE REPORT
            # ==========================================
            node(
                func=generate_classification_report,
                inputs=[
                    "classification_comparison_report",
                    "classification_visualizations",
                    "parameters"
                ],
                outputs="classification_evaluation_report",
                name="generate_classification_report_node",
                tags=["classification", "reporting", "final"]
            )
        ],
        #namespace="classification_evaluation",
        tags=["evaluation", "classification", "metrics"]
    )


def create_individual_model_evaluation_pipeline(**kwargs) -> Pipeline:
    """
    Creates pipeline for evaluating individual classification models.
    Useful for testing single models or debugging.
    """
    return pipeline(
        [
            # Evaluate individual models - you can create nodes for specific models
            # Example for logistic regression:
            node(
                func=evaluate_single_classifier,
                inputs=[
                    "logistic_regression_model",
                    "X_test_classification",
                    "y_test_classification",
                    "params:logistic_regression_name",
                    "parameters"
                ],
                outputs="logistic_regression_evaluation",
                name="evaluate_logistic_regression_node",
                tags=["classification", "evaluation", "logistic_regression"]
            ),
            
            # se pueden añadir neuvos nodos en caso de
            # node(
            #     func=evaluate_single_classifier,
            #     inputs=[
            #         "random_forest_classification_model",
            #         "X_test_classification", 
            #         "y_test_classification",
            #         "params:random_forest_name",
            #         "parameters"
            #     ],
            #     outputs="random_forest_classification_evaluation",
            #     name="evaluate_random_forest_classification_node",
            #     tags=["classification", "evaluation", "random_forest"]
            # ),
        ],
        namespace="individual_classification_evaluation",
        tags=["evaluation", "classification", "individual"]
    )