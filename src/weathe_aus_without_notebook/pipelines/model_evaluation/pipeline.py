from kedro.pipeline import Pipeline, node
from .nodes import (
    evaluate_regression_models,
    evaluate_all_classification_models,
    generate_evaluation_visualizations,
    create_final_evaluation_report
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the model evaluation pipeline.
    
    This pipeline evaluates both regression and classification models,
    generates visualizations, and creates a comprehensive evaluation report.
    """
    return Pipeline(
        [
            # Evaluate regression models
            node(
                func=evaluate_regression_models,
                inputs=[
                    "gaussian_nb_regressor_metrics",
                    "gradient_boosting_regressor_metrics", 
                    "ridge_regressor_metrics",
                    "lasso_regressor_metrics",
                    "svr_metrics"
                ],
                outputs=["regression_evaluation_df", "regression_evaluation_summary"],
                name="evaluate_regression_models"
            ),
            
            # Evaluate classification models
            node(
                func=evaluate_all_classification_models,
                inputs={
                    "classification_data": "classification_data",
                    "classification_model": "classification_model",
                    "bayes_model": "bayes_model"
                },
                outputs="classification_evaluation_summary",
                name="evaluate_classification_models"
            ),
            
            # Generate visualizations
            node(
                func=generate_evaluation_visualizations,
                inputs=[
                    "regression_evaluation_df",
                    "classification_evaluation_summary"
                ],
                outputs="evaluation_visualization_paths",
                name="generate_evaluation_visualizations"
            ),
            
            # Create final report
            node(
                func=create_final_evaluation_report,
                inputs=[
                    "regression_evaluation_summary",
                    "classification_evaluation_summary",
                    "evaluation_visualization_paths"
                ],
                outputs="final_evaluation_report",
                name="create_final_evaluation_report"
            )
        ]
    )