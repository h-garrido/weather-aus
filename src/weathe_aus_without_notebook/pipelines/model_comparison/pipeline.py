from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    load_all_regression_metrics,
    compare_regression_models,
    create_model_comparison_report,
    generate_comparison_visualizations
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_all_regression_metrics,
                inputs=[
                    "linear_regression_metrics",
                    "random_forest_metrics", 
                    "knn_regressor_metrics",
                    "lasso_regressor_metrics",
                    "ridge_regressor_metrics",
                    "svr_metrics",
                    "gaussian_nb_regressor_metrics",
                    "gradient_boosting_regressor_metrics"
                ],
                outputs="regression_metrics_df",
                name="load_regression_metrics_node"
            ),
            node(
                func=compare_regression_models,
                inputs="regression_metrics_df",
                outputs="regression_comparison_results",
                name="compare_regression_models_node"
            ),
            node(
                func=create_model_comparison_report,
                inputs=[
                    "regression_comparison_results"
                ],
                outputs="model_comparison_report",
                name="create_comparison_report_node"
            ),
            node(
                func=generate_comparison_visualizations,
                inputs=[
                    "regression_metrics_df",
                    "regression_comparison_results"
                ],
                outputs="comparison_visualizations",
                name="generate_visualizations_node"
            )
        ]
    )