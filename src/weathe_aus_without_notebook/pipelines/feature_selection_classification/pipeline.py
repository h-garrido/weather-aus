from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_classification_data,
    correlation_analysis_classification,
    mutual_information_analysis,
    chi_square_analysis,
    feature_importance_analysis,
    recursive_feature_elimination,
    select_best_features_classification,
    create_feature_selection_report_classification
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Preparar datos para clasificación
        node(
            func=prepare_classification_data,
            inputs="transformed_data",
            outputs="classification_data",
            name="prepare_classification_data_node",
        ),
        
        # Análisis de correlación para clasificación
        node(
            func=correlation_analysis_classification,
            inputs="classification_data",
            outputs="correlation_results_classification",
            name="correlation_analysis_classification_node",
        ),
        
        # Análisis de información mutua
        node(
            func=mutual_information_analysis,
            inputs="classification_data",
            outputs="mutual_info_results",
            name="mutual_information_analysis_node",
        ),
        
        # Análisis Chi-square para variables categóricas
        node(
            func=chi_square_analysis,
            inputs="classification_data",
            outputs="chi_square_results",
            name="chi_square_analysis_node",
        ),
        
        # Feature importance usando Random Forest
        node(
            func=feature_importance_analysis,
            inputs="classification_data",
            outputs="feature_importance_results",
            name="feature_importance_analysis_node",
        ),
        
        # Recursive Feature Elimination
        node(
            func=recursive_feature_elimination,
            inputs=["classification_data", "params:rfe_params"],
            outputs="rfe_results",
            name="recursive_feature_elimination_node",
        ),
        
        # Selección final de mejores features
        node(
            func=select_best_features_classification,
            inputs=[
                "correlation_results_classification",
                "mutual_info_results", 
                "chi_square_results",
                "feature_importance_results",
                "rfe_results",
                "params:feature_selection_params"
            ],
            outputs="selected_features_classification",
            name="select_best_features_classification_node",
        ),
        
        # Crear reporte de feature selection
        node(
            func=create_feature_selection_report_classification,
            inputs=[
                "correlation_results_classification",
                "mutual_info_results",
                "chi_square_results", 
                "feature_importance_results",
                "rfe_results",
                "selected_features_classification"
            ],
            outputs="feature_selection_report_classification",
            name="create_feature_selection_report_classification_node",
        ),
    ])