"""
Model Registry Pipeline - SIMPLIFICADO
======================================

Pipeline para gestión de modelos usando tus datasets existentes.
NO necesita prepare_test_data_from_transformed.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    initialize_model_registry,
    register_models_batch_direct,
    compare_all_models,
    generate_model_leaderboard,
    create_registry_summary_report,
    update_best_models,
    archive_old_models,
    validate_model_metrics,
    track_model_lineage,
    create_model_comparison_report,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline principal del Model Registry.
    USA TUS DATASETS EXISTENTES - No necesita funciones adicionales.
    
    Returns:
        Pipeline: Model registry management pipeline
    """
    return pipeline(
        [
            # ==========================================
            # 1. INITIALIZE MODEL REGISTRY
            # ==========================================
            node(
                func=initialize_model_registry,
                inputs=["model_registry_config", "parameters"],
                outputs="model_registry_instance",
                name="initialize_model_registry_node",
                tags=["model_registry", "initialization", "setup"]
            ),
            
            # ==========================================
            # 2. REGISTER ALL TRAINED MODELS
            # ==========================================
            
            # ==========================================
            # 3. VALIDATE MODEL METRICS
            # ==========================================
            node(
                func=validate_model_metrics,
                inputs=[
                    "model_registry_instance",
                    "auto_registered_models",
                    "parameters"
                ],
                outputs="model_metrics_validation",
                name="validate_model_metrics_node",
                tags=["model_registry", "validation", "metrics"]
            ),
            
            # ==========================================
            # 4. TRACK MODEL LINEAGE
            # ==========================================
            node(
                func=track_model_lineage,
                inputs=[
                    "model_registry_instance",
                    "auto_registered_models",
                    "parameters"
                ],
                outputs="model_lineage_data",
                name="track_model_lineage_node",
                tags=["model_registry", "lineage", "tracking"]
            ),
            
            # ==========================================
            # 5. COMPARE ALL MODELS
            # ==========================================
            node(
                func=compare_all_models,
                inputs=[
                    "model_registry_instance",
                    "auto_registered_models",
                    "X_test_unified",  
                    "y_classification_test_unified",
                    "parameters"
                ],
                outputs="model_comparison_results",
                name="compare_all_models_node",
                tags=["model_registry", "comparison", "evaluation"]
            ),
            
            # ==========================================
            # 6. GENERATE MODEL LEADERBOARD
            # ==========================================
            node(
                func=generate_model_leaderboard,
                inputs=[
                    "model_registry_instance",
                    "model_comparison_results",
                    "parameters"
                ],
                outputs="model_leaderboard_data",
                name="generate_model_leaderboard_node",
                tags=["model_registry", "leaderboard", "ranking"]
            ),
            
            # ==========================================
            # 7. UPDATE BEST MODELS
            # ==========================================
            node(
                func=update_best_models,
                inputs=[
                    "model_registry_instance",
                    "model_leaderboard_data",
                    "parameters"
                ],
                outputs=["best_regression_model", "best_classification_model", "best_models_metadata"],
                name="update_best_models_node", 
                tags=["model_registry", "best_models", "update"]
            ),
            
            # ==========================================
            # 8. CREATE COMPREHENSIVE REPORTS
            # ==========================================
            node(
                func=create_registry_summary_report,
                inputs=[
                    "model_registry_instance",
                    "model_leaderboard_data",
                    "model_comparison_results",
                    "model_lineage_data",
                    "parameters"
                ],
                outputs="model_registry_summary_report",
                name="create_registry_summary_report_node",
                tags=["model_registry", "reporting", "summary"]
            ),
            
            node(
                func=create_model_comparison_report,
                inputs=[
                    "model_comparison_results",
                    "model_leaderboard_data",
                    "parameters"
                ],
                outputs="model_comparison_report",
                name="create_model_comparison_report_node",
                tags=["model_registry", "reporting", "comparison"]
            ),
            
            # ==========================================
            # 9. ARCHIVE OLD MODELS (OPTIONAL)
            # ==========================================
            node(
                func=archive_old_models,
                inputs=[
                    "model_registry_instance",
                    "parameters"
                ],
                outputs="archived_models_log",
                name="archive_old_models_node",
                tags=["model_registry", "archival", "cleanup"]
            )
        ],
        tags=["model_registry", "management", "mlops"]
    )


def create_model_registration_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline ligero solo para registro de modelos.
    SOLUCIÓN: No pasar registry instance, solo la config.
    """
    return pipeline(
        [
            node(
                func=register_models_batch_direct,
                inputs=[
                    "model_registry_config",
                    "parameters",
                    "X_test_unified",
                    "y_classification_test_unified",
                    # SOLO los modelos que SÍ existen en tu catalog:
                    "gaussian_nb_regressor_model",
                    "lasso_regressor_model", 
                    "ridge_regressor_model",
                    "svr_model",
                    "logistic_regression_model",
                    "random_forest_classification_model",
                    "decision_tree_model",
                    "svm_classification_model",
                    "bayes_classification_model", 
                    "knn_model",
                    "gradient_boosting_classification_model"
                ],
                outputs="auto_registered_models",
                name="register_models_direct_node",
                tags=["model_registry", "registration"]
            )
        ],
        tags=["registration", "lightweight"]
    )


def create_model_comparison_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline enfocado en comparación de modelos.
    """
    return pipeline(
        [
            node(
                func=initialize_model_registry,
                inputs=["model_registry_config", "parameters"],
                outputs="model_registry_instance",
                name="init_registry_for_comparison",
                tags=["model_registry", "comparison"]
            ),
            
            node(
                func=compare_all_models,
                inputs=[
                    "model_registry_instance",
                    "auto_registered_models",  # Debe venir de pipeline anterior
                    "X_test_unified",
                    "y_classification_test_unified",
                    "parameters"
                ],
                outputs="comparison_results",
                name="compare_models_node",
                tags=["model_registry", "comparison"]
            ),
            
            node(
                func=generate_model_leaderboard,
                inputs=[
                    "model_registry_instance", 
                    "comparison_results",
                    "parameters"
                ],
                outputs="leaderboard_results",
                name="generate_leaderboard_node",
                tags=["model_registry", "leaderboard"]
            ),
            
            node(
                func=create_model_comparison_report,
                inputs=[
                    "comparison_results",
                    "leaderboard_results", 
                    "parameters"
                ],
                outputs="comparison_report",
                name="create_comparison_report_node",
                tags=["model_registry", "reporting"]
            )
        ],
        tags=["comparison", "evaluation"]
    )


def create_model_monitoring_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline para monitoreo de modelos.
    """
    return pipeline(
        [
            node(
                func=initialize_model_registry,
                inputs=["model_registry_config", "parameters"],
                outputs="model_registry_instance",
                name="init_registry_for_monitoring",
                tags=["model_registry", "monitoring"]
            ),
            
            node(
                func=validate_model_metrics,
                inputs=[
                    "model_registry_instance",
                    "auto_registered_models",  # Debe existir de pipeline anterior
                    "parameters"
                ],
                outputs="metrics_validation_report",
                name="validate_existing_metrics_node",
                tags=["model_registry", "validation", "monitoring"]
            ),
            
            node(
                func=archive_old_models,
                inputs=[
                    "model_registry_instance",
                    "parameters"
                ],
                outputs="archival_report",
                name="archive_outdated_models_node",
                tags=["model_registry", "archival", "monitoring"]
            ),
            
            node(
                func=create_registry_summary_report,
                inputs=[
                    "model_registry_instance",
                    "model_leaderboard_data",  
                    "model_comparison_results",  
                    "model_lineage_data",  
                    "parameters"
                ],
                outputs="monitoring_summary_report",
                name="create_monitoring_report_node",
                tags=["model_registry", "monitoring", "reporting"]
            )
        ],
        tags=["monitoring", "maintenance"]
    )