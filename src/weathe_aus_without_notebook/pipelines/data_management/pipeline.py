"""
Data Management Pipeline Definition
==================================
Pipeline que unifica todo el procesamiento de datos para MLOps automatizado.
Reemplaza los múltiples pipelines dispersos con un flujo unificado.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    create_data_snapshot,
    create_unified_temporal_splits,
    apply_unified_feature_engineering,
    validate_data_quality
)


def create_data_management_pipeline(**kwargs) -> Pipeline:
    """
    Creates the unified data management pipeline.
    
    Este pipeline:
    1. 📸 Crea snapshot versionado de los datos
    2. 🔪 Genera splits temporales unificados 
    3. ⚙️ Aplica feature engineering consistente
    4. 🔍 Valida calidad de datos
    
    Returns:
        Pipeline: Unified data management pipeline
    """
    return pipeline(
        [
            # ==========================================
            # 1. DATA VERSIONING & SNAPSHOT
            # ==========================================
            node(
                func=create_data_snapshot,
                inputs="weather_data_postgres",  # From your existing postgres pipeline
                outputs="versioned_data",
                name="create_data_snapshot_node",
                tags=["data_versioning", "snapshot"]
            ),
            
            # ==========================================
            # 2. UNIFIED TEMPORAL SPLITS  
            # ==========================================
            node(
                func=create_unified_temporal_splits,
                inputs=["versioned_data", "parameters"],
                outputs=[
                    "X_train_unified",
                    "X_test_unified", 
                    "y_classification_train_unified",
                    "y_classification_test_unified",
                    "y_regression_train_unified", 
                    "y_regression_test_unified"
                ],
                name="create_unified_splits_node",
                tags=["data_splitting", "temporal"]
            ),
            
            # ==========================================
            # 3. UNIFIED FEATURE ENGINEERING
            # ==========================================
            node(
                func=apply_unified_feature_engineering,
                inputs=[
                    "X_train_unified",
                    "X_test_unified",
                    "y_classification_train_unified", 
                    "y_regression_train_unified",
                    "parameters"
                ],
                outputs=[
                    "X_train_processed",
                    "X_test_processed", 
                    "feature_metadata"
                ],
                name="apply_feature_engineering_node",
                tags=["feature_engineering", "scaling", "selection"]
            ),
            
            # ==========================================
            # 4. DATA QUALITY VALIDATION
            # ==========================================
            node(
                func=validate_data_quality,
                inputs=[
                    "X_train_processed",
                    "X_test_processed",
                    "y_classification_train_unified",
                    "y_regression_train_unified", 
                    "parameters"
                ],
                outputs="data_quality_report",
                name="validate_data_quality_node",
                tags=["data_quality", "validation"]
            )
        ],
        #namespace="data_management",
        tags=["unified_data_processing", "mlops_foundation"]
    )


def create_legacy_compatibility_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline de compatibilidad para mantener tus pipelines existentes funcionando.
    Mapea los outputs unificados a los nombres que tus modelos actuales esperan.
    """
    return pipeline(
        [
            # Mapping para clasificación
            node(
                func=lambda x: x,  # Identity function
                inputs="X_train_processed",
                outputs="X_train_classification",
                name="map_classification_train_features",
                tags=["compatibility", "classification"]
            ),
            
            node(
                func=lambda x: x,
                inputs="X_test_processed", 
                outputs="X_test_classification",
                name="map_classification_test_features",
                tags=["compatibility", "classification"]
            ),
            
            node(
                func=lambda x: x,
                inputs="y_classification_train_unified",
                outputs="y_train_classification", 
                name="map_classification_train_target",
                tags=["compatibility", "classification"]
            ),
            
            node(
                func=lambda x: x,
                inputs="y_classification_test_unified",
                outputs="y_test_classification",
                name="map_classification_test_target", 
                tags=["compatibility", "classification"]
            ),
            
            # Mapping para regresión
            node(
                func=lambda x: x,
                inputs="X_train_processed",
                outputs="X_train_regression",
                name="map_regression_train_features",
                tags=["compatibility", "regression"]
            ),
            
            node(
                func=lambda x: x,
                inputs="X_test_processed",
                outputs="X_test_regression", 
                name="map_regression_test_features",
                tags=["compatibility", "regression"]
            ),
            
            node(
                func=lambda x: x,
                inputs="y_regression_train_unified",
                outputs="y_train_regression",
                name="map_regression_train_target",
                tags=["compatibility", "regression"]
            ),
            
            node(
                func=lambda x: x,
                inputs="y_regression_test_unified", 
                outputs="y_test_regression",
                name="map_regression_test_target",
                tags=["compatibility", "regression"]
            )
        ],
        #namespace="legacy_compatibility",
        tags=["backwards_compatibility"]
    )


def create_full_data_management_pipeline(**kwargs) -> Pipeline:
    """
    Combines both the new unified pipeline and legacy compatibility.
    """
    return (
        create_data_management_pipeline(**kwargs) + 
        create_legacy_compatibility_pipeline(**kwargs)
    )