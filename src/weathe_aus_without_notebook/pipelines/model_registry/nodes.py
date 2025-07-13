"""
Model Registry Nodes
===================

Functions for managing model registration, comparison, and tracking
in the Weather Australia MLOps project.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss,
    r2_score, mean_squared_error, mean_absolute_error
)

from ...model_registry.registry import ModelRegistry
from ...model_registry.models import ModelMetadata, ModelComparison
from ...model_registry.exceptions import ModelRegistryError

logger = logging.getLogger(__name__)


def initialize_model_registry(config: Dict[str, Any], parameters: Dict[str, Any]) -> ModelRegistry:
    """
    Initialize the Model Registry instance.
    
    Args:
        config: Model registry configuration
        parameters: Kedro parameters
        
    Returns:
        ModelRegistry: Initialized registry instance
    """
    try:
        # Get database credentials from parameters
        db_config = parameters.get("database", {})
        connection_params = {
            "host": db_config.get("host", "postgres"),
            "port": db_config.get("port", 5432),
            "database": db_config.get("database", "kedro_db"),
            "user": db_config.get("user", "kedro"),
            "password": db_config.get("password", "kedro")
        }
        
        # Get registry configuration
        registry_config = config.get("model_registry", {})
        base_path = registry_config.get("base_path", "data/09_model_registry")
        
        # Initialize registry
        registry = ModelRegistry(
            connection_params=connection_params,
            base_path=base_path
        )
        
        logger.info("Model Registry initialized successfully")
        return registry
        
    except Exception as e:
        logger.error(f"Failed to initialize Model Registry: {str(e)}")
        raise ModelRegistryError(f"Registry initialization failed: {str(e)}")


def register_models_batch_direct(
    # Registry config (from params:model_registry)
    model_registry_config: Dict[str, Any],
    parameters: Dict[str, Any],
    # Test data (required)
    X_test: Any,
    y_test: Any,
    # Regression models (all optional)
    gaussian_nb_regressor_model: Any = None,
    lasso_regressor_model: Any = None,
    ridge_regressor_model: Any = None,
    svr_model: Any = None,
    # Classification models (all optional)
    logistic_regression_model: Any = None,
    random_forest_classification_model: Any = None,
    decision_tree_model: Any = None,
    svm_classification_model: Any = None,
    bayes_classification_model: Any = None,
    knn_model: Any = None,
    gradient_boosting_classification_model: Any = None
) -> Dict[str, Any]:
    """
    Register multiple models in batch - DIRECT VERSION.
    Creates its own registry instance to avoid serialization issues.
    
    Args:
        model_registry_config: Your existing model_registry configuration
        parameters: Kedro parameters
        X_test: Test features
        y_test: Test targets  
        Various model parameters (all optional)
        
    Returns:
        Dict with registration results
    """
    # Create registry instance directly in this function
    # Extract config from your model_registry structure
    base_path = model_registry_config.get("base_path", "data/09_model_registry")
    
    # PostgreSQL connection params (NO database_url)
    connection_params = {
        "user": "kedro",     # From your docker-compose
        "password": "kedro", # From your docker-compose
        "host": "postgres",  # Service name in docker-compose
        "port": 5432,
        "database": "kedro_db",  # From your docker-compose
    }
    
    registry = ModelRegistry(connection_params=connection_params,
        base_path=base_path)
    
    registered_models = {
        "regression": [],
        "classification": [],
        "failed": [],
        "summary": {}
    }
    
    # ✅ AGREGAR ESTA SECCIÓN AQUÍ (después de línea 125)
    # CHECK FOR EXISTING MODELS
    try:
        existing_models = registry.list_models()
        existing_names = set(existing_models['model_name'].tolist()) if not existing_models.empty else set()
        logger.info(f"Found {len(existing_names)} existing models: {existing_names}")
    except Exception as e:
        logger.warning(f"Could not check existing models: {e}")
        existing_names = set()


    # Define model mappings - ONLY models that exist in your catalog
    regression_models = {
        "gaussian_nb_regressor": gaussian_nb_regressor_model,
        "lasso_regressor": lasso_regressor_model,
        "ridge_regressor": ridge_regressor_model,
        "svr": svr_model
    }
    
    classification_models = {
        "logistic_regression": logistic_regression_model,
        "random_forest_classification": random_forest_classification_model,
        "decision_tree": decision_tree_model,
        "svm_classification": svm_classification_model,
        "bayes_classification": bayes_classification_model,
        "knn_classification": knn_model,
        "gradient_boosting_classification": gradient_boosting_classification_model
    }
    
    # Debug: Let's see what we actually have
    logger.info("=== DEBUGGING MODEL TYPES ===")
    all_models = {**regression_models, **classification_models}
    for name, model in all_models.items():
        logger.info(f"{name}: type={type(model)}, hasattr_predict={hasattr(model, 'predict') if model else False}")
        if isinstance(model, dict):
            logger.info(f"  Dict keys: {list(model.keys())}")
            # Extraer el modelo real del diccionario
            if 'model' in model:
                actual_model = model['model']
                logger.info(f"  Found 'model' key with type: {type(actual_model)}")
                # Reemplazar con el modelo real
                if name in regression_models:
                    regression_models[name] = actual_model
                else:
                    classification_models[name] = actual_model
    logger.info("=== END DEBUGGING ===")

    def align_features(X, model, model_name):
        
        """Alinear features entre datos y modelo"""
        try:
            # ✅ CASO ESPECIAL: GaussianNBRegressor personalizado
            if model_name == "gaussian_nb_regressor":
                if hasattr(model, 'model_') and hasattr(model.model_, 'n_features_in_'):
                    expected_features = model.model_.n_features_in_
                elif hasattr(model, 'n_features_in_'):
                    expected_features = model.n_features_in_
                else:
                    # Si no podemos determinar, usar las primeras 17 features
                    expected_features = 17
            # ✅ CASO NORMAL: sklearn models
            elif hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
            else:
                return X
            
            if X.shape[1] != expected_features:
                logger.warning(f"Feature mismatch for {model_name}: Expected {expected_features}, got {X.shape[1]}")
                # Tomar solo las primeras N features que el modelo espera
                X_aligned = X.iloc[:, :expected_features] if hasattr(X, 'iloc') else X[:, :expected_features]
                logger.info(f"✅ Aligned features for {model_name}: {X.shape[1]} -> {X_aligned.shape[1]}")
                return X_aligned
            return X
            
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                if X.shape[1] != expected_features:
                    logger.warning(f"Feature mismatch for {model_name}: Expected {expected_features}, got {X.shape[1]}")
                    # Tomar solo las primeras N features que el modelo espera
                    X_aligned = X.iloc[:, :expected_features] if hasattr(X, 'iloc') else X[:, :expected_features]
                    logger.info(f"✅ Aligned features for {model_name}: {X.shape[1]} -> {X_aligned.shape[1]}")
                    return X_aligned
            return X
        except Exception as e:
            logger.warning(f"Could not align features for {model_name}: {e}")
            # ✅ FALLBACK: Si hay error, intentar con las primeras 17 features para regresión o 13 para clasificación
            try:
                fallback_features = 17 if "regressor" in model_name else 13
                X_fallback = X.iloc[:, :fallback_features] if hasattr(X, 'iloc') else X[:, :fallback_features]
                logger.info(f"✅ Using fallback alignment for {model_name}: {X.shape[1]} -> {fallback_features}")
                return X_fallback
            except:
                return X
    
    # Register regression models
    for model_name, model in regression_models.items():
        if model is not None:

            if model_name in existing_names:
                logger.info(f"⏭️ Skipping {model_name}: Already exists in registry")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": "Model already exists (skipped)"
                })
                continue

            try:
                # Verificar si el modelo es válido
                if not hasattr(model, 'predict'):
                    logger.warning(f"Skipping {model_name}: Not a valid model object (type: {type(model)})")
                    continue
                
                # Calculate metrics if test data is available and compatible
                metrics = {}
                if X_test is not None and y_test is not None:
                    try:
                        # ✅ ALINEAR FEATURES
                        X_test_aligned = align_features(X_test, model, model_name)
                        
                        y_pred = model.predict(X_test_aligned)
                        metrics = {
                            "r2_score": r2_score(y_test, y_pred),
                            "mean_squared_error": mean_squared_error(y_test, y_pred),
                            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
                            "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred))
                        }
                        logger.info(f"✅ Metrics calculated for {model_name}")
                    except Exception as metric_error:
                        logger.warning(f"⚠️ Could not calculate metrics for {model_name}: {metric_error}")
                        metrics = {"error": str(metric_error)}
                
                # Get hyperparameters
                hyperparams = _extract_hyperparameters(model)
                
                # Register model (even without metrics)
                model_id = registry.register_model(
                    model=model,
                    model_name=model_name,
                    model_type="regression",
                    algorithm=model_name,
                    metrics=metrics,
                    hyperparameters=hyperparams,
                    training_data=None,
                    test_data=None,
                    description=f"Auto-registered {model_name} regression model",
                    tags={
                        "auto_registered": True,
                        "model_type": "regression",
                        "algorithm": model_name,
                        "project": "weather_australia",
                        "status": "registered_without_metrics" if not metrics or "error" in metrics else "complete"
                    }
                )
                
                registered_models["regression"].append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
                
                logger.info(f"✅ Successfully registered regression model: {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register {model_name}: Model registration failed: {str(e)}")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": str(e)
                })
    
    # Register classification models  
    for model_name, model in classification_models.items():
        if model is not None:
            if model_name in existing_names:
                logger.info(f"⏭️ Skipping {model_name}: Already exists in registry")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": "Model already exists (skipped)"
                })
                continue
            try:
                # Verificar si el modelo es válido
                if not hasattr(model, 'predict'):
                    logger.warning(f"Skipping {model_name}: Not a valid model object (type: {type(model)})")
                    continue
                
                # Calculate metrics if test data is available and compatible
                metrics = {}
                if X_test is not None and y_test is not None:
                    try:
                        # ✅ ALINEAR FEATURES
                        X_test_aligned = align_features(X_test, model, model_name)
                        
                        y_pred = model.predict(X_test_aligned)
                        y_pred_proba = None
                        
                        # Try to get prediction probabilities
                        if hasattr(model, 'predict_proba'):
                            try:
                                y_pred_proba = model.predict_proba(X_test_aligned)[:, 1]
                            except:
                                pass
                        
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        }
                        
                        # Add ROC AUC if probabilities available
                        if y_pred_proba is not None:
                            try:
                                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
                            except:
                                pass
                        
                        logger.info(f"✅ Metrics calculated for {model_name}")
                    except Exception as metric_error:
                        logger.warning(f"⚠️ Could not calculate metrics for {model_name}: {metric_error}")
                        metrics = {"error": str(metric_error)}
                
                # Get hyperparameters
                hyperparams = _extract_hyperparameters(model)
                
                # Register model (even without metrics)
                model_id = registry.register_model(
                    model=model,
                    model_name=model_name,
                    model_type="classification",
                    algorithm=model_name,
                    metrics=metrics,
                    hyperparameters=hyperparams,
                    training_data=None,
                    test_data=None,
                    description=f"Auto-registered {model_name} classification model",
                    tags={
                        "auto_registered": True,
                        "model_type": "classification", 
                        "algorithm": model_name,
                        "project": "weather_australia",
                        "status": "registered_without_metrics" if not metrics or "error" in metrics else "complete"
                    }
                )
                
                registered_models["classification"].append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
                
                logger.info(f"✅ Successfully registered classification model: {model_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to register {model_name}: Model registration failed: {str(e)}")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": str(e)
                })
    
    # Create summary
    registered_models["summary"] = {
        "total_registered": len(registered_models["regression"]) + len(registered_models["classification"]),
        "regression_count": len(registered_models["regression"]),
        "classification_count": len(registered_models["classification"]),
        "failed_count": len(registered_models["failed"]),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Model registration completed: {registered_models['summary']}")
    
    return registered_models


def register_models_batch(
    registry: ModelRegistry,
    # Test data (required)
    X_test: Any,
    y_test: Any,
    parameters: Dict[str, Any],
    # Regression models (all optional)
    gaussian_nb_regressor_model: Any = None,
    lasso_regressor_model: Any = None,
    ridge_regressor_model: Any = None,
    svr_model: Any = None,
    # Classification models (all optional)
    logistic_regression_model: Any = None,
    random_forest_classification_model: Any = None,
    decision_tree_model: Any = None,
    svm_classification_model: Any = None,
    bayes_classification_model: Any = None,
    knn_model: Any = None,
    gradient_boosting_classification_model: Any = None
) -> Dict[str, Any]:
    """
    Register multiple models in batch - SIMPLIFIED VERSION.
    Only registers models that exist in your catalog.
    
    Args:
        registry: Model registry instance
        X_test: Test features
        y_test: Test targets  
        parameters: Kedro parameters
        Various model parameters (all optional)
        
    Returns:
        Dict with registration results
    """
    registered_models = {
        "regression": [],
        "classification": [],
        "failed": [],
        "summary": {}
    }
    
    # Define model mappings - ONLY models that exist in your catalog
    regression_models = {
        "gaussian_nb_regressor": gaussian_nb_regressor_model,
        "lasso_regressor": lasso_regressor_model,
        "ridge_regressor": ridge_regressor_model,
        "svr": svr_model
    }
    
    classification_models = {
        "logistic_regression": logistic_regression_model,
        "random_forest_classification": random_forest_classification_model,
        "decision_tree": decision_tree_model,
        "svm_classification": svm_classification_model,
        "bayes_classification": bayes_classification_model,
        "knn_classification": knn_model,
        "gradient_boosting_classification": gradient_boosting_classification_model
    }
    
    # Register regression models
    for model_name, model in regression_models.items():
        if model is not None:
            if model_name in existing_names:
                logger.info(f"⏭️ Skipping {model_name}: Already exists in registry")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": "Model already exists (skipped)"
                })
                continue
            try:
                # Calculate metrics if test data is available
                metrics = {}
                if X_test is not None and y_test is not None:
                    y_pred = model.predict(X_test)
                    metrics = {
                        "r2_score": r2_score(y_test, y_pred),
                        "mean_squared_error": mean_squared_error(y_test, y_pred),
                        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
                        "root_mean_squared_error": np.sqrt(mean_squared_error(y_test, y_pred))
                    }
                
                # Get hyperparameters
                hyperparams = _extract_hyperparameters(model)
                
                # Register model
                model_id = registry.register_model(
                    model=model,
                    model_name=model_name,
                    model_type="regression",
                    algorithm=model_name,
                    metrics=metrics,
                    hyperparameters=hyperparams,
                    training_data=X_test,  # Using test data as proxy
                    test_data=X_test,
                    description=f"Auto-registered {model_name} regression model",
                    tags={
                        "auto_registered": True,
                        "model_type": "regression",
                        "algorithm": model_name,
                        "project": "weather_australia"
                    }
                )
                
                registered_models["regression"].append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
                
                logger.info(f"Successfully registered regression model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to register {model_name}: {str(e)}")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": str(e)
                })
    
    # Register classification models  
    for model_name, model in classification_models.items():
        if model is not None:
            if model_name in existing_names:
                logger.info(f"⏭️ Skipping {model_name}: Already exists in registry")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": "Model already exists (skipped)"
                })
                continue
            try:
                # Calculate metrics if test data is available
                metrics = {}
                if X_test is not None and y_test is not None:
                    y_pred = model.predict(X_test)
                    y_pred_proba = None
                    
                    # Try to get prediction probabilities
                    if hasattr(model, 'predict_proba'):
                        try:
                            y_pred_proba = model.predict_proba(X_test)[:, 1]
                        except:
                            pass
                    
                    metrics = {
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                    
                    # Add ROC AUC if probabilities available
                    if y_pred_proba is not None:
                        try:
                            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
                        except:
                            pass
                
                # Get hyperparameters
                hyperparams = _extract_hyperparameters(model)
                
                # Register model
                model_id = registry.register_model(
                    model=model,
                    model_name=model_name,
                    model_type="classification",
                    algorithm=model_name,
                    metrics=metrics,
                    hyperparameters=hyperparams,
                    training_data=X_test,  # Using test data as proxy
                    test_data=X_test,
                    description=f"Auto-registered {model_name} classification model",
                    tags={
                        "auto_registered": True,
                        "model_type": "classification", 
                        "algorithm": model_name,
                        "project": "weather_australia"
                    }
                )
                
                registered_models["classification"].append({
                    "model_id": model_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
                
                logger.info(f"Successfully registered classification model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to register {model_name}: {str(e)}")
                registered_models["failed"].append({
                    "model_name": model_name,
                    "error": str(e)
                })
    
    # Create summary
    registered_models["summary"] = {
        "total_registered": len(registered_models["regression"]) + len(registered_models["classification"]),
        "regression_count": len(registered_models["regression"]),
        "classification_count": len(registered_models["classification"]),
        "failed_count": len(registered_models["failed"]),
        "registration_time": datetime.now().isoformat()
    }
    
    logger.info(f"Batch registration completed: {registered_models['summary']}")
    return registered_models


def _extract_hyperparameters(model: Any) -> Dict[str, Any]:
    """
    Extract hyperparameters from a model.
    
    Args:
        model: Trained model
        
    Returns:
        Dict with hyperparameters
    """
    hyperparams = {}
    
    # Try to get parameters using get_params() method (sklearn models)
    if hasattr(model, 'get_params'):
        try:
            hyperparams = model.get_params()
        except:
            pass
    
    # Clean up parameters (remove non-serializable objects)
    cleaned_params = {}
    for key, value in hyperparams.items():
        if value is None:
            continue
        try:
            # Test if value is JSON serializable
            json.dumps(value)
            cleaned_params[key] = value
        except (TypeError, ValueError):
            # Convert to string if not serializable
            cleaned_params[key] = str(value)
    
    return cleaned_params


def validate_model_metrics(
    registry: ModelRegistry,
    registered_models: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate model metrics against configured rules.
    
    Args:
        registry: Model registry instance
        registered_models: Results from model registration
        parameters: Kedro parameters
        
    Returns:
        Dict with validation results
    """
    validation_results = {
        "valid_models": [],
        "invalid_models": [],
        "warnings": [],
        "summary": {}
    }
    
    # Get validation rules from parameters
    metrics_config = parameters.get("metrics", {})
    regression_rules = metrics_config.get("regression", {}).get("validation_rules", {})
    classification_rules = metrics_config.get("classification", {}).get("validation_rules", {})
    
    # Validate regression models
    for model_info in registered_models.get("regression", []):
        model_name = model_info["model_name"]
        metrics = model_info["metrics"]
        
        validation_result = _validate_metrics_against_rules(metrics, regression_rules, model_name)
        
        if validation_result["is_valid"]:
            validation_results["valid_models"].append(validation_result)
        else:
            validation_results["invalid_models"].append(validation_result)
        
        validation_results["warnings"].extend(validation_result["warnings"])
    
    # Validate classification models
    for model_info in registered_models.get("classification", []):
        model_name = model_info["model_name"]
        metrics = model_info["metrics"]
        
        validation_result = _validate_metrics_against_rules(metrics, classification_rules, model_name)
        
        if validation_result["is_valid"]:
            validation_results["valid_models"].append(validation_result)
        else:
            validation_results["invalid_models"].append(validation_result)
        
        validation_results["warnings"].extend(validation_result["warnings"])
    
    # Create summary
    validation_results["summary"] = {
        "total_models": len(validation_results["valid_models"]) + len(validation_results["invalid_models"]),
        "valid_count": len(validation_results["valid_models"]),
        "invalid_count": len(validation_results["invalid_models"]),
        "warning_count": len(validation_results["warnings"]),
        "validation_time": datetime.now().isoformat()
    }
    
    logger.info(f"Metrics validation completed: {validation_results['summary']}")
    return validation_results


def _validate_metrics_against_rules(metrics: Dict[str, float], rules: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Validate metrics against validation rules.
    
    Args:
        metrics: Model metrics
        rules: Validation rules
        model_name: Name of the model
        
    Returns:
        Dict with validation result
    """
    result = {
        "model_name": model_name,
        "is_valid": True,
        "violations": [],
        "warnings": [],
        "metrics": metrics
    }
    
    for metric_name, metric_value in metrics.items():
        if metric_name in rules:
            rule = rules[metric_name]
            
            # Check minimum value
            if "min_value" in rule and metric_value < rule["min_value"]:
                result["is_valid"] = False
                result["violations"].append(f"{metric_name} ({metric_value}) below minimum ({rule['min_value']})")
            
            # Check maximum value
            if "max_value" in rule and metric_value > rule["max_value"]:
                result["is_valid"] = False
                result["violations"].append(f"{metric_name} ({metric_value}) above maximum ({rule['max_value']})")
            
            # Check warning threshold
            if "warning_threshold" in rule and metric_value < rule["warning_threshold"]:
                result["warnings"].append(f"{model_name}: {metric_name} ({metric_value}) below warning threshold ({rule['warning_threshold']})")
    
    return result


def track_model_lineage(
    registry: ModelRegistry,
    registered_models: Dict[str, Any],
    parameters: Dict[str, Any]
) -> pd.DataFrame:  # ← CAMBIO: Devuelve DataFrame en lugar de Dict
    """
    Track lineage information for registered models.
    FIXED: Returns DataFrame for SQLTableDataset compatibility.
    
    Args:
        registry: Model registry instance
        registered_models: Results from model registration
        parameters: Kedro parameters
        
    Returns:
        pd.DataFrame: Lineage tracking results as DataFrame
    """
    # ← NUEVO: Lista para crear DataFrame
    lineage_records = []
    
    # Get lineage configuration
    lineage_config = parameters.get("lineage_tracking", {})
    
    if not lineage_config.get("enabled", True):
        logger.info("Lineage tracking is disabled")
        # ← RETORNAR DATAFRAME VACÍO
        return pd.DataFrame({
            'model_name': [],
            'model_type': [],
            'algorithm': [],
            'pipeline_name': [],
            'node_name': [],
            'input_datasets': [],
            'git_commit_hash': [],
            'kedro_version': [],
            'environment_info': [],
            'tracked_at': []
        })
    
    # Track lineage for all registered models
    all_models = registered_models.get("regression", []) + registered_models.get("classification", [])
    
    tracked_count = 0
    failed_count = 0
    
    for model_info in all_models:
        try:
            model_name = model_info["model_name"]
            
            # Determine model type
            model_type = "regression" if model_info in registered_models.get("regression", []) else "classification"
            
            # ← CREAR REGISTRO DE LINEAGE COMO DICT PARA DATAFRAME
            lineage_record = {
                'model_name': model_name,
                'model_type': model_type,
                'algorithm': model_name,  # Usually algorithm == model_name in your case
                'pipeline_name': "model_training",
                'node_name': f"train_{model_name}_node",
                'input_datasets': "X_train_unified,y_train_unified",  # String format for SQL
                'git_commit_hash': _get_git_commit_hash() if lineage_config.get("track_git_info") else None,
                'kedro_version': _get_kedro_version() if lineage_config.get("track_pipeline_info") else None,
                'environment_info': str(_get_environment_info()) if lineage_config.get("track_environment_info") else None,
                'tracked_at': datetime.now().isoformat()
            }
            
            lineage_records.append(lineage_record)
            tracked_count += 1
            
        except Exception as e:
            logger.error(f"Failed to track lineage for {model_info.get('model_name', 'unknown')}: {str(e)}")
            failed_count += 1
    
    # ← CREAR DATAFRAME
    lineage_df = pd.DataFrame(lineage_records)
    
    # Log summary (pero no lo devolvemos)
    summary = {
        "total_models": len(all_models),
        "tracked_count": tracked_count,
        "failed_count": failed_count,
        "tracking_time": datetime.now().isoformat()
    }
    
    logger.info(f"Lineage tracking completed: {summary}")
    
    # ← RETORNAR DATAFRAME
    return lineage_df
    
    # Create summary
    lineage_results["summary"] = {
        "total_models": len(all_models),
        "tracked_count": len(lineage_results["tracked_models"]),
        "failed_count": len(lineage_results["failed_tracking"]),
        "tracking_time": datetime.now().isoformat()
    }
    
    logger.info(f"Lineage tracking completed: {lineage_results['summary']}")
    return lineage_results


def _get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def _get_kedro_version() -> Optional[str]:
    """Get Kedro version."""
    try:
        import kedro
        return kedro.__version__
    except:
        return None


def _get_environment_info() -> Dict[str, str]:
    """Get environment information."""
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0]
    }


def compare_all_models(
    registry: ModelRegistry,
    registered_models: Dict[str, Any],
    X_test: Any,
    y_test: Any,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare all registered models.
    
    Args:
        registry: Model registry instance
        registered_models: Results from model registration
        X_test: Test features
        y_test: Test targets
        parameters: Kedro parameters
        
    Returns:
        Dict with comparison results
    """
    comparison_results = {
        "regression_comparison": {},
        "classification_comparison": {},
        "cross_comparison": {},
        "summary": {}
    }
    
    try:
        # Compare regression models
        regression_models = registered_models.get("regression", [])
        if len(regression_models) > 1:
            comparison_results["regression_comparison"] = _compare_models_by_type(
                regression_models, "regression", parameters
            )
        
        # Compare classification models
        classification_models = registered_models.get("classification", [])
        if len(classification_models) > 1:
            comparison_results["classification_comparison"] = _compare_models_by_type(
                classification_models, "classification", parameters
            )
        
        # Create summary
        comparison_results["summary"] = {
            "regression_models_compared": len(regression_models),
            "classification_models_compared": len(classification_models),
            "comparison_time": datetime.now().isoformat(),
            "comparison_id": str(uuid.uuid4())
        }
        
        logger.info(f"Model comparison completed: {comparison_results['summary']}")
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        comparison_results["error"] = str(e)
    
    return comparison_results


def _compare_models_by_type(models: List[Dict], model_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare models of the same type.
    
    Args:
        models: List of model information
        model_type: Type of models ('regression' or 'classification')
        parameters: Kedro parameters
        
    Returns:
        Dict with comparison results for this type
    """
    # Get primary metric for comparison
    metrics_config = parameters.get("metrics", {})
    primary_metric = metrics_config.get(model_type, {}).get("primary_metric")
    
    if not primary_metric:
        primary_metric = "r2_score" if model_type == "regression" else "accuracy"
    
    # Sort models by primary metric
    sorted_models = sorted(
        models,
        key=lambda x: x["metrics"].get(primary_metric, -float('inf')),
        reverse=True
    )
    
    # Create comparison result
    comparison = {
        "primary_metric": primary_metric,
        "model_ranking": [],
        "best_model": None,
        "performance_gap": None
    }
    
    for i, model in enumerate(sorted_models):
        ranking_info = {
            "rank": i + 1,
            "model_name": model["model_name"],
            "model_id": model["model_id"],
            "primary_metric_value": model["metrics"].get(primary_metric),
            "all_metrics": model["metrics"]
        }
        comparison["model_ranking"].append(ranking_info)
    
    # Set best model
    if sorted_models:
        comparison["best_model"] = sorted_models[0]
        
        # Calculate performance gap
        if len(sorted_models) > 1:
            best_score = sorted_models[0]["metrics"].get(primary_metric, 0)
            second_score = sorted_models[1]["metrics"].get(primary_metric, 0)
            comparison["performance_gap"] = best_score - second_score
    
    return comparison


def generate_model_leaderboard(
    registry: ModelRegistry,
    comparison_results: Dict[str, Any],
    parameters: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate model leaderboard from comparison results.
    
    Args:
        registry: Model registry instance
        comparison_results: Results from model comparison
        parameters: Kedro parameters
        
    Returns:
        DataFrame with model leaderboard
    """
    try:
        # Get leaderboard from database
        leaderboard_df = registry.get_model_leaderboard("regression", top_n=20)
        
        # Add classification models
        classification_leaderboard = registry.get_model_leaderboard("classification", top_n=20)
        
        # Combine leaderboards
        combined_leaderboard = pd.concat([leaderboard_df, classification_leaderboard], ignore_index=True)
        
        # Add additional information from comparison results
        combined_leaderboard["last_updated"] = datetime.now().isoformat()
        
        logger.info(f"Generated leaderboard with {len(combined_leaderboard)} models")
        return combined_leaderboard
        
    except Exception as e:
        logger.error(f"Failed to generate leaderboard: {str(e)}")
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=["model_type", "algorithm", "model_name", "version", "primary_metric", "rank"])


def update_best_models(
    registry: ModelRegistry,
    leaderboard_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Update best model references.
    
    Args:
        registry: Model registry instance
        leaderboard_data: Model leaderboard data
        parameters: Kedro parameters
        
    Returns:
        Tuple of (best_regression_model, best_classification_model, metadata)
    """
    best_models_metadata = {
        "best_regression": None,
        "best_classification": None,
        "updated_at": datetime.now().isoformat()
    }
    
    best_regression_model = None
    best_classification_model = None
    
    try:
        # Get best regression model
        regression_models = leaderboard_data[leaderboard_data["model_type"] == "regression"]
        if not regression_models.empty:
            best_regression_row = regression_models.iloc[0]
            best_regression_model, metadata = registry.get_model(
                best_regression_row["model_name"],
                best_regression_row["version"]
            )
            best_models_metadata["best_regression"] = metadata.to_dict()
            logger.info(f"Best regression model: {best_regression_row['model_name']} v{best_regression_row['version']}")
        
        # Get best classification model
        classification_models = leaderboard_data[leaderboard_data["model_type"] == "classification"]
        if not classification_models.empty:
            best_classification_row = classification_models.iloc[0]
            best_classification_model, metadata = registry.get_model(
                best_classification_row["model_name"],
                best_classification_row["version"]
            )
            best_models_metadata["best_classification"] = metadata.to_dict()
            logger.info(f"Best classification model: {best_classification_row['model_name']} v{best_classification_row['version']}")
        
    except Exception as e:
        logger.error(f"Failed to update best models: {str(e)}")
        best_models_metadata["error"] = str(e)
    
    return best_regression_model, best_classification_model, best_models_metadata


def create_registry_summary_report(
    registry: ModelRegistry,
    leaderboard_data: pd.DataFrame,
    comparison_results: Dict[str, Any],
    lineage_data: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comprehensive registry summary report.
    
    Args:
        registry: Model registry instance
        leaderboard_data: Model leaderboard data
        comparison_results: Model comparison results
        lineage_data: Model lineage data
        parameters: Kedro parameters
        
    Returns:
        Dict with comprehensive summary report
    """
    try:
        # Get all models from registry
        all_models = registry.list_models()
        
        # Create summary statistics
        summary_stats = {
            "total_models": len(all_models),
            "regression_models": len(all_models[all_models["model_type"] == "regression"]),
            "classification_models": len(all_models[all_models["model_type"] == "classification"]),
            "active_models": len(all_models[all_models["status"] == "active"]),
            "archived_models": len(all_models[all_models["status"] == "archived"]),
        }
        
        # Algorithm distribution
        algorithm_distribution = all_models["algorithm"].value_counts().to_dict()
        
        # Performance summary
        performance_summary = {
            "best_regression_r2": leaderboard_data[leaderboard_data["model_type"] == "regression"]["primary_metric"].max() if not leaderboard_data.empty else None,
            "best_classification_accuracy": leaderboard_data[leaderboard_data["model_type"] == "classification"]["primary_metric"].max() if not leaderboard_data.empty else None,
        }
        
        # Create comprehensive report
        report = {
            "summary_statistics": summary_stats,
            "algorithm_distribution": algorithm_distribution,
            "performance_summary": performance_summary,
            "leaderboard_preview": leaderboard_data.head(10).to_dict("records") if not leaderboard_data.empty else [],
            "comparison_summary": comparison_results.get("summary", {}),
            "lineage_summary": lineage_data.get("summary", {}),
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0.0",
                "total_pages": 1
            }
        }
        
        logger.info("Registry summary report created successfully")
        return report
        
    except Exception as e:
        logger.error(f"Failed to create summary report: {str(e)}")
        return {"error": str(e), "generated_at": datetime.now().isoformat()}


def create_model_comparison_report(
    comparison_results: Dict[str, Any],
    leaderboard_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create detailed model comparison report.
    
    Args:
        comparison_results: Model comparison results
        leaderboard_data: Model leaderboard data
        parameters: Kedro parameters
        
    Returns:
        Dict with detailed comparison report
    """
    try:
        report = {
            "executive_summary": {
                "total_models_compared": comparison_results.get("summary", {}).get("regression_models_compared", 0) + 
                                       comparison_results.get("summary", {}).get("classification_models_compared", 0),
                "best_regression_model": None,
                "best_classification_model": None,
                "key_insights": []
            },
            "detailed_comparison": {
                "regression": comparison_results.get("regression_comparison", {}),
                "classification": comparison_results.get("classification_comparison", {})
            },
            "performance_analysis": {},
            "recommendations": [],
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "comparison_id": comparison_results.get("summary", {}).get("comparison_id"),
                "report_version": "1.0.0"
            }
        }
        
        # Add best models to executive summary
        if not leaderboard_data.empty:
            regression_models = leaderboard_data[leaderboard_data["model_type"] == "regression"]
            classification_models = leaderboard_data[leaderboard_data["model_type"] == "classification"]
            
            if not regression_models.empty:
                best_reg = regression_models.iloc[0]
                report["executive_summary"]["best_regression_model"] = {
                    "name": best_reg["model_name"],
                    "algorithm": best_reg["algorithm"],
                    "performance": best_reg["primary_metric"]
                }
            
            if not classification_models.empty:
                best_clf = classification_models.iloc[0]
                report["executive_summary"]["best_classification_model"] = {
                    "name": best_clf["model_name"],
                    "algorithm": best_clf["algorithm"],
                    "performance": best_clf["primary_metric"]
                }
        
        # Add key insights
        insights = []
        if "regression_comparison" in comparison_results:
            reg_comp = comparison_results["regression_comparison"]
            if "performance_gap" in reg_comp and reg_comp["performance_gap"]:
                insights.append(f"Performance gap between top 2 regression models: {reg_comp['performance_gap']:.4f}")
        
        if "classification_comparison" in comparison_results:
            clf_comp = comparison_results["classification_comparison"]
            if "performance_gap" in clf_comp and clf_comp["performance_gap"]:
                insights.append(f"Performance gap between top 2 classification models: {clf_comp['performance_gap']:.4f}")
        
        report["executive_summary"]["key_insights"] = insights
        
        logger.info("Model comparison report created successfully")
        return report
        
    except Exception as e:
        logger.error(f"Failed to create comparison report: {str(e)}")
        return {"error": str(e), "generated_at": datetime.now().isoformat()}


def archive_old_models(
    registry: ModelRegistry,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Archive old or underperforming models.
    
    Args:
        registry: Model registry instance
        parameters: Kedro parameters
        
    Returns:
        Dict with archival results
    """
    archival_config = parameters.get("archival", {})
    
    if not archival_config.get("enabled", False):
        logger.info("Model archival is disabled")
        return {
            "archived_models": [],
            "summary": {"total_archived": 0, "archival_enabled": False}
        }
    
    archival_results = {
        "archived_models": [],
        "failed_archival": [],
        "summary": {}
    }
    
    try:
        # Get all models that could be archived
        all_models = registry.list_models()
        
        # Apply archival rules (placeholder - implement based on your needs)
        models_to_archive = []
        
        max_versions = archival_config.get("auto_archive", {}).get("max_versions_per_model", 10)
        
        # Group by model name and keep only recent versions
        for model_name in all_models["model_name"].unique():
            model_versions = all_models[all_models["model_name"] == model_name].sort_values("created_at", ascending=False)
            
            if len(model_versions) > max_versions:
                models_to_archive.extend(model_versions.iloc[max_versions:].to_dict("records"))
        
        # Archive selected models
        for model in models_to_archive:
            try:
                success = registry.archive_model(model["model_name"], model["version"])
                if success:
                    archival_results["archived_models"].append({
                        "model_name": model["model_name"],
                        "version": model["version"],
                        "archived_at": datetime.now().isoformat()
                    })
                else:
                    archival_results["failed_archival"].append({
                        "model_name": model["model_name"],
                        "version": model["version"],
                        "reason": "Archive operation failed"
                    })
            except Exception as e:
                archival_results["failed_archival"].append({
                    "model_name": model["model_name"],
                    "version": model["version"],
                    "reason": str(e)
                })
        
        # Create summary
        archival_results["summary"] = {
            "total_archived": len(archival_results["archived_models"]),
            "failed_count": len(archival_results["failed_archival"]),
            "archival_enabled": True,
            "archival_time": datetime.now().isoformat()
        }
        
        logger.info(f"Model archival completed: {archival_results['summary']}")
        
    except Exception as e:
        logger.error(f"Model archival failed: {str(e)}")
        archival_results["error"] = str(e)
    
    return archival_results