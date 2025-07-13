"""
Model Training Pipeline - Nodes
Trains classification and regression models for weather prediction.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib

logger = logging.getLogger(__name__)


def train_classification_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train classification model to predict rain tomorrow.
    
    Args:
        X_train: Training features
        y_train: Training targets (rain tomorrow)
        parameters: Model parameters from conf/
        
    Returns:
        Dictionary containing trained model and metadata
    """
    logger.info("🤖 Training classification model for rain prediction...")
    
    # Get model parameters
    model_params = parameters.get("model_training", {}).get("classification", {})
    
    # Initialize model
    if model_params.get("algorithm", "random_forest") == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 10),
            random_state=42,
            n_jobs=-1
        )
    else:
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

    # Debug: Check data shapes
    print(f"🔍 DEBUG - X_train shape: {X_train.shape}")
    print(f"🔍 DEBUG - y_train shape: {y_train.shape}")
    print(f"🔍 DEBUG - X_train columns: {list(X_train.columns)}")
    print(f"🔍 DEBUG - y_train values: {y_train.values}")
    print(f"🔍 DEBUG - First few rows of X_train:")
    print(X_train.head() if not X_train.empty else "X_train is EMPTY")
    
    # Train model
    model.fit(X_train, y_train.values.ravel())
    
    # Get feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Log top 5 features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"🎯 Top 5 important features: {[f[0] for f in top_features]}")
    
    # Training predictions for basic metrics
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    logger.info(f"✅ Classification model trained successfully!")
    logger.info(f"📊 Training accuracy: {train_accuracy:.4f}")
    
    return {
        "model": model,
        "model_type": "classification",
        "algorithm": model_params.get("algorithm", "random_forest"),
        "train_accuracy": float(train_accuracy),
        "feature_importance": feature_importance,
        "n_features": len(X_train.columns),
        "n_training_samples": len(X_train)
    }


def train_regression_model(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train regression model to predict max temperature.
    
    Args:
        X_train: Training features  
        y_train: Training targets (max temperature)
        parameters: Model parameters from conf/
        
    Returns:
        Dictionary containing trained model and metadata
    """
    logger.info("🌡️ Training regression model for temperature prediction...")
    
    # Get model parameters
    model_params = parameters.get("model_training", {}).get("regression", {})
    
    # Initialize model
    if model_params.get("algorithm", "random_forest") == "random_forest":
        model = RandomForestRegressor(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", 10),
            random_state=42,
            n_jobs=-1
        )
    else:
        model = LinearRegression()
    
    # Train model
    model.fit(X_train, y_train.values.ravel())
    
    # Get feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        
        # Log top 5 features
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"🎯 Top 5 important features: {[f[0] for f in top_features]}")
    
    # Training predictions for basic metrics
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    logger.info(f"✅ Regression model trained successfully!")
    logger.info(f"📊 Training RMSE: {train_rmse:.4f}")
    logger.info(f"📊 Training R²: {train_r2:.4f}")
    
    return {
        "model": model,
        "model_type": "regression", 
        "algorithm": model_params.get("algorithm", "random_forest"),
        "train_rmse": float(train_rmse),
        "train_r2": float(train_r2),
        "feature_importance": feature_importance,
        "n_features": len(X_train.columns),
        "n_training_samples": len(X_train)
    }


def evaluate_models_on_test(
    classification_model_dict: Dict[str, Any],
    regression_model_dict: Dict[str, Any], 
    X_test_classification: pd.DataFrame,
    X_test_regression: pd.DataFrame,
    y_test_classification: pd.DataFrame,
    y_test_regression: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate both models on test data and generate comprehensive metrics.
    
    Args:
        classification_model_dict: Trained classification model and metadata
        regression_model_dict: Trained regression model and metadata
        X_test_classification: Test features for classification
        X_test_regression: Test features for regression  
        y_test_classification: Test targets for classification
        y_test_regression: Test targets for regression
        
    Returns:
        Dictionary with evaluation metrics for both models
    """
    logger.info("🔍 Evaluating models on test data...")
    
    results = {
        "classification": {},
        "regression": {},
        "evaluation_timestamp": pd.Timestamp.now().isoformat()
    }
    
    # Classification evaluation
    clf_model = classification_model_dict["model"]
    y_pred_clf = clf_model.predict(X_test_classification)
    y_pred_proba_clf = clf_model.predict_proba(X_test_classification)[:, 1] if hasattr(clf_model, 'predict_proba') else None
    
    results["classification"] = {
        "test_accuracy": float(accuracy_score(y_test_classification, y_pred_clf)),
        "classification_report": classification_report(y_test_classification, y_pred_clf, output_dict=True),
        "model_metadata": {
            "algorithm": classification_model_dict["algorithm"],
            "train_accuracy": classification_model_dict["train_accuracy"],
            "n_features": classification_model_dict["n_features"],
            "n_training_samples": classification_model_dict["n_training_samples"]
        }
    }
    
    # Regression evaluation  
    reg_model = regression_model_dict["model"]
    y_pred_reg = reg_model.predict(X_test_regression)
    
    results["regression"] = {
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_regression, y_pred_reg))),
        "test_mae": float(mean_absolute_error(y_test_regression, y_pred_reg)),
        "test_r2": float(r2_score(y_test_regression, y_pred_reg)),
        "model_metadata": {
            "algorithm": regression_model_dict["algorithm"],
            "train_rmse": regression_model_dict["train_rmse"], 
            "train_r2": regression_model_dict["train_r2"],
            "n_features": regression_model_dict["n_features"],
            "n_training_samples": regression_model_dict["n_training_samples"]
        }
    }
    
    logger.info("✅ Model evaluation completed!")
    logger.info(f"🎯 Classification accuracy: {results['classification']['test_accuracy']:.4f}")
    logger.info(f"🌡️ Regression RMSE: {results['regression']['test_rmse']:.4f}")
    logger.info(f"🌡️ Regression R²: {results['regression']['test_r2']:.4f}")
    
    return results