"""
Data Management Pipeline - Unified Data Processing for MLOps
============================================================
Este pipeline unifica todo el procesamiento de datos para que todos los modelos 
usen exactamente la misma data, facilitando comparación justa y automatización.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

logger = logging.getLogger(__name__)

# ==========================================
# 1. DATA VERSIONING & SNAPSHOT
# ==========================================

def create_data_snapshot(weather_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Crea un snapshot versionado de los datos con metadata para tracking.
    
    Args:
        weather_data: DataFrame from PostgreSQL
    
    Returns:
        Dict con data versionada y metadata
    """
    logger.info("🔄 Creating data snapshot for MLOps tracking...")
    
    # Metadata del snapshot
    snapshot_metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_shape": weather_data.shape,
        "data_hash": pd.util.hash_pandas_object(weather_data).sum(),
        "columns": list(weather_data.columns),
        "missing_values": weather_data.isnull().sum().to_dict(),
        "date_range": {
            "start": weather_data['Date'].min() if 'Date' in weather_data.columns else None,
            "end": weather_data['Date'].max() if 'Date' in weather_data.columns else None
        }
    }
    
    # Log info importante
    logger.info(f"📊 Data snapshot created: {snapshot_metadata['data_shape']} rows x cols")
    logger.info(f"📅 Date range: {snapshot_metadata['date_range']['start']} to {snapshot_metadata['date_range']['end']}")
    
    return {
        "data": weather_data.copy(),
        "metadata": snapshot_metadata,
        "version": f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }


# ==========================================
# 2. UNIFIED TEMPORAL SPLITS
# ==========================================

def create_unified_temporal_splits(
    versioned_data: Dict[str, Any], 
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Crea splits temporales unificados para TODOS los modelos.
    Esto asegura que todos los modelos usen exactamente la misma partición.
    
    Args:
        versioned_data: Output del snapshot anterior
        parameters: Config desde parameters.yml
    
    Returns:
        X_train, X_test, y_classification_train, y_classification_test, 
        y_regression_train, y_regression_test
    """
    logger.info("🔪 Creating unified temporal splits for all models...")
    
    data = versioned_data["data"].copy()
    params = parameters.get("data_management", {})
    
    # ===== 1. DATA CLEANING UNIFICADO =====
    logger.info("🧹 Applying unified data cleaning...")
    
    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Fill categorical missing values
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
    
    # ===== 2. FEATURE PREPARATION =====
    logger.info("🔧 Preparing features for unified processing...")
    
    # Convert Date to datetime if exists
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
    
    # Identify target variables
    classification_target = params.get('classification_target', 'RainTomorrow')
    regression_target = params.get('regression_target', 'Rainfall')
    
    # Exclude targets and ALL datetime columns (not just 'Date')
    datetime_columns = data.select_dtypes(include=['datetime64']).columns.tolist()
    exclude_columns = [classification_target, regression_target] + datetime_columns
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    
    X = data[feature_columns].copy()
    y_classification = data[classification_target].copy() if classification_target in data.columns else None
    y_regression = data[regression_target].copy() if regression_target in data.columns else None
    
    # ===== 3. TEMPORAL SPLIT STRATEGY =====
    split_method = params.get('split_method', 'temporal')
    test_size = params.get('test_size', 0.2)
    
    if split_method == 'temporal' and 'Date' in data.columns:
        logger.info("📅 Using temporal split (chronological order)")
        # Split by time: últimos X% para test
        split_idx = int(len(data) * (1 - test_size))
        
        X_train = X.iloc[:split_idx].copy()
        X_test = X.iloc[split_idx:].copy()
        
        if y_classification is not None:
            y_classification_train = y_classification.iloc[:split_idx].copy()
            y_classification_test = y_classification.iloc[split_idx:].copy()
        else:
            y_classification_train = pd.Series()
            y_classification_test = pd.Series()
            
        if y_regression is not None:
            y_regression_train = y_regression.iloc[:split_idx].copy()
            y_regression_test = y_regression.iloc[split_idx:].copy()
        else:
            y_regression_train = pd.Series()
            y_regression_test = pd.Series()
            
    else:
        logger.info("🎲 Using random split (stratified)")
        # Random split con estratificación para clasificación
        if y_classification is not None:
            X_train, X_test, y_classification_train, y_classification_test = train_test_split(
                X, y_classification, 
                test_size=test_size, 
                random_state=params.get('random_state', 42),
                stratify=y_classification
            )
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=params.get('random_state', 42)
            )
            y_classification_train = pd.Series()
            y_classification_test = pd.Series()
        
        if y_regression is not None:
            # Para regresión, usar los mismos índices
            y_regression_train = y_regression.loc[X_train.index]
            y_regression_test = y_regression.loc[X_test.index]
        else:
            y_regression_train = pd.Series()
            y_regression_test = pd.Series()
    
    # ===== 4. ENCODE CATEGORICAL VARIABLES =====
    logger.info("🏷️ Encoding categorical variables...")
    
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        # Fit en train, transform en ambos
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        # Para test, handle unseen categories
        X_test[col] = X_test[col].astype(str)
        X_test[col] = X_test[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
        # Add 'Unknown' to classes if needed
        if 'Unknown' not in le.classes_:
            le.classes_ = np.append(le.classes_, 'Unknown')
        X_test[col] = le.transform(X_test[col])
        label_encoders[col] = le
    
    # Encode target de clasificación si es string
    if y_classification is not None and y_classification_train.dtype == 'object':
        target_encoder = LabelEncoder()
        y_classification_train = pd.Series(target_encoder.fit_transform(y_classification_train), 
                                         index=y_classification_train.index)
        y_classification_test = pd.Series(target_encoder.transform(y_classification_test), 
                                        index=y_classification_test.index)
    
    # ===== 5. LOGGING FINAL =====
    logger.info("✅ Unified splits created successfully!")
    logger.info(f"📊 Training set: {X_train.shape}")
    logger.info(f"📊 Test set: {X_test.shape}")
    if y_classification is not None:
        logger.info(f"🎯 Classification target distribution (train): {y_classification_train.value_counts().to_dict()}")
    if y_regression is not None:
        logger.info(f"🎯 Regression target stats (train): mean={y_regression_train.mean():.2f}, std={y_regression_train.std():.2f}")

    # Debug: Check final shapes before return
    print(f"🔍 DEBUG - Final X_train shape: {X_train.shape}")
    print(f"🔍 DEBUG - Final y_classification_train shape: {pd.DataFrame({'target': y_classification_train}).shape}")
    print(f"🔍 DEBUG - Feature columns selected: {len(feature_columns)}")
    print(f"🔍 DEBUG - Original data shape: {data.shape}")
    
    return (
        X_train,  
        X_test,  
        pd.DataFrame({'target': y_classification_train}) if isinstance(y_classification_train, pd.Series) else y_classification_train,
        pd.DataFrame({'target': y_classification_test}) if isinstance(y_classification_test, pd.Series) else y_classification_test,
        pd.DataFrame({'target': y_regression_train}) if isinstance(y_regression_train, pd.Series) else y_regression_train,
        pd.DataFrame({'target': y_regression_test}) if isinstance(y_regression_test, pd.Series) else y_regression_test
    )




# ==========================================
# 3. UNIFIED FEATURE ENGINEERING
# ==========================================

def apply_unified_feature_engineering(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_classification_train: pd.Series,
    y_regression_train: pd.Series,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Aplica feature engineering unificado y selección de features.
    
    Args:
        X_train, X_test: Features splits
        y_classification_train, y_regression_train: Targets para feature selection
        parameters: Config
    
    Returns:
        X_train_processed, X_test_processed, feature_metadata
    """
    logger.info("⚙️ Applying unified feature engineering...")
    
    params = parameters.get("feature_engineering", {})
    
    # ===== 1. SCALING =====
    if params.get("apply_scaling", True):
        logger.info("📏 Applying feature scaling...")
        scaler = StandardScaler()
        
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
    else:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        scaler = None
    
    # ===== 2. FEATURE SELECTION =====
    feature_selection_method = params.get("feature_selection_method", "correlation")
    max_features = params.get("max_features", 20)
    
    if feature_selection_method == "correlation" and not y_regression_train.empty:
        logger.info("🔍 Selecting features based on correlation...")
        
        # Calculate correlation with regression target
        correlations = X_train_scaled.corrwith(y_regression_train['target']).abs()
        selected_features = correlations.nlargest(max_features).index.tolist()
        
    elif feature_selection_method == "mutual_info" and not y_classification_train.empty:
        logger.info("🔍 Selecting features based on mutual information...")
        
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X_train_scaled, y_classification_train['target'])
        feature_scores = pd.Series(mi_scores, index=X_train_scaled.columns)
        selected_features = feature_scores.nlargest(max_features).index.tolist()
        
    else:
        logger.info("🔍 Using all features (no selection)")
        selected_features = X_train_scaled.columns.tolist()
    
    # Apply feature selection
    X_train_final = X_train_scaled[selected_features].copy()
    X_test_final = X_test_scaled[selected_features].copy()
    
    # ===== 3. FEATURE METADATA =====
    feature_metadata = {
        "timestamp": datetime.now().isoformat(),
        "selected_features": selected_features,
        "total_features": len(selected_features),
        "selection_method": feature_selection_method,
        "scaler_applied": scaler is not None,
        "feature_stats": {
            "train_shape": X_train_final.shape,
            "test_shape": X_test_final.shape,
            "numeric_features": len(X_train_final.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(X_train_final.select_dtypes(exclude=[np.number]).columns)
        }
    }
    
    # Save scaler for production use
    if scaler is not None:
        feature_metadata["scaler_applied"] = True
        feature_metadata["scaler_type"] = "StandardScaler"
    
    logger.info("✅ Feature engineering completed!")
    logger.info(f"📊 Final feature set: {X_train_final.shape[1]} features")
    logger.info(f"🎯 Selected features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}")
    
    return X_train_final, X_test_final, feature_metadata


# ==========================================
# 4. DATA QUALITY CHECKS
# ==========================================

def validate_data_quality(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_classification_train: pd.Series,
    y_regression_train: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates data quality and generates quality report.
    """
    logger.info("🔍 Validating data quality...")
    
    params = parameters.get("data_quality", {})
    
    quality_report = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "passed": True,
        "warnings": [],
        "errors": []
    }
    
    # Check 1: No empty datasets
    if X_train.empty or X_test.empty:
        quality_report["errors"].append("Empty training or test set")
        quality_report["passed"] = False
    
    # Check 2: Reasonable train/test split
    total_samples = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total_samples
    if test_ratio < 0.1 or test_ratio > 0.5:
        quality_report["warnings"].append(f"Unusual test ratio: {test_ratio:.2%}")
    
    # Check 3: Feature consistency
    if not set(X_train.columns) == set(X_test.columns):
        quality_report["errors"].append("Feature mismatch between train and test")
        quality_report["passed"] = False
    
    # Check 4: Missing values
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    max_missing_threshold = params.get("max_missing_ratio", 0.1)
    
    if train_missing > len(X_train) * len(X_train.columns) * max_missing_threshold:
        quality_report["warnings"].append(f"High missing values in training set: {train_missing}")
    
    # Check 5: Target distribution
    if not y_classification_train.empty:
        class_dist = y_classification_train.value_counts(normalize=True)
        min_class_ratio = params.get("min_class_ratio", 0.05)
        if class_dist.min() < min_class_ratio:
            quality_report["warnings"].append(f"Imbalanced classes: min ratio {class_dist.min():.2%}")
    
    quality_report["checks"] = {
        "empty_datasets": X_train.empty or X_test.empty,
        "test_ratio": test_ratio,
        "feature_consistency": set(X_train.columns) == set(X_test.columns),
        "train_missing": int(train_missing),
        "test_missing": int(test_missing),
        "test_ratio": float(test_ratio), 
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }
    
    logger.info(f"✅ Data quality check completed. Passed: {quality_report['passed']}")
    if quality_report["warnings"]:
        logger.warning(f"⚠️ Warnings: {quality_report['warnings']}")
    if quality_report["errors"]:
        logger.error(f"❌ Errors: {quality_report['errors']}")
    
    return quality_report