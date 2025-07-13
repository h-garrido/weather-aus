import pandas as pd
import numpy as np
import logging
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def train_svm_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena un modelo SVM Classifier para predicción de lluvia.
    
    Args:
        classification_data: DataFrame con los datos de clasificación
        selected_features: Diccionario con las features seleccionadas
        params: Parámetros del modelo SVM
        
    Returns:
        Dict con el modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de SVM Classifier")
    
    # Información del dataset
    logger.info(f"DataFrame shape: {classification_data.shape}")
    logger.info(f"DataFrame columns: {list(classification_data.columns)}")
    logger.info(classification_data.info())
    logger.info(f"Selected features: {selected_features}")
    
    # Verificar que la columna target existe
    if 'rain_tomorrow_binary' not in classification_data.columns:
        logger.error("Column 'rain_tomorrow_binary' not found in DataFrame!")
        logger.error(f"Available columns: {list(classification_data.columns)}")
        raise KeyError("Column 'rain_tomorrow_binary' not found in classification_data")
    
    # Preparar datos
    features = selected_features['selected_features']
    X = classification_data[features]
    y = classification_data['rain_tomorrow_binary']
    
    logger.info(f"Features utilizadas: {features}")
    logger.info(f"Tamaño de X: {X.shape}")
    logger.info(f"Distribución de y: {y.value_counts().to_dict()}")
    
    # División train/test
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Train set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # ⚠️ IMPORTANTE: SVM requiere escalado de features
    logger.info("Escalando features para SVM...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar SVM
    svm_params = {
        'C': params.get('C', 1.0),
        'kernel': params.get('kernel', 'rbf'),
        'gamma': params.get('gamma', 'scale'),
        'degree': params.get('degree', 3),
        'probability': True,  # Para obtener probabilidades
        'random_state': random_state,
        'class_weight': params.get('class_weight', 'balanced')
    }
    
    logger.info(f"SVM parameters: {svm_params}")
    
    # Entrenar modelo
    logger.info("Entrenando SVM...")
    svm_model = SVC(**svm_params)
    svm_model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred = svm_model.predict(X_train_scaled)
    y_test_pred = svm_model.predict(X_test_scaled)
    y_test_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Validación cruzada (con datos escalados)
    svm_cv = SVC(**svm_params)
    cv_scores = cross_val_score(svm_cv, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info("SVM Classifier entrenado:")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Información del modelo
    logger.info(f"Support vectors count: {svm_model.n_support_}")
    logger.info(f"Total support vectors: {svm_model.support_vectors_.shape[0]}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_test_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Preparar resultados
    results = {
        'model': svm_model,
        'scaler': scaler,  # IMPORTANTE: Guardar el scaler
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        },
        'model_info': {
            'n_support_vectors': svm_model.n_support_.tolist(),
            'total_support_vectors': svm_model.support_vectors_.shape[0],
            'kernel': svm_params['kernel'],
            'C': svm_params['C'],
            'gamma': svm_params['gamma']
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'model_params': svm_params,
        'features_used': features,
        'dataset_info': {
            'total_samples': len(classification_data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(features),
            'class_distribution': y.value_counts().to_dict()
        }
    }
    
    return results