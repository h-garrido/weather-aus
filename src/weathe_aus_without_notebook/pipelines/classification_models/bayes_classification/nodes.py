import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def train_bayes_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    bayes_params: Dict
) -> Dict[str, Any]:
    """
    Entrena modelo Naive Bayes Classifier.
    
    Args:
        classification_data: DataFrame con datos de clasificación
        selected_features: Features seleccionadas del pipeline anterior
        bayes_params: Parámetros para Naive Bayes
        
    Returns:
        Diccionario con modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de Naive Bayes Classifier")
    
    # 🔍 DEBUG: Verificar el DataFrame recibido
    logger.info(f"DataFrame shape: {classification_data.shape}")
    logger.info(f"DataFrame columns: {list(classification_data.columns)}")
    logger.info(f"DataFrame info: {classification_data.info()}")
    logger.info(f"Selected features: {selected_features}")
    
    # Verificar si 'rain_tomorrow_binary' existe
    if 'rain_tomorrow_binary' not in classification_data.columns:
        logger.error("Column 'rain_tomorrow_binary' not found in DataFrame!")
        logger.error(f"Available columns: {list(classification_data.columns)}")
        raise KeyError("Column 'rain_tomorrow_binary' not found in classification_data")
    
    # Preparar datos
    features = selected_features['selected_features']
    logger.info(f"Using features: {features}")
    
    # Verificar que todas las features existen
    missing_features = [f for f in features if f not in classification_data.columns]
    if missing_features:
        logger.error(f"Missing features: {missing_features}")
        logger.error(f"Available columns: {list(classification_data.columns)}")
        raise KeyError(f"Missing features in classification_data: {missing_features}")
    
    X = classification_data[features].fillna(0)
    y = classification_data['rain_tomorrow_binary']

    
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")
    logger.info(f"y value counts: {y.value_counts()}")
    
    # Split train/test
    test_size = bayes_params.get('test_size', 0.2)
    random_state = bayes_params.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Configurar modelo Gaussian Naive Bayes
    model = GaussianNB(
        var_smoothing=bayes_params.get('var_smoothing', 1e-9)
    )
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Calcular métricas principales
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Feature importance (aproximada para Naive Bayes)
    # Para Naive Bayes no hay feature_importance directa, 
    # usamos la varianza de cada feature por clase como proxy
    feature_importance = {}
    for i, feature in enumerate(features):
        class_0_var = np.var(X_train[y_train == 0].iloc[:, i])
        class_1_var = np.var(X_train[y_train == 1].iloc[:, i])
        # Importancia inversa a la varianza promedio (menor varianza = más discriminativo)
        avg_var = (class_0_var + class_1_var) / 2
        importance = 1 / (1 + avg_var) if avg_var > 0 else 1.0
        feature_importance[feature] = importance
    
    # Ordenar por importancia
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    results = {
        'model': model,
        'model_name': 'Naive Bayes Classifier',
        'feature_names': features,
        'training_data': {
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy
        },
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'confusion_matrix': {
            'matrix': cm.tolist(),
            'labels': ['No Rain', 'Rain']
        },
        'cross_validation': {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        },
        'feature_importance': sorted_importance,
        'model_params': model.get_params(),
        'model_complexity': {
            'n_features': len(features),
            'var_smoothing': model.var_smoothing,
            'model_type': 'Gaussian Naive Bayes'
        },
        'predictions': {
            'y_test': y_test.tolist(),
            'y_pred': y_pred_test.tolist(),
            'y_pred_proba': y_pred_proba_test.tolist()
        }
    }
    
    logger.info(f"Naive Bayes Classifier entrenado:")
    logger.info(f"  Train Accuracy: {results['training_data']['train_accuracy']:.4f}")
    logger.info(f"  Test Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results