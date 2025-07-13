import pandas as pd
import numpy as np
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def train_knn_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena un modelo KNN Classifier para predicción de lluvia.
    
    Args:
        classification_data: DataFrame con los datos de clasificación
        selected_features: Diccionario con las features seleccionadas
        params: Parámetros del modelo KNN
        
    Returns:
        Dict con el modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de KNN Classifier")
    
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
    
    # ⚠️ IMPORTANTE: KNN es sensible a la escala
    logger.info("Escalando features para KNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar KNN
    knn_params = {
        'n_neighbors': params.get('n_neighbors', 5),
        'weights': params.get('weights', 'uniform'),
        'algorithm': params.get('algorithm', 'auto'),
        'leaf_size': params.get('leaf_size', 30),
        'metric': params.get('metric', 'minkowski'),
        'p': params.get('p', 2),  # Para métrica minkowski (2=euclidean, 1=manhattan)
        'n_jobs': params.get('n_jobs', -1)
    }
    
    logger.info(f"KNN parameters: {knn_params}")
    
    # Optimización automática de k (si está habilitada)
    if params.get('optimize_k', False):
        logger.info("Optimizando número de vecinos (k)...")
        k_range = params.get('k_range', [3, 5, 7, 9, 11, 15, 21])
        
        grid_params = {'n_neighbors': k_range}
        grid_search = GridSearchCV(
            KNeighborsClassifier(**{k: v for k, v in knn_params.items() if k != 'n_neighbors'}),
            grid_params,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        best_k = grid_search.best_params_['n_neighbors']
        logger.info(f"Mejor k encontrado: {best_k}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        knn_params['n_neighbors'] = best_k
    
    # Entrenar modelo
    logger.info("Entrenando KNN...")
    knn_model = KNeighborsClassifier(**knn_params)
    knn_model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred = knn_model.predict(X_train_scaled)
    y_test_pred = knn_model.predict(X_test_scaled)
    y_test_pred_proba = knn_model.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Validación cruzada
    cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Analizar distancias promedio (insight único de KNN)
    if params.get('analyze_distances', True):
        logger.info("Analizando distancias de vecinos más cercanos...")
        distances, indices = knn_model.kneighbors(X_test_scaled)
        avg_distance = distances.mean()
        std_distance = distances.std()
        logger.info(f"Distancia promedio a vecinos: {avg_distance:.4f} ± {std_distance:.4f}")
    
    logger.info("KNN Classifier entrenado:")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Número de vecinos usado: {knn_params['n_neighbors']}")
    logger.info(f"  Algoritmo: {knn_params['algorithm']}")
    logger.info(f"  Métrica de distancia: {knn_params['metric']}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_test_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Preparar resultados
    results = {
        'model': knn_model,
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
            'n_neighbors': knn_params['n_neighbors'],
            'weights': knn_params['weights'],
            'algorithm': knn_params['algorithm'],
            'metric': knn_params['metric'],
            'avg_distance_to_neighbors': avg_distance if params.get('analyze_distances', True) else None,
            'std_distance_to_neighbors': std_distance if params.get('analyze_distances', True) else None
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'model_params': knn_params,
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