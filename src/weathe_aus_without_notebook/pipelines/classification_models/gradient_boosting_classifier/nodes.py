import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def train_gradient_boosting_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena un modelo Gradient Boosting Classifier para predicción de lluvia.
    
    Args:
        classification_data: DataFrame con los datos de clasificación
        selected_features: Diccionario con las features seleccionadas
        params: Parámetros del modelo Gradient Boosting
        
    Returns:
        Dict con el modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de Gradient Boosting Classifier")
    
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
    
    # Configurar Gradient Boosting
    gb_params = {
        'n_estimators': params.get('n_estimators', 100),        # Número de árboles
        'learning_rate': params.get('learning_rate', 0.1),      # Tasa de aprendizaje
        'max_depth': params.get('max_depth', 3),                # Profundidad de cada árbol
        'min_samples_split': params.get('min_samples_split', 2), # Mínimo para dividir nodo
        'min_samples_leaf': params.get('min_samples_leaf', 1),   # Mínimo en hoja
        'subsample': params.get('subsample', 1.0),              # Fracción de muestras por árbol
        'max_features': params.get('max_features', None),        # Máximo features por split
        'random_state': random_state,
        'validation_fraction': params.get('validation_fraction', 0.1),  # Para early stopping
        'n_iter_no_change': params.get('n_iter_no_change', 5),  # Early stopping patience
        'tol': params.get('tol', 1e-4)                          # Tolerancia para early stopping
    }
    
    logger.info(f"Gradient Boosting parameters: {gb_params}")
    
    # Optimización de hiperparámetros (si está habilitada)
    if params.get('optimize_hyperparams', False):
        logger.info("Optimizando hiperparámetros...")
        
        param_grid = {
            'n_estimators': params.get('n_estimators_range', [50, 100, 200]),
            'learning_rate': params.get('learning_rate_range', [0.01, 0.1, 0.2]),
            'max_depth': params.get('max_depth_range', [3, 5, 7]),
            'min_samples_split': params.get('min_samples_split_range', [2, 5, 10])
        }
        
        grid_search = GridSearchCV(
            GradientBoostingClassifier(
                random_state=random_state,
                subsample=gb_params['subsample']
            ),
            param_grid,
            cv=3,  # Usar CV menor porque GB es lento
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        # Actualizar parámetros con los mejores encontrados
        gb_params.update(grid_search.best_params_)
    
    # Entrenar modelo
    logger.info("Entrenando Gradient Boosting...")
    gb_model = GradientBoostingClassifier(**gb_params)
    gb_model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = gb_model.predict(X_train)
    y_test_pred = gb_model.predict(X_test)
    y_test_pred_proba = gb_model.predict_proba(X_test)[:, 1]
    
    # Predicciones staged (por número de estimadores)
    train_staged_scores = []
    test_staged_scores = []
    
    for i, y_staged_pred in enumerate(gb_model.staged_predict(X_train)):
        train_staged_scores.append(accuracy_score(y_train, y_staged_pred))
    
    for i, y_staged_pred in enumerate(gb_model.staged_predict(X_test)):
        test_staged_scores.append(accuracy_score(y_test, y_staged_pred))
    
    # Métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Validación cruzada
    cv_scores = cross_val_score(gb_model, X_train, y_train, cv=3, scoring='accuracy')  # CV menor por velocidad
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Análisis de importancia de features (insight único de Gradient Boosting)
    logger.info("Analizando importancia de features...")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 features más importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Análisis de convergencia
    logger.info("Analizando convergencia del modelo...")
    best_iteration = np.argmax(test_staged_scores) + 1
    best_test_score = max(test_staged_scores)
    final_train_score = train_staged_scores[-1]
    final_test_score = test_staged_scores[-1]
    
    logger.info(f"Mejor iteración: {best_iteration} (test accuracy: {best_test_score:.4f})")
    logger.info(f"Score final: Train={final_train_score:.4f}, Test={final_test_score:.4f}")
    
    # Detectar overfitting durante entrenamiento
    overfitting_point = None
    for i in range(10, len(test_staged_scores)):
        if test_staged_scores[i] < test_staged_scores[i-10]:
            overfitting_point = i
            break
    
    if overfitting_point:
        logger.warning(f"⚠️ Overfitting detectado alrededor de la iteración {overfitting_point}")
        logger.warning("Considera usar early stopping o reducir n_estimators")
    
    logger.info("Gradient Boosting Classifier entrenado:")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Número de estimadores: {gb_params['n_estimators']}")
    logger.info(f"  Learning rate: {gb_params['learning_rate']}")
    logger.info(f"  Max depth por árbol: {gb_params['max_depth']}")
    logger.info(f"  Training loss final: {gb_model.train_score_[-1]:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_test_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Preparar resultados
    results = {
        'model': gb_model,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'best_iteration': best_iteration,
            'best_test_score': best_test_score
        },
        'model_info': {
            'n_estimators': gb_params['n_estimators'],
            'learning_rate': gb_params['learning_rate'],
            'max_depth': gb_params['max_depth'],
            'subsample': gb_params['subsample'],
            'final_train_loss': gb_model.train_score_[-1],
            'overfitting_point': overfitting_point
        },
        'feature_importance': {
            'importances': gb_model.feature_importances_.tolist(),
            'feature_names': features,
            'top_features': feature_importance.head(15).to_dict('records')
        },
        'training_evolution': {
            'train_scores': train_staged_scores,
            'test_scores': test_staged_scores,
            'train_losses': gb_model.train_score_.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'model_params': gb_params,
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