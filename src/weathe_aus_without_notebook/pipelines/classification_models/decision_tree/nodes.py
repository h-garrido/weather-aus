import pandas as pd
import numpy as np
import logging
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import io
import base64

logger = logging.getLogger(__name__)

def train_decision_tree_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena un modelo Decision Tree Classifier para predicción de lluvia.
    
    Args:
        classification_data: DataFrame con los datos de clasificación
        selected_features: Diccionario con las features seleccionadas
        params: Parámetros del modelo Decision Tree
        
    Returns:
        Dict con el modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de Decision Tree Classifier")
    
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
    
    # Configurar Decision Tree
    dt_params = {
        'criterion': params.get('criterion', 'gini'),           # Métrica de impureza
        'max_depth': params.get('max_depth', None),             # Profundidad máxima
        'min_samples_split': params.get('min_samples_split', 2), # Mínimo para dividir nodo
        'min_samples_leaf': params.get('min_samples_leaf', 1),   # Mínimo en hoja
        'max_features': params.get('max_features', None),        # Máximo features por split
        'class_weight': params.get('class_weight', 'balanced'),  # Balanceo de clases
        'random_state': random_state,
        'min_impurity_decrease': params.get('min_impurity_decrease', 0.0),
        'max_leaf_nodes': params.get('max_leaf_nodes', None)
    }
    
    logger.info(f"Decision Tree parameters: {dt_params}")
    
    # Optimización de hiperparámetros (si está habilitada)
    if params.get('optimize_hyperparams', False):
        logger.info("Optimizando hiperparámetros...")
        
        param_grid = {
            'max_depth': params.get('max_depth_range', [None, 5, 10, 15, 20]),
            'min_samples_split': params.get('min_samples_split_range', [2, 5, 10, 20]),
            'min_samples_leaf': params.get('min_samples_leaf_range', [1, 2, 5, 10]),
            'criterion': params.get('criterion_options', ['gini', 'entropy'])
        }
        
        grid_search = GridSearchCV(
            DecisionTreeClassifier(
                class_weight=dt_params['class_weight'],
                random_state=random_state
            ),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        # Actualizar parámetros con los mejores encontrados
        dt_params.update(grid_search.best_params_)
    
    # Entrenar modelo
    logger.info("Entrenando Decision Tree...")
    dt_model = DecisionTreeClassifier(**dt_params)
    dt_model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    y_test_pred_proba = dt_model.predict_proba(X_test)[:, 1]
    
    # Métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Validación cruzada
    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Análisis de importancia de features (insight único de Decision Tree)
    logger.info("Analizando importancia de features...")
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 features más importantes:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Información del árbol
    tree_depth = dt_model.get_depth()
    tree_leaves = dt_model.get_n_leaves()
    
    logger.info("Decision Tree Classifier entrenado:")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Profundidad del árbol: {tree_depth}")
    logger.info(f"  Número de hojas: {tree_leaves}")
    logger.info(f"  Criterio usado: {dt_params['criterion']}")
    
    # Detectar overfitting
    overfitting_gap = train_accuracy - test_accuracy
    if overfitting_gap > 0.1:
        logger.warning(f"⚠️ Posible overfitting detectado! Gap: {overfitting_gap:.4f}")
        logger.warning("Considera reducir max_depth o aumentar min_samples_split/leaf")
    
    # Generar reglas del árbol (primeras 20 reglas)
    tree_rules = export_text(dt_model, feature_names=features, max_depth=3)
    logger.info("Primeras reglas del árbol de decisión:")
    logger.info(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_test_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Preparar resultados
    results = {
        'model': dt_model,
        'metrics': {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'overfitting_gap': overfitting_gap
        },
        'model_info': {
            'tree_depth': tree_depth,
            'n_leaves': tree_leaves,
            'criterion': dt_params['criterion'],
            'max_depth': dt_params['max_depth'],
            'min_samples_split': dt_params['min_samples_split'],
            'min_samples_leaf': dt_params['min_samples_leaf']
        },
        'feature_importance': {
            'importances': dt_model.feature_importances_.tolist(),
            'feature_names': features,
            'top_features': feature_importance.head(15).to_dict('records')
        },
        'tree_rules': tree_rules,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'model_params': dt_params,
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