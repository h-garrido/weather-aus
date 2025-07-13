import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def train_logistic_regression_classification(
    classification_data: pd.DataFrame,
    selected_features: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entrena un modelo Logistic Regression Classifier para predicción de lluvia.
    
    Args:
        classification_data: DataFrame con los datos de clasificación
        selected_features: Diccionario con las features seleccionadas
        params: Parámetros del modelo Logistic Regression
        
    Returns:
        Dict con el modelo entrenado y métricas
    """
    logger.info("Iniciando entrenamiento de Logistic Regression Classifier")
    
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
    
    # Escalado (recomendado para Logistic Regression)
    if params.get('scale_features', True):
        logger.info("Escalando features para Logistic Regression...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        logger.info("Sin escalado de features")
        scaler = None
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Configurar Logistic Regression
    lr_params = {
        'C': params.get('C', 1.0),                          # Regularización
        'penalty': params.get('penalty', 'l2'),             # Tipo de regularización
        'solver': params.get('solver', 'lbfgs'),            # Algoritmo de optimización
        'max_iter': params.get('max_iter', 1000),           # Máximo de iteraciones
        'class_weight': params.get('class_weight', 'balanced'),  # Balanceo de clases
        'random_state': random_state,
        'n_jobs': params.get('n_jobs', -1)
    }
    
    # Validar compatibilidad solver-penalty
    if lr_params['penalty'] == 'elasticnet' and lr_params['solver'] not in ['saga']:
        logger.warning("ElasticNet requiere solver 'saga'. Cambiando solver...")
        lr_params['solver'] = 'saga'
    
    if lr_params['penalty'] == 'none' and lr_params['solver'] not in ['lbfgs', 'newton-cg', 'sag', 'saga']:
        logger.warning("No penalty requiere solver compatible. Cambiando a lbfgs...")
        lr_params['solver'] = 'lbfgs'
    
    logger.info(f"Logistic Regression parameters: {lr_params}")
    
    # Optimización de hiperparámetros (si está habilitada)
    if params.get('optimize_hyperparams', False):
        logger.info("Optimizando hiperparámetros...")
        
        param_grid = {
            'C': params.get('C_range', [0.01, 0.1, 1.0, 10.0, 100.0]),
            'penalty': params.get('penalty_options', ['l1', 'l2']),
            'solver': ['liblinear']  # Compatible con l1 y l2
        }
        
        grid_search = GridSearchCV(
            LogisticRegression(
                max_iter=lr_params['max_iter'],
                class_weight=lr_params['class_weight'],
                random_state=random_state,
                n_jobs=-1
            ),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        # Actualizar parámetros con los mejores encontrados
        lr_params.update(grid_search.best_params_)
    
    # Entrenar modelo
    logger.info("Entrenando Logistic Regression...")
    lr_model = LogisticRegression(**lr_params)
    lr_model.fit(X_train_scaled, y_train)
    
    # Predicciones
    y_train_pred = lr_model.predict(X_train_scaled)
    y_test_pred = lr_model.predict(X_test_scaled)
    y_test_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Métricas de evaluación
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # Validación cruzada
    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Análisis de coeficientes (insight único de Logistic Regression)
    logger.info("Analizando coeficientes del modelo...")
    coef_importance = pd.DataFrame({
        'feature': features,
        'coefficient': lr_model.coef_[0],
        'abs_coefficient': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    logger.info("Top 5 features más importantes (por coeficiente):")
    for idx, row in coef_importance.head().iterrows():
        direction = "AUMENTA" if row['coefficient'] > 0 else "DISMINUYE"
        logger.info(f"  {row['feature']}: {row['coefficient']:.4f} ({direction} prob. lluvia)")
    
    logger.info("Logistic Regression Classifier entrenado:")
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Regularización C: {lr_params['C']}")
    logger.info(f"  Penalty: {lr_params['penalty']}")
    logger.info(f"  Solver: {lr_params['solver']}")
    logger.info(f"  Converged: {lr_model.n_iter_[0] < lr_params['max_iter']}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Reporte de clasificación
    class_report = classification_report(y_test, y_test_pred)
    logger.info(f"Classification Report:\n{class_report}")
    
    # Preparar resultados
    results = {
        'model': lr_model,
        'scaler': scaler,  # Puede ser None si no se escaló
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
            'C': lr_params['C'],
            'penalty': lr_params['penalty'],
            'solver': lr_params['solver'],
            'max_iter': lr_params['max_iter'],
            'n_iter': lr_model.n_iter_[0],
            'converged': lr_model.n_iter_[0] < lr_params['max_iter'],
            'intercept': lr_model.intercept_[0]
        },
        'feature_importance': {
            'coefficients': lr_model.coef_[0].tolist(),
            'feature_names': features,
            'top_features': coef_importance.head(10).to_dict('records')
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'model_params': lr_params,
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