import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, 
    chi2, 
    mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

def prepare_classification_data(transformed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para clasificación binaria de rain_tomorrow_binary.
    
    Args:
        transformed_data: DataFrame con datos transformados
        
    Returns:
        DataFrame preparado para clasificación
    """
    logger.info("Preparando datos para clasificación")
    
    # Hacer copia para no modificar original
    data = transformed_data.copy()
    
    # Verificar que existe la columna target
    if 'rain_tomorrow_binary' not in data.columns:
        raise ValueError("La columna 'rain_tomorrow_binary' no existe en los datos")
    
    # La columna ya está en formato binario (0/1), solo renombrar para consistencia
    data['target'] = data['rain_tomorrow_binary']
    
    # Eliminar filas con valores faltantes en el target
    data = data.dropna(subset=['target'])
    
    # Información del dataset
    logger.info(f"Dataset preparado: {data.shape[0]} filas, {data.shape[1]} columnas")
    logger.info(f"Distribución del target:")
    logger.info(f"No lluvia (0): {(data['target'] == 0).sum()}")
    logger.info(f"Lluvia (1): {(data['target'] == 1).sum()}")
    
    return data

def correlation_analysis_classification(classification_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis de correlación entre features y target de clasificación.
    
    Args:
        classification_data: DataFrame preparado para clasificación
        
    Returns:
        Diccionario con resultados de correlación
    """
    logger.info("Realizando análisis de correlación para clasificación")
    
    # Seleccionar solo columnas numéricas
    numeric_cols = classification_data.select_dtypes(include=[np.number]).columns
    numeric_data = classification_data[numeric_cols]
    
    # Calcular correlación con el target
    target_corr = numeric_data.corr()['target'].abs().sort_values(ascending=False)
    
    # Excluir el target de sí mismo y rain_tomorrow_binary original
    target_corr = target_corr.drop(['target', 'rain_tomorrow_binary'], errors='ignore')
    
    # Matriz de correlación completa
    correlation_matrix = numeric_data.corr()
    
    results = {
        'target_correlation': target_corr.to_dict(),
        'correlation_matrix': correlation_matrix.to_dict(),
        'top_correlated_features': target_corr.head(10).index.tolist(),
        'correlation_scores': target_corr.head(10).values.tolist()
    }
    
    logger.info(f"Top 5 features correlacionadas con rain_tomorrow_binary:")
    for feature, corr in target_corr.head(5).items():
        logger.info(f"  {feature}: {corr:.4f}")
    
    return results

def mutual_information_analysis(classification_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis de información mutua para features de clasificación.
    
    Args:
        classification_data: DataFrame preparado para clasificación
        
    Returns:
        Diccionario con resultados de información mutua
    """
    logger.info("Realizando análisis de información mutua")
    
    # Preparar features (solo numéricas)
    numeric_cols = classification_data.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(['target', 'rain_tomorrow_binary'], errors='ignore')  # Excluir targets
    
    X = classification_data[numeric_cols].fillna(0)  # Rellenar NaN con 0
    y = classification_data['target']
    
    # Calcular información mutua
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Crear Series con nombres de features
    mi_series = pd.Series(mi_scores, index=numeric_cols).sort_values(ascending=False)
    
    results = {
        'mutual_info_scores': mi_series.to_dict(),
        'top_mi_features': mi_series.head(10).index.tolist(),
        'mi_values': mi_series.head(10).values.tolist()
    }
    
    logger.info(f"Top 5 features por información mutua:")
    for feature, score in mi_series.head(5).items():
        logger.info(f"  {feature}: {score:.4f}")
    
    return results

def chi_square_analysis(classification_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis Chi-square para variables categóricas vs target de clasificación.
    
    Args:
        classification_data: DataFrame preparado para clasificación
        
    Returns:
        Diccionario with resultados de chi-square
    """
    logger.info("Realizando análisis Chi-square para variables categóricas")
    
    # Identificar columnas categóricas
    categorical_cols = classification_data.select_dtypes(include=['object']).columns
    # Excluir columnas que no queremos analizar
    categorical_cols = categorical_cols.drop(['rain_tomorrow_binary'], errors='ignore')
    
    chi_square_results = {}
    
    if len(categorical_cols) == 0:
        logger.info("No se encontraron variables categóricas para análisis chi-square")
        return {
            'chi_square_results': {},
            'significant_features': []
        }
    
    for col in categorical_cols:
        try:
            # Crear tabla de contingencia
            contingency_table = pd.crosstab(
                classification_data[col], 
                classification_data['target']
            )
            
            # Realizar test chi-square
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            chi_square_results[col] = {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'significant': p_value < 0.05
            }
            
        except Exception as e:
            logger.warning(f"No se pudo calcular chi-square para {col}: {str(e)}")
            chi_square_results[col] = {
                'chi2_statistic': 0,
                'p_value': 1,
                'degrees_of_freedom': 0,
                'significant': False
            }
    
    # Ordenar por estadístico chi-square
    sorted_results = dict(sorted(
        chi_square_results.items(), 
        key=lambda x: x[1]['chi2_statistic'], 
        reverse=True
    ))
    
    logger.info(f"Variables categóricas analizadas: {len(categorical_cols)}")
    logger.info(f"Variables significativas (p<0.05): {sum(1 for r in chi_square_results.values() if r['significant'])}")
    
    return {
        'chi_square_results': sorted_results,
        'significant_features': [k for k, v in sorted_results.items() if v['significant']]
    }

def feature_importance_analysis(classification_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Análisis de importancia de features usando Random Forest.
    
    Args:
        classification_data: DataFrame preparado para clasificación
        
    Returns:
        Diccionario con importancia de features
    """
    logger.info("Realizando análisis de importancia con Random Forest")
    
    # Preparar features numéricas
    numeric_cols = classification_data.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(['target', 'rain_tomorrow_binary'], errors='ignore')  # Excluir targets
    
    X = classification_data[numeric_cols].fillna(0)
    y = classification_data['target']
    
    # Entrenar Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Obtener importancia de features
    feature_importance = pd.Series(
        rf.feature_importances_, 
        index=numeric_cols
    ).sort_values(ascending=False)
    
    results = {
        'feature_importance': feature_importance.to_dict(),
        'top_important_features': feature_importance.head(10).index.tolist(),
        'importance_values': feature_importance.head(10).values.tolist(),
        'model_accuracy': rf.score(X, y)
    }
    
    logger.info(f"Accuracy del Random Forest: {rf.score(X, y):.4f}")
    logger.info(f"Top 5 features por importancia:")
    for feature, importance in feature_importance.head(5).items():
        logger.info(f"  {feature}: {importance:.4f}")
    
    return results

def recursive_feature_elimination(classification_data: pd.DataFrame, rfe_params: Dict) -> Dict[str, Any]:
    """
    Recursive Feature Elimination para selección de features.
    
    Args:
        classification_data: DataFrame preparado para clasificación
        rfe_params: Parámetros para RFE
        
    Returns:
        Diccionario con resultados de RFE
    """
    logger.info("Realizando Recursive Feature Elimination")
    
    # Preparar datos
    numeric_cols = classification_data.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(['target', 'rain_tomorrow_binary'], errors='ignore')
    
    X = classification_data[numeric_cols].fillna(0)
    y = classification_data['target']
    
    # Configurar estimador
    estimator = RandomForestClassifier(
        n_estimators=rfe_params.get('n_estimators', 50),
        random_state=42
    )
    
    # Configurar RFE
    n_features = rfe_params.get('n_features_to_select', 10)
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    
    # Ajustar RFE
    rfe.fit(X, y)
    
    # Obtener features seleccionadas
    selected_features = X.columns[rfe.support_].tolist()
    feature_ranking = pd.Series(rfe.ranking_, index=X.columns).sort_values()
    
    results = {
        'selected_features': selected_features,
        'feature_ranking': feature_ranking.to_dict(),
        'n_features_selected': len(selected_features),
        'rfe_support': rfe.support_.tolist()
    }
    
    logger.info(f"RFE seleccionó {len(selected_features)} features:")
    for feature in selected_features[:5]:  # Mostrar solo las primeras 5
        logger.info(f"  {feature}")
    
    return results

def select_best_features_classification(
    correlation_results: Dict[str, Any],
    mutual_info_results: Dict[str, Any],
    chi_square_results: Dict[str, Any],
    feature_importance_results: Dict[str, Any],
    rfe_results: Dict[str, Any],
    feature_selection_params: Dict
) -> Dict[str, Any]:
    """
    Selecciona las mejores features combinando múltiples métodos.
    
    Args:
        correlation_results: Resultados de correlación
        mutual_info_results: Resultados de información mutua
        chi_square_results: Resultados de chi-square
        feature_importance_results: Resultados de importancia
        rfe_results: Resultados de RFE
        feature_selection_params: Parámetros de selección
        
    Returns:
        Diccionario con features seleccionadas finales
    """
    logger.info("Seleccionando features finales combinando métodos")
    
    # Obtener top features de cada método
    top_corr = correlation_results['top_correlated_features'][:10]
    top_mi = mutual_info_results['top_mi_features'][:10]
    top_importance = feature_importance_results['top_important_features'][:10]
    top_rfe = rfe_results['selected_features']
    top_chi_square = chi_square_results.get('significant_features', [])[:10]
    
    # Contar votos por feature
    feature_votes = {}
    
    # Dar pesos a cada método según parámetros
    weights = feature_selection_params.get('method_weights', {
        'correlation': 1,
        'mutual_info': 2,
        'chi_square': 1,
        'feature_importance': 2,
        'rfe': 3
    })
    
    # Sumar votos ponderados
    for feature in top_corr:
        feature_votes[feature] = feature_votes.get(feature, 0) + weights['correlation']
    
    for feature in top_mi:
        feature_votes[feature] = feature_votes.get(feature, 0) + weights['mutual_info']
    
    for feature in top_chi_square:
        feature_votes[feature] = feature_votes.get(feature, 0) + weights['chi_square']
    
    for feature in top_importance:
        feature_votes[feature] = feature_votes.get(feature, 0) + weights['feature_importance']
    
    for feature in top_rfe:
        feature_votes[feature] = feature_votes.get(feature, 0) + weights['rfe']
    
    # Ordenar por votos
    sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
    
    # Seleccionar top N features
    n_features = feature_selection_params.get('n_final_features', 15)
    selected_features = [feature for feature, votes in sorted_features[:n_features]]
    
    results = {
        'selected_features': selected_features,
        'feature_votes': dict(sorted_features),
        'selection_summary': {
            'total_candidates': len(feature_votes),
            'final_selected': len(selected_features),
            'selection_threshold': sorted_features[n_features-1][1] if len(sorted_features) >= n_features else 0
        },
        'method_contributions': {
            'correlation': top_corr,
            'mutual_info': top_mi,
            'chi_square': top_chi_square,
            'feature_importance': top_importance,
            'rfe': top_rfe
        }
    }
    
    logger.info(f"Features seleccionadas finales ({len(selected_features)}):")
    for i, feature in enumerate(selected_features[:10], 1):
        votes = feature_votes[feature]
        logger.info(f"  {i}. {feature} (votos: {votes})")
    
    return results

def create_feature_selection_report_classification(
    correlation_results: Dict[str, Any],
    mutual_info_results: Dict[str, Any],
    chi_square_results: Dict[str, Any],
    feature_importance_results: Dict[str, Any],
    rfe_results: Dict[str, Any],
    selected_features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crea un reporte completo del proceso de selección de features.
    
    Returns:
        Diccionario con reporte completo
    """
    logger.info("Creando reporte de feature selection para clasificación")
    
    report = {
        'summary': {
            'total_methods_used': 5,
            'final_features_selected': len(selected_features['selected_features']),
            'selection_approach': 'ensemble_voting'
        },
        'method_results': {
            'correlation_analysis': {
                'top_features': correlation_results['top_correlated_features'][:5],
                'method_description': 'Correlación lineal con target binario (rain_tomorrow_binary)'
            },
            'mutual_information': {
                'top_features': mutual_info_results['top_mi_features'][:5],
                'method_description': 'Información mutua para clasificación'
            },
            'chi_square_test': {
                'significant_features': chi_square_results.get('significant_features', [])[:5],
                'method_description': 'Chi-square test para variables categóricas'
            },
            'feature_importance': {
                'top_features': feature_importance_results['top_important_features'][:5],
                'model_accuracy': feature_importance_results['model_accuracy'],
                'method_description': 'Random Forest feature importance'
            },
            'rfe': {
                'selected_features': rfe_results['selected_features'][:5],
                'total_selected': len(rfe_results['selected_features']),
                'method_description': 'Recursive Feature Elimination'
            }
        },
        'final_selection': {
            'selected_features': selected_features['selected_features'],
            'selection_rationale': 'Combinación ponderada de múltiples métodos',
            'feature_votes': selected_features['feature_votes']
        },
        'recommendations': [
            'Validar features seleccionadas con validación cruzada',
            'Considerar interacciones entre features',
            'Monitorear performance con diferentes subconjuntos',
            'Evaluar estabilidad de la selección con bootstrap'
        ]
    }
    
    logger.info("Reporte de feature selection completado")
    logger.info(f"Métodos utilizados: {report['summary']['total_methods_used']}")
    logger.info(f"Features finales: {report['summary']['final_features_selected']}")
    
    return report