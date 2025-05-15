"""
Este módulo contiene nodos para la evaluación comparativa de los modelos de regresión.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def collect_model_metrics(metrics_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Recolecta métricas de diferentes modelos y las organiza en un DataFrame.
    
    Args:
        metrics_paths: Diccionario con nombres de modelos y rutas a sus archivos de métricas
        
    Returns:
        metrics_df: DataFrame con las métricas de todos los modelos
    """
    logger.info("Recolectando métricas de todos los modelos de regresión...")
    
    all_metrics = []
    
    # Métricas comunes a extraer de todos los modelos
    common_metrics = ['mse', 'rmse', 'mae', 'r2', 'cv_rmse']
    
    for model_name, path in metrics_paths.items():
        try:
            with open(path, 'r') as f:
                metrics = json.load(f)
                
            # Extraer métricas comunes
            model_metrics = {'model': model_name}
            for metric in common_metrics:
                if metric in metrics:
                    model_metrics[metric] = metrics[metric]
                else:
                    model_metrics[metric] = None
                    
            # Agregar información adicional específica de cada modelo
            if model_name == 'random_forest' or model_name == 'gradient_boosting':
                if 'top_features' in metrics:
                    model_metrics['top_features'] = metrics['top_features']
                    
            elif model_name == 'lasso' or model_name == 'ridge':
                if 'top_coefficients' in metrics or 'top_non_zero_coefficients' in metrics:
                    model_metrics['top_coefficients'] = metrics.get('top_coefficients') or metrics.get('top_non_zero_coefficients')
                    
            elif model_name == 'svr':
                if 'n_support_vectors' in metrics:
                    model_metrics['n_support_vectors'] = metrics['n_support_vectors']
                
            elif model_name == 'gaussian_nb_regressor':
                # Métricas específicas de Gaussian Naive Bayes
                pass
                
            elif model_name == 'knn_regressor':
                # Métricas específicas de KNN
                pass
                
            all_metrics.append(model_metrics)
            logger.info(f"Métricas recolectadas para el modelo {model_name}")
            
        except Exception as e:
            logger.error(f"Error al procesar métricas del modelo {model_name}: {str(e)}")
            continue
    
    # Crear DataFrame con todas las métricas
    metrics_df = pd.DataFrame(all_metrics)
    
    # Ordenar por R2 score (de mayor a menor)
    if 'r2' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('r2', ascending=False).reset_index(drop=True)
    
    return metrics_df


def create_comparative_visualizations(
    metrics_df: pd.DataFrame,
    output_dir: str
) -> Dict[str, str]:
    """
    Crea visualizaciones comparativas de los diferentes modelos.
    
    Args:
        metrics_df: DataFrame con las métricas de todos los modelos
        output_dir: Directorio donde guardar las visualizaciones
        
    Returns:
        paths: Diccionario con rutas a las visualizaciones generadas
    """
    logger.info("Creando visualizaciones comparativas de los modelos...")
    
    # Asegurar que el directorio existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Definir paleta de colores para consistencia
    palette = sns.color_palette("viridis", len(metrics_df))
    
    # Crear diferentes visualizaciones
    plot_paths = {}
    
    # 1. Comparación de R2 scores
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='r2', data=metrics_df, palette=palette)
    plt.title('Comparación de R² Score entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    r2_plot_path = os.path.join(output_dir, 'r2_comparison.png')
    plt.savefig(r2_plot_path)
    plt.close()
    plot_paths['r2_comparison'] = r2_plot_path
    
    # 2. Comparación de RMSE
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='rmse', data=metrics_df, palette=palette)
    plt.title('Comparación de RMSE entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('RMSE (Root Mean Squared Error)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    rmse_plot_path = os.path.join(output_dir, 'rmse_comparison.png')
    plt.savefig(rmse_plot_path)
    plt.close()
    plot_paths['rmse_comparison'] = rmse_plot_path
    
    # 3. Comparación de MAE
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='mae', data=metrics_df, palette=palette)
    plt.title('Comparación de MAE entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('MAE (Mean Absolute Error)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    mae_plot_path = os.path.join(output_dir, 'mae_comparison.png')
    plt.savefig(mae_plot_path)
    plt.close()
    plot_paths['mae_comparison'] = mae_plot_path
    
    # 4. Comparación de métricas múltiples por modelo (gráfico de radar o araña)
    if all(metric in metrics_df.columns for metric in ['r2', 'rmse', 'mae']):
        # Normalizar las métricas para el gráfico de radar
        metrics_radar = metrics_df.copy()
        
        # Para r2, valores más altos son mejores (escala 0-1, normalmente)
        # Para rmse/mae, valores más bajos son mejores, así que invertimos la escala
        if 'rmse' in metrics_radar.columns:
            max_rmse = metrics_radar['rmse'].max()
            metrics_radar['rmse_normalized'] = 1 - (metrics_radar['rmse'] / max_rmse)
        
        if 'mae' in metrics_radar.columns:
            max_mae = metrics_radar['mae'].max()
            metrics_radar['mae_normalized'] = 1 - (metrics_radar['mae'] / max_mae)
        
        # r2 ya está en escala 0-1 (mayor es mejor)
        
        # Gráfico de radar
        # (Esta visualización sería más compleja y podría requerir bibliotecas adicionales)
        # Para simplificar, haremos un gráfico de paralelas
        plt.figure(figsize=(12, 8))
        
        # Preparar datos para gráfico de coordenadas paralelas
        parallel_cols = ['model', 'r2']
        if 'rmse_normalized' in metrics_radar.columns:
            parallel_cols.append('rmse_normalized')
        if 'mae_normalized' in metrics_radar.columns:
            parallel_cols.append('mae_normalized')
        
        pd.plotting.parallel_coordinates(
            metrics_radar[parallel_cols], 'model', 
            colormap=plt.cm.viridis, 
            alpha=0.7
        )
        plt.title('Comparación de Múltiples Métricas por Modelo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        multi_plot_path = os.path.join(output_dir, 'multi_metric_comparison.png')
        plt.savefig(multi_plot_path)
        plt.close()
        plot_paths['multi_metric_comparison'] = multi_plot_path
    
    # 5. Heatmap con todas las métricas
    plt.figure(figsize=(10, 8))
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    heatmap_data = metrics_df.set_index('model')[numeric_cols]
    
    # Normalizar datos para mejor visualización
    heatmap_normalized = heatmap_data.copy()
    for col in heatmap_normalized.columns:
        if col != 'r2':  # Para RMSE, MAE, menor es mejor
            max_val = heatmap_normalized[col].max()
            if max_val > 0:
                heatmap_normalized[col] = 1 - (heatmap_normalized[col] / max_val)
    
    sns.heatmap(heatmap_normalized, annot=True, cmap='viridis', linewidths=.5)
    plt.title('Comparación de Métricas entre Modelos (Normalizado)')
    plt.tight_layout()
    heatmap_plot_path = os.path.join(output_dir, 'metrics_heatmap.png')
    plt.savefig(heatmap_plot_path)
    plt.close()
    plot_paths['metrics_heatmap'] = heatmap_plot_path
    
    logger.info(f"Se generaron {len(plot_paths)} visualizaciones comparativas")
    return plot_paths


def generate_model_ranking_report(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un reporte con el ranking de los modelos basado en diferentes métricas.
    
    Args:
        metrics_df: DataFrame con las métricas de todos los modelos
        
    Returns:
        report: Diccionario con el reporte de ranking
    """
    logger.info("Generando reporte de ranking de modelos...")
    
    report = {
        'best_models': {},
        'rankings': {},
        'summary': {}
    }
    
    # Métricas para rankear
    ranking_metrics = {
        'r2': 'higher',  # Mayor es mejor
        'rmse': 'lower', # Menor es mejor
        'mae': 'lower',  # Menor es mejor
        'cv_rmse': 'lower' # Menor es mejor
    }
    
    # Encontrar los mejores modelos por cada métrica
    for metric, direction in ranking_metrics.items():
        if metric in metrics_df.columns:
            if direction == 'higher':
                best_idx = metrics_df[metric].idxmax()
            else:
                best_idx = metrics_df[metric].idxmin()
                
            best_model = metrics_df.loc[best_idx, 'model']
            best_score = metrics_df.loc[best_idx, metric]
            
            report['best_models'][metric] = {
                'model': best_model,
                'score': best_score
            }
    
    # Crear rankings completos
    for metric, direction in ranking_metrics.items():
        if metric in metrics_df.columns:
            if direction == 'higher':
                sorted_df = metrics_df.sort_values(by=metric, ascending=False)
            else:
                sorted_df = metrics_df.sort_values(by=metric, ascending=True)
                
            report['rankings'][metric] = sorted_df[['model', metric]].values.tolist()
    
    # Crear un ranking general basado en la posición promedio en todas las métricas
    ranking_positions = pd.DataFrame({'model': metrics_df['model']})
    
    for metric, direction in ranking_metrics.items():
        if metric in metrics_df.columns:
            if direction == 'higher':
                sorted_df = metrics_df.sort_values(by=metric, ascending=False)
            else:
                sorted_df = metrics_df.sort_values(by=metric, ascending=True)
                
            positions = sorted_df[['model']].reset_index(drop=True).reset_index()
            positions = positions.rename(columns={'index': f'{metric}_rank'})
            
            ranking_positions = ranking_positions.merge(positions, on='model')
    
    # Calcular ranking promedio
    rank_columns = [col for col in ranking_positions.columns if col.endswith('_rank')]
    ranking_positions['avg_rank'] = ranking_positions[rank_columns].mean(axis=1)
    ranking_positions = ranking_positions.sort_values('avg_rank').reset_index(drop=True)
    
    # Preparar ranking general
    general_ranking = []
    for idx, row in ranking_positions.iterrows():
        general_ranking.append({
            'position': idx + 1,
            'model': row['model'],
            'avg_rank': row['avg_rank']
        })
    
    report['summary']['general_ranking'] = general_ranking
    report['summary']['best_overall_model'] = general_ranking[0]['model']
    
    # Recomendaciones basadas en los resultados
    report['summary']['recommendations'] = [
        f"El modelo con mejor desempeño general es: {general_ranking[0]['model']}",
        f"El modelo con mejor R²: {report['best_models'].get('r2', {}).get('model', 'N/A')}",
        f"El modelo con mejor RMSE: {report['best_models'].get('rmse', {}).get('model', 'N/A')}"
    ]
    
    # Agregar recomendaciones específicas basadas en características del dataset
    if 'random_forest' in metrics_df['model'].values and 'linear_regression' in metrics_df['model'].values:
        # Calcular ratio de rendimiento entre modelos no lineales vs lineales
        rf_idx = metrics_df[metrics_df['model'] == 'random_forest'].index
        lr_idx = metrics_df[metrics_df['model'] == 'linear_regression'].index
        
        if (len(rf_idx) > 0 and len(lr_idx) > 0 and 
            'r2' in metrics_df.columns and 
            metrics_df.loc[rf_idx[0], 'r2'] > 0 and 
            metrics_df.loc[lr_idx[0], 'r2'] > 0):
            
            ratio = metrics_df.loc[rf_idx[0], 'r2'] / metrics_df.loc[lr_idx[0], 'r2']
            
            if ratio > 1.2:
                report['summary']['recommendations'].append(
                    "Los modelos no lineales (como Random Forest) superan significativamente a los lineales, "
                    "lo que sugiere relaciones complejas entre las variables que podrían beneficiarse de "
                    "más características de ingeniería o modelos más avanzados."
                )
            elif ratio < 0.9:
                report['summary']['recommendations'].append(
                    "Los modelos lineales tienen un rendimiento comparable o superior a los no lineales, "
                    "lo que sugiere que las relaciones en los datos son mayormente lineales y que los modelos "
                    "más simples podrían ser preferibles."
                )
    
    logger.info("Reporte de ranking de modelos generado exitosamente")
    return report


def create_evaluation_summary(
    metrics_df: pd.DataFrame,
    plot_paths: Dict[str, str],
    ranking_report: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crea un resumen de evaluación general con todas las métricas, 
    visualizaciones y recomendaciones.
    
    Args:
        metrics_df: DataFrame con las métricas de todos los modelos
        plot_paths: Diccionario con rutas a las visualizaciones generadas
        ranking_report: Reporte de ranking de modelos
        
    Returns:
        summary: Diccionario con el resumen completo de evaluación
    """
    logger.info("Creando resumen de evaluación general...")
    
    summary = {
        'metrics_table': metrics_df.to_dict(orient='records'),
        'visualizations': plot_paths,
        'rankings': ranking_report['rankings'],
        'best_models': ranking_report['best_models'],
        'general_ranking': ranking_report['summary']['general_ranking'],
        'best_overall_model': ranking_report['summary']['best_overall_model'],
        'recommendations': ranking_report['summary']['recommendations'],
        'evaluation_timestamp': pd.Timestamp.now().isoformat()
    }
    
    logger.info("Resumen de evaluación general creado exitosamente")
    return summary