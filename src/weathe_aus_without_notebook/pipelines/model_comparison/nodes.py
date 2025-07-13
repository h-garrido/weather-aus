import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_all_regression_metrics(
    linear_metrics: Dict,
    random_forest_metrics: Dict,
    knn_metrics: Dict,
    lasso_metrics: Dict,
    ridge_metrics: Dict,
    svr_metrics: Dict,
    gaussian_nb_metrics: Dict,
    gradient_boosting_metrics: Dict
) -> pd.DataFrame:
    """
    Carga y combina todas las métricas de regresión
    """
    logger.info("📊 Loading all regression metrics...")
    
    metrics_dict = {
        "Linear Regression": linear_metrics,
        "Random Forest": random_forest_metrics,
        "KNN": knn_metrics,
        "Lasso": lasso_metrics,
        "Ridge": ridge_metrics,
        "SVR": svr_metrics,
        "Gaussian NB": gaussian_nb_metrics,
        "Gradient Boosting": gradient_boosting_metrics
    }
    
    # Convertir a DataFrame
    df_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    # Ordenar por R2 descendente
    df_metrics = df_metrics.sort_values('r2', ascending=False)
    
    logger.info(f"✅ Loaded metrics for {len(df_metrics)} regression models")
    return df_metrics

def compare_regression_models(metrics_df: pd.DataFrame) -> Dict:
    """
    Compara modelos de regresión y genera rankings
    """
    logger.info("🔍 Comparing regression models...")
    
    # Rankings por métrica
    rankings = {
        "r2_ranking": metrics_df.sort_values('r2', ascending=False).index.tolist(),
        "rmse_ranking": metrics_df.sort_values('rmse', ascending=True).index.tolist(),
        "mae_ranking": metrics_df.sort_values('mae', ascending=True).index.tolist()
    }
    
    # Mejor modelo overall (basado en R2)
    best_model = rankings["r2_ranking"][0]
    
    # Estadísticas summary
    summary_stats = {
        "best_model": best_model,
        "best_r2": float(metrics_df.loc[best_model, 'r2']),
        "best_rmse": float(metrics_df.loc[rankings["rmse_ranking"][0], 'rmse']),
        "best_mae": float(metrics_df.loc[rankings["mae_ranking"][0], 'mae']),
        "average_r2": float(metrics_df['r2'].mean()),
        "average_rmse": float(metrics_df['rmse'].mean()),
        "average_mae": float(metrics_df['mae'].mean())
    }
    
    comparison_results = {
        "rankings": rankings,
        "summary_stats": summary_stats,
        "full_metrics": metrics_df.to_dict()
    }
    
    logger.info(f"🏆 Best model overall: {best_model} (R²: {summary_stats['best_r2']:.4f})")
    
    return comparison_results

def create_model_comparison_report(
    regression_comparison: Dict,
    # classification_comparison: Dict  # Para cuando lo implementes
) -> Dict:
    """
    Crea reporte unificado de comparación
    """
    logger.info("📝 Creating model comparison report...")
    
    report = {
        "regression": {
            "best_model": regression_comparison["summary_stats"]["best_model"],
            "rankings": regression_comparison["rankings"],
            "summary": regression_comparison["summary_stats"],
            "detailed_metrics": regression_comparison["full_metrics"]
        },
        # "classification": classification_comparison  # Para después
    }
    
    logger.info("✅ Model comparison report created")
    return report

def generate_comparison_visualizations(
    metrics_df: pd.DataFrame,
    comparison_results: Dict
) -> Dict:
    """
    Genera visualizaciones comparativas
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("📊 Generating comparison visualizations...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    visualizations = {}
    
    # 1. Comparación de R² por modelo
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.sort_values('r2', ascending=True).plot(
        kind='barh', 
        y='r2', 
        ax=ax,
        color='skyblue',
        edgecolor='navy'
    )
    ax.set_xlabel('R² Score')
    ax.set_title('Model Comparison - R² Scores', fontsize=16, fontweight='bold')
    ax.set_xlim(0, 1)
    
    # Agregar valores en las barras
    for i, (idx, row) in enumerate(metrics_df.sort_values('r2', ascending=True).iterrows()):
        ax.text(row['r2'] + 0.01, i, f'{row["r2"]:.3f}', va='center')
    
    plt.tight_layout()
    visualizations['r2_comparison'] = fig
    
    # 2. Comparación múltiple de métricas
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # R²
    metrics_df.sort_values('r2', ascending=False).plot(
        kind='bar', y='r2', ax=axes[0], color='green', alpha=0.7
    )
    axes[0].set_title('R² Score (Higher is Better)')
    axes[0].set_ylabel('R²')
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # RMSE
    metrics_df.sort_values('rmse', ascending=True).plot(
        kind='bar', y='rmse', ax=axes[1], color='orange', alpha=0.7
    )
    axes[1].set_title('RMSE (Lower is Better)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # MAE
    metrics_df.sort_values('mae', ascending=True).plot(
        kind='bar', y='mae', ax=axes[2], color='red', alpha=0.7
    )
    axes[2].set_title('MAE (Lower is Better)')
    axes[2].set_ylabel('MAE')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Regression Models - Metrics Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    visualizations['metrics_comparison'] = fig
    
    logger.info(f"✅ Generated {len(visualizations)} visualizations")
    
    return visualizations