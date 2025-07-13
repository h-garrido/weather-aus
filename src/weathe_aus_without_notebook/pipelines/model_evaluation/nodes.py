from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_regression_models(
    metrics_gaussian: Dict[str, float],
    metrics_gradient: Dict[str, float],
    metrics_ridge: Dict[str, float],
    metrics_lasso: Dict[str, float],
    metrics_svr: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate and compare all regression models.
    """
    logger.info("Evaluating regression models...")
    
    # Compile all metrics
    models_metrics = {
        'Gaussian NB': metrics_gaussian,
        'Gradient Boosting': metrics_gradient,
        'Ridge': metrics_ridge,
        'Lasso': metrics_lasso,
        'SVR': metrics_svr
    }
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(models_metrics).T
    metrics_df = metrics_df.round(4)
    
    # Find best model based on R2 score
    best_model = metrics_df['r2'].idxmax()
    
    # Create evaluation summary
    evaluation_summary = {
        'evaluation_date': datetime.now().isoformat(),
        'total_models': len(models_metrics),
        'best_model': best_model,
        'best_r2_score': float(metrics_df.loc[best_model, 'r2']),
        'models_summary': {
            model: {
                'r2_score': float(metrics[model]['r2']),
                'mse': float(metrics[model]['mse']),
                'mae': float(metrics[model]['mae'])
            }
            for model, metrics in models_metrics.items()
        },
        'metrics_statistics': {
            'r2': {
                'mean': float(metrics_df['r2'].mean()),
                'std': float(metrics_df['r2'].std()),
                'min': float(metrics_df['r2'].min()),
                'max': float(metrics_df['r2'].max())
            },
            'mse': {
                'mean': float(metrics_df['mse'].mean()),
                'std': float(metrics_df['mse'].std()),
                'min': float(metrics_df['mse'].min()),
                'max': float(metrics_df['mse'].max())
            }
        }
    }
    
    logger.info(f"Best regression model: {best_model} with R2 score: {evaluation_summary['best_r2_score']}")
    
    return metrics_df, evaluation_summary


def evaluate_classification_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Classification Model"
) -> Dict[str, Any]:
    """
    Evaluate a single classification model.
    """
    logger.info(f"Evaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted')),
        'recall': float(recall_score(y_test, y_pred, average='weighted')),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
    }
    
    # Try to calculate AUC-ROC if model supports probability predictions
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics['auc_roc'] = float(roc_auc_score(y_test, y_proba))
    except:
        logger.warning(f"Could not calculate AUC-ROC for {model_name}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    evaluation = {
        'model_name': model_name,
        'evaluation_date': datetime.now().isoformat(),
        'metrics': metrics,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'test_set_size': len(y_test),
        'class_distribution': {
            'true': dict(pd.Series(y_test).value_counts()),
            'predicted': dict(pd.Series(y_pred).value_counts())
        }
    }
    
    return evaluation


def evaluate_all_classification_models(
    classification_data: pd.DataFrame,
    classification_model: Any = None,
    bayes_model: Any = None
) -> Dict[str, Any]:
    """
    Evaluate all available classification models.
    """
    logger.info("Evaluating all classification models...")
    
    # Prepare test data
    X = classification_data.drop('RainTomorrow', axis=1)
    y = classification_data['RainTomorrow']
    
    # Split data (using last 20% as test set)
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    evaluations = {}
    
    # Evaluate each available model
    if classification_model is not None:
        eval_result = evaluate_classification_model(
            classification_model, X_test, y_test, "Main Classification Model"
        )
        evaluations['main_classification'] = eval_result
    
    if bayes_model is not None:
        eval_result = evaluate_classification_model(
            bayes_model, X_test, y_test, "Naive Bayes"
        )
        evaluations['naive_bayes'] = eval_result
    
    # Create summary
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'total_models_evaluated': len(evaluations),
        'test_set_size': len(y_test),
        'models_performance': {
            name: {
                'accuracy': eval['metrics']['accuracy'],
                'f1_score': eval['metrics']['f1_score']
            }
            for name, eval in evaluations.items()
        }
    }
    
    # Find best model
    if evaluations:
        best_model = max(
            evaluations.items(),
            key=lambda x: x[1]['metrics']['accuracy']
        )[0]
        summary['best_model'] = best_model
        summary['best_accuracy'] = evaluations[best_model]['metrics']['accuracy']
    
    return summary


def generate_evaluation_visualizations(
    regression_metrics_df: pd.DataFrame,
    classification_summary: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate visualization plots for model evaluation.
    """
    logger.info("Generating evaluation visualizations...")
    
    visualization_paths = {}
    
    # 1. Regression models comparison
    plt.figure(figsize=(12, 6))
    
    # R2 scores
    plt.subplot(1, 2, 1)
    regression_metrics_df['r2'].plot(kind='bar', color='skyblue')
    plt.title('R² Scores by Model')
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # MSE comparison
    plt.subplot(1, 2, 2)
    regression_metrics_df['mse'].plot(kind='bar', color='coral')
    plt.title('MSE by Model')
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    regression_plot_path = 'data/08_reporting/regression_models_evaluation.png'
    plt.savefig(regression_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths['regression_evaluation'] = regression_plot_path
    
    # 2. Classification models comparison (if available)
    if classification_summary.get('models_performance'):
        models_perf = classification_summary['models_performance']
        
        plt.figure(figsize=(10, 6))
        models = list(models_perf.keys())
        accuracies = [models_perf[m]['accuracy'] for m in models]
        f1_scores = [models_perf[m]['f1_score'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy', color='lightgreen')
        plt.bar(x + width/2, f1_scores, width, label='F1 Score', color='lightcoral')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Classification Models Performance')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        classification_plot_path = 'data/08_reporting/classification_models_evaluation.png'
        plt.savefig(classification_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['classification_evaluation'] = classification_plot_path
    
    logger.info(f"Generated {len(visualization_paths)} visualizations")
    return visualization_paths


def create_final_evaluation_report(
    regression_summary: Dict[str, Any],
    classification_summary: Dict[str, Any],
    visualization_paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create the final comprehensive evaluation report.
    """
    logger.info("Creating final evaluation report...")
    
    report = {
        'report_metadata': {
            'creation_date': datetime.now().isoformat(),
            'report_type': 'Model Evaluation Summary',
            'version': '1.0'
        },
        'regression_evaluation': regression_summary,
        'classification_evaluation': classification_summary,
        'visualizations': visualization_paths,
        'overall_summary': {
            'total_models_evaluated': (
                regression_summary.get('total_models', 0) + 
                classification_summary.get('total_models_evaluated', 0)
            ),
            'best_regression_model': regression_summary.get('best_model'),
            'best_classification_model': classification_summary.get('best_model'),
            'evaluation_completed': True
        }
    }
    
    logger.info("Model evaluation completed successfully")
    return report