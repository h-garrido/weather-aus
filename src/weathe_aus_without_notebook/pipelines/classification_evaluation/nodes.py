import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import json

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ==========================================
# 1. SINGLE MODEL EVALUATION
# ==========================================

def evaluate_single_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single classification model with comprehensive metrics.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model for reporting
        parameters: Configuration parameters
    
    Returns:
        Dict with evaluation metrics and predictions
    """
    logger.info(f"🔍 Evaluating {model_name} classifier...")
    
    try:
        # Extraer el modelo si viene en un diccionario
        if isinstance(model, dict):
            logger.info(f"📦 Model {model_name} is a dictionary with keys: {list(model.keys())}")
            
            # Buscar el modelo real
            actual_model = None
            features_used = None
            scaler = None
            
            # Extraer modelo
            for key in ['model', 'classifier', 'estimator', 'trained_model']:
                if key in model:
                    actual_model = model[key]
                    logger.info(f"✅ Found model in key: '{key}'")
                    break
            
            # Extraer características usadas
            if 'features_used' in model:
                features_used = model['features_used']
                logger.info(f"📋 Found features used: {features_used}")
            
            # Extraer scaler si existe
            if 'scaler' in model:
                scaler = model['scaler']
                logger.info(f"🔧 Found scaler for {model_name}")
            
            if actual_model is None:
                raise ValueError(f"No model found in dictionary for {model_name}")
                
            # IMPORTANTE: Filtrar X_test para usar solo las características correctas
            if features_used is not None:
                logger.info(f"🔧 Filtering X_test from {X_test.shape[1]} to {len(features_used)} features")
                X_test_filtered = X_test[features_used].copy()
                
                # Aplicar scaler si existe
                if scaler is not None:
                    logger.info(f"⚡ Applying scaler to features")
                    X_test_filtered = pd.DataFrame(
                        scaler.transform(X_test_filtered),
                        columns=features_used,
                        index=X_test_filtered.index
                    )
            else:
                X_test_filtered = X_test
                
            model = actual_model
            X_test = X_test_filtered
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_positive = y_pred_proba[:, 1]
            else:
                y_pred_proba_positive = y_pred_proba
        except:
            y_pred_proba_positive = None
            logger.warning(f"⚠️ {model_name} doesn't support predict_proba")

        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba_positive = y_pred_proba[:, 1]  # Probability of positive class
            else:
                y_pred_proba_positive = y_pred_proba
        except:
            y_pred_proba_positive = None
            logger.warning(f"⚠️ {model_name} doesn't support predict_proba")
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Calculate ROC AUC if probabilities available
        try:
            if y_pred_proba_positive is not None and len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_pred_proba_positive)
                avg_precision = average_precision_score(y_test, y_pred_proba_positive)
            else:
                roc_auc = None
                avg_precision = None
        except:
            roc_auc = None
            avg_precision = None
            logger.warning(f"⚠️ Could not calculate ROC AUC for {model_name}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Compile results
        results = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc) if roc_auc is not None else None,
                "average_precision": float(avg_precision) if avg_precision is not None else None
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
            "predictions": {
                "y_pred": y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                "y_true": y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                "y_pred_proba": y_pred_proba_positive.tolist() if y_pred_proba_positive is not None else None
            },
            "data_info": {
                "test_size": len(y_test),
                "n_features": X_test.shape[1],
                "class_distribution": y_test.value_counts().to_dict()
            }
        }
        
        logger.info(f"✅ {model_name} evaluation completed!")
        logger.info(f"📊 Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error evaluating {model_name}: {str(e)}")
        return {
            "model_name": model_name,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def create_classification_models_dict(
    logistic_regression_model=None,
    random_forest_classification_model=None,
    svm_classification_model=None,
    decision_tree_model=None,
    bayes_model=None,
    knn_classification_model=None,
    gradient_boosting_classification_model=None
) -> Dict[str, Any]:
    """
    Consolidate all classification models into a single dictionary.
    
    Returns:
        Dict with all available classification models
    """
    logger.info("📦 Creating classification models dictionary...")
    
    models_dict = {}
    
    # Add each model if it exists
    if logistic_regression_model is not None:
        models_dict["logistic_regression"] = logistic_regression_model
        
    if random_forest_classification_model is not None:
        models_dict["random_forest"] = random_forest_classification_model
        
    if svm_classification_model is not None:
        models_dict["svm"] = svm_classification_model
        
    if decision_tree_model is not None:
        models_dict["decision_tree"] = decision_tree_model
        
    if bayes_model is not None:
        models_dict["bayes"] = bayes_model
        
    if knn_classification_model is not None:
        models_dict["knn"] = knn_classification_model
        
    if gradient_boosting_classification_model is not None:
        models_dict["gradient_boosting"] = gradient_boosting_classification_model
    
    logger.info(f"✅ Created models dictionary with {len(models_dict)} models")
    return models_dict


# ==========================================
# 2. MULTIPLE MODEL COMPARISON
# ==========================================

def prepare_test_data_from_transformed(
    transformed_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extrae X_test e y_test del dataset transformado
    """
    # Obtener el porcentaje de test
    test_size = parameters.get("test_size", 0.2)
    
    # Separar características y target
    target_col = parameters.get("classification_target", "rain_tomorrow_binary")
    X = transformed_data.drop(columns=[target_col])
    y = transformed_data[target_col]
    
    # Tomar los últimos registros como test (o usar el mismo split que usaste antes)
    split_idx = int(len(transformed_data) * (1 - test_size))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    logger.info(f"📊 Prepared test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
    return X_test, y_test

def compare_classification_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare multiple classification models and generate comparison report.
    
    Args:
        models_dict: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: Test targets
        parameters: Configuration parameters
    
    Returns:
        Dict with comparison results and rankings
    """
    logger.info(f"📊 Comparing {len(models_dict)} classification models...")
    
    # Evaluate each model
    model_results = {}
    for model_name, model in models_dict.items():
        result = evaluate_single_classifier(model, X_test, y_test, model_name, parameters)
        model_results[model_name] = result
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in model_results.items():
        if "error" not in result:
            metrics = result["metrics"]
            comparison_data.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics.get("roc_auc"),
                "test_size": result["data_info"]["test_size"]
            })
    
    if not comparison_data:
        logger.error("❌ No models were successfully evaluated")
        return {"error": "No models successfully evaluated"}
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Ranking by different metrics
    rankings = {}
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        if metric in comparison_df.columns:
            valid_data = comparison_df.dropna(subset=[metric])
            if not valid_data.empty:
                rankings[metric] = valid_data.nlargest(len(valid_data), metric)["model"].tolist()
    
    # Best model overall (by F1 score)
    best_model_by_f1 = comparison_df.loc[comparison_df["f1_score"].idxmax(), "model"]
    
    # Model statistics
    model_stats = {
        "total_models": len(models_dict),
        "successful_evaluations": len(comparison_data),
        "failed_evaluations": len(models_dict) - len(comparison_data),
        "best_model_overall": best_model_by_f1,
        "metric_averages": {
            "accuracy": float(comparison_df["accuracy"].mean()),
            "precision": float(comparison_df["precision"].mean()),
            "recall": float(comparison_df["recall"].mean()),
            "f1_score": float(comparison_df["f1_score"].mean()),
        }
    }
    
    # Add ROC AUC average if available
    if comparison_df["roc_auc"].notna().any():
        model_stats["metric_averages"]["roc_auc"] = float(comparison_df["roc_auc"].mean())
    
    comparison_report = {
        "timestamp": datetime.now().isoformat(),
        "model_results": model_results,
        "comparison_table": comparison_df.to_dict('records'),
        "rankings": rankings,
        "model_statistics": model_stats,
        "best_models": {
            "by_accuracy": rankings.get("accuracy", [None])[0],
            "by_precision": rankings.get("precision", [None])[0],
            "by_recall": rankings.get("recall", [None])[0],
            "by_f1_score": rankings.get("f1_score", [None])[0],
            "by_roc_auc": rankings.get("roc_auc", [None])[0] if "roc_auc" in rankings else None,
            "overall_best": best_model_by_f1
        }
    }
    
    logger.info(f"🏆 Best model overall (F1): {best_model_by_f1}")
    logger.info(f"📈 Average F1 score: {model_stats['metric_averages']['f1_score']:.4f}")
    
    return comparison_report


# ==========================================
# 3. VISUALIZATION GENERATION
# ==========================================

def generate_classification_visualizations(
    comparison_report: Dict[str, Any],
    parameters: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate visualization plots for classification results.
    
    Args:
        comparison_report: Output from compare_classification_models
        parameters: Configuration parameters
    
    Returns:
        Dict with paths to saved visualization files
    """
    logger.info("📊 Generating classification visualizations...")
    
    try:
        # Extract data
        comparison_data = comparison_report["comparison_table"]
        if not comparison_data:
            logger.warning("⚠️ No data available for visualizations")
            return {}
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create visualizations directory
        viz_dir = "data/08_reporting/classification_visualizations/"
        import os
        os.makedirs(viz_dir, exist_ok=True)
        
        saved_plots = {}
        
        # 1. Metrics comparison bar plot
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
        
        x = np.arange(len(comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in comparison_df.columns:
                plt.bar(x + i*width, comparison_df[metric], width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Classification Models Performance Comparison')
        plt.xticks(x + width*1.5, comparison_df['model'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        metrics_plot_path = os.path.join(viz_dir, 'metrics_comparison.png')
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots["metrics_comparison"] = metrics_plot_path
        
        # 2. ROC AUC comparison (if available)
        if "roc_auc" in comparison_df.columns and comparison_df["roc_auc"].notna().any():
            plt.figure(figsize=(10, 6))
            valid_roc_data = comparison_df.dropna(subset=["roc_auc"])
            
            bars = plt.bar(valid_roc_data['model'], valid_roc_data['roc_auc'])
            plt.ylabel('ROC AUC Score')
            plt.title('ROC AUC Comparison Across Models')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            roc_plot_path = os.path.join(viz_dir, 'roc_auc_comparison.png')
            plt.savefig(roc_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots["roc_auc_comparison"] = roc_plot_path
        
        # 3. Model ranking heatmap
        rankings = comparison_report.get("rankings", {})
        if rankings:
            plt.figure(figsize=(10, 6))
            
            # Create ranking matrix
            models = comparison_df['model'].tolist()
            metrics = list(rankings.keys())
            ranking_matrix = np.zeros((len(models), len(metrics)))
            
            for j, metric in enumerate(metrics):
                for i, model in enumerate(models):
                    try:
                        rank = rankings[metric].index(model) + 1
                        ranking_matrix[i, j] = rank
                    except (ValueError, IndexError):
                        ranking_matrix[i, j] = len(models) + 1  # Worst rank for missing
            
            sns.heatmap(ranking_matrix, 
                       xticklabels=[m.replace('_', ' ').title() for m in metrics],
                       yticklabels=models,
                       annot=True, fmt='.0f', 
                       cmap='RdYlGn_r',
                       cbar_kws={'label': 'Rank (1=Best)'})
            
            plt.title('Model Rankings by Metric (Lower is Better)')
            plt.tight_layout()
            ranking_plot_path = os.path.join(viz_dir, 'model_rankings.png')
            plt.savefig(ranking_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_plots["model_rankings"] = ranking_plot_path
        
        logger.info(f"✅ Generated {len(saved_plots)} visualization plots")
        return saved_plots
        
    except Exception as e:
        logger.error(f"❌ Error generating visualizations: {str(e)}")
        return {"error": str(e)}


# ==========================================
# 4. EVALUATION REPORT GENERATION
# ==========================================

def generate_classification_report(
    comparison_report: Dict[str, Any],
    visualizations: Dict[str, str],
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive classification evaluation report.
    
    Args:
        comparison_report: Model comparison results
        visualizations: Paths to visualization files
        parameters: Configuration parameters
    
    Returns:
        Dict with comprehensive evaluation report
    """
    logger.info("📝 Generating comprehensive classification report...")
    
    # Executive summary
    best_models = comparison_report.get("best_models", {})
    model_stats = comparison_report.get("model_statistics", {})
    
    executive_summary = {
        "evaluation_date": datetime.now().isoformat(),
        "total_models_evaluated": model_stats.get("total_models", 0),
        "successful_evaluations": model_stats.get("successful_evaluations", 0),
        "overall_best_model": best_models.get("overall_best", "Unknown"),
        "average_performance": model_stats.get("metric_averages", {}),
        "key_insights": []
    }
    
    # Generate insights
    if model_stats.get("metric_averages"):
        avg_f1 = model_stats["metric_averages"].get("f1_score", 0)
        avg_accuracy = model_stats["metric_averages"].get("accuracy", 0)
        
        if avg_f1 > 0.8:
            executive_summary["key_insights"].append("Excellent model performance with F1 > 0.8")
        elif avg_f1 > 0.7:
            executive_summary["key_insights"].append("Good model performance with F1 > 0.7")
        else:
            executive_summary["key_insights"].append("Model performance may need improvement")
        
        if abs(avg_accuracy - avg_f1) > 0.1:
            executive_summary["key_insights"].append("Potential class imbalance detected")
    
    # Model recommendations
    recommendations = []
    if best_models.get("overall_best"):
        recommendations.append(f"Deploy {best_models['overall_best']} for production (best F1 score)")
    
    if best_models.get("by_roc_auc") and best_models["by_roc_auc"] != best_models.get("overall_best"):
        recommendations.append(f"Consider {best_models['by_roc_auc']} for probability-based decisions (best ROC AUC)")
    
    # Compile final report
    final_report = {
        "report_metadata": {
            "report_type": "classification_evaluation",
            "generated_at": datetime.now().isoformat(),
            "kedro_pipeline": "classification_evaluation"
        },
        "executive_summary": executive_summary,
        "detailed_results": comparison_report,
        "visualizations": visualizations,
        "recommendations": recommendations,
        "data_quality": {
            "test_samples": comparison_report.get("model_results", {}).get(list(comparison_report.get("model_results", {}).keys())[0], {}).get("data_info", {}).get("test_size") if comparison_report.get("model_results") else None,
            "feature_count": comparison_report.get("model_results", {}).get(list(comparison_report.get("model_results", {}).keys())[0], {}).get("data_info", {}).get("n_features") if comparison_report.get("model_results") else None
        }
    }
    
    logger.info("✅ Classification evaluation report generated successfully!")
    return final_report