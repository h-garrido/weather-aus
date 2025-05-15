"""
Este módulo contiene la definición del pipeline para la evaluación comparativa
de los modelos de regresión.
"""

from kedro.pipeline import Pipeline, node
from typing import Dict

from .nodes import (
    collect_model_metrics,
    create_comparative_visualizations,
    generate_model_ranking_report,
    create_evaluation_summary
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea un pipeline para la evaluación comparativa de los modelos de regresión.
    
    Args:
        **kwargs: Argumentos de palabras clave adicionales
        
    Returns:
        pipeline: Pipeline de Kedro para evaluación de regresión
    """
    return Pipeline(
        [
            node(
                func=collect_model_metrics,
                inputs="params:model_metrics_paths",
                outputs="regression_models_metrics",
                name="collect_model_metrics_node",
            ),
            node(
                func=create_comparative_visualizations,
                inputs=[
                    "regression_models_metrics", 
                    "params:visualization_output_dir"
                ],
                outputs="regression_visualization_paths",
                name="create_visualizations_node",
            ),
            node(
                func=generate_model_ranking_report,
                inputs="regression_models_metrics",
                outputs="regression_models_ranking",
                name="generate_ranking_report_node",
            ),
            node(
                func=create_evaluation_summary,
                inputs=[
                    "regression_models_metrics",
                    "regression_visualization_paths",
                    "regression_models_ranking"
                ],
                outputs="regression_evaluation_summary",
                name="create_evaluation_summary_node",
            ),
        ]
    )