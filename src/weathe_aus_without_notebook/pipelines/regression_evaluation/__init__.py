"""
Este es un paquete para la evaluación comparativa de modelos de regresión en Kedro.
"""

from .pipeline import create_pipeline
from .nodes import (
    collect_model_metrics,
    create_comparative_visualizations,
    generate_model_ranking_report,
    create_evaluation_summary
)

__all__ = [
    "create_pipeline",
    "collect_model_metrics",
    "create_comparative_visualizations",
    "generate_model_ranking_report",
    "create_evaluation_summary",
]