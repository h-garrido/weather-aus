"""
Classification Evaluation Pipeline Package
==========================================
Comprehensive evaluation and comparison of classification models.
"""

from .pipeline import (
    create_pipeline,
    create_individual_model_evaluation_pipeline
)

__all__ = [
    "create_pipeline",
    "create_individual_model_evaluation_pipeline"
]