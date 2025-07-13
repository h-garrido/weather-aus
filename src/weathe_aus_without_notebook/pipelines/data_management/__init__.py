"""
Data Management Pipeline Package
===============================
Unified data processing pipeline for MLOps automation.
"""

from .pipeline import (
    create_data_management_pipeline,
    create_legacy_compatibility_pipeline, 
    create_full_data_management_pipeline
)

__all__ = [
    "create_data_management_pipeline",
    "create_legacy_compatibility_pipeline",
    "create_full_data_management_pipeline"
]