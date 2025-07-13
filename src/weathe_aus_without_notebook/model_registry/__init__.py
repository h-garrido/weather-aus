"""
Model Registry Package
=====================

Central registry for managing ML models with versioning, metadata tracking,
and lineage management for the Weather Australia MLOps project.

This package provides:
- Model registration and versioning
- Performance tracking and comparison
- Lineage and metadata management
- Integration with Kedro pipelines
"""

from .registry import ModelRegistry
from .models import (
    ModelMetadata,
    ModelVersion,
    ModelComparison,
    ModelMetric,
    ModelHyperparameter,
    ModelLineage,
    ModelRegistryConfig,
    ModelDeploymentInfo,
    ModelStatus,
    ModelType
)
from .exceptions import (
    ModelRegistryError,
    ModelNotFoundError,
    DuplicateModelError,
    InvalidModelError,
    DatabaseConnectionError,
    ModelFileError,
    VersionError,
    MetricValidationError,
    HyperparameterError,
    LineageError,
    ModelComparisonError,
    ModelArchiveError,
    ConfigurationError
)

__version__ = "1.0.0"
__author__ = "Weather Australia MLOps Team"

# Package-level exports
__all__ = [
    # Core classes
    "ModelRegistry",
    
    # Data models
    "ModelMetadata",
    "ModelVersion", 
    "ModelComparison",
    "ModelMetric",
    "ModelHyperparameter",
    "ModelLineage",
    "ModelRegistryConfig",
    "ModelDeploymentInfo",
    
    # Enums
    "ModelStatus",
    "ModelType",
    
    # Exceptions
    "ModelRegistryError",
    "ModelNotFoundError",
    "DuplicateModelError",
    "InvalidModelError",
    "DatabaseConnectionError",
    "ModelFileError",
    "VersionError",
    "MetricValidationError",
    "HyperparameterError",
    "LineageError",
    "ModelComparisonError",
    "ModelArchiveError",
    "ConfigurationError",
]

# Package metadata
PACKAGE_INFO = {
    "name": "model_registry",
    "version": __version__,
    "description": "Model Registry for Weather Australia MLOps",
    "author": __author__,
    "license": "MIT",
    "repository": "weather-australia-mlops",
    "supported_python_versions": ["3.8", "3.9", "3.10", "3.11"],
    "dependencies": [
        "pandas>=1.3.0",
        "psycopg2-binary>=2.9.0",
        "sqlalchemy>=1.4.0",
        "kedro>=0.18.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.0.0"
    ]
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO.copy()

def get_version():
    """Get package version."""
    return __version__