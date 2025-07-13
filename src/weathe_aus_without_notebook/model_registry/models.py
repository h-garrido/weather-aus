"""
Data models for Model Registry
=============================

Dataclasses and Pydantic models for type safety and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class ModelStatus(Enum):
    """Model status enumeration."""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    FAILED = "failed"


class ModelType(Enum):
    """Model type enumeration."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"
    OTHER = "other"


@dataclass
class ModelMetadata:
    """
    Metadata for a registered model.
    """
    id: int
    model_name: str
    model_type: str
    algorithm: str
    version: str
    model_hash: str
    created_at: datetime
    status: str = "active"
    created_by: str = "kedro_pipeline"
    training_data_hash: Optional[str] = None
    test_data_hash: Optional[str] = None
    feature_set_version: Optional[str] = None
    model_path: Optional[str] = None
    artifacts_path: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'algorithm': self.algorithm,
            'version': self.version,
            'model_hash': self.model_hash,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'status': self.status,
            'created_by': self.created_by,
            'training_data_hash': self.training_data_hash,
            'test_data_hash': self.test_data_hash,
            'feature_set_version': self.feature_set_version,
            'model_path': self.model_path,
            'artifacts_path': self.artifacts_path,
            'description': self.description,
            'tags': self.tags
        }


@dataclass
class ModelMetric:
    """
    Individual model metric.
    """
    model_id: int
    metric_name: str
    metric_value: float
    dataset_type: str  # 'train', 'validation', 'test'
    created_at: datetime


@dataclass
class ModelHyperparameter:
    """
    Model hyperparameter.
    """
    model_id: int
    parameter_name: str
    parameter_value: Any
    parameter_type: str
    created_at: datetime


@dataclass
class ModelLineage:
    """
    Model lineage information.
    """
    model_id: int
    pipeline_name: str
    pipeline_version: Optional[str] = None
    node_name: Optional[str] = None
    input_datasets: List[str] = field(default_factory=list)
    parent_model_ids: List[int] = field(default_factory=list)
    git_commit_hash: Optional[str] = None
    kedro_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelVersion:
    """
    Complete model version information.
    """
    metadata: ModelMetadata
    metrics: List[ModelMetric] = field(default_factory=list)
    hyperparameters: List[ModelHyperparameter] = field(default_factory=list)
    lineage: Optional[ModelLineage] = None
    
    def get_metric(self, metric_name: str, dataset_type: str = 'test') -> Optional[float]:
        """Get specific metric value."""
        for metric in self.metrics:
            if metric.metric_name == metric_name and metric.dataset_type == dataset_type:
                return metric.metric_value
        return None
    
    def get_hyperparameter(self, param_name: str) -> Any:
        """Get specific hyperparameter value."""
        for param in self.hyperparameters:
            if param.parameter_name == param_name:
                return param.parameter_value
        return None


@dataclass
class ModelComparison:
    """
    Results of comparing multiple models.
    """
    comparison_id: int
    comparison_name: str
    model_ids: List[int]
    comparison_type: str
    winner_model_id: Optional[int] = None
    comparison_results: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "kedro_pipeline"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'comparison_id': self.comparison_id,
            'comparison_name': self.comparison_name,
            'model_ids': self.model_ids,
            'comparison_type': self.comparison_type,
            'winner_model_id': self.winner_model_id,
            'comparison_results': self.comparison_results,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'created_by': self.created_by
        }


@dataclass
class ModelRegistryConfig:
    """
    Configuration for Model Registry.
    """
    connection_params: Dict[str, str]
    base_path: str = "data/09_model_registry"
    auto_version: bool = True
    auto_archive_old_versions: bool = False
    max_versions_per_model: int = 10
    enable_lineage_tracking: bool = True
    enable_data_hash_validation: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'connection_params': self.connection_params,
            'base_path': self.base_path,
            'auto_version': self.auto_version,
            'auto_archive_old_versions': self.auto_archive_old_versions,
            'max_versions_per_model': self.max_versions_per_model,
            'enable_lineage_tracking': self.enable_lineage_tracking,
            'enable_data_hash_validation': self.enable_data_hash_validation
        }


@dataclass
class ModelDeploymentInfo:
    """
    Information about model deployment.
    """
    model_id: int
    deployment_id: str
    environment: str  # 'dev', 'staging', 'production'
    endpoint_url: Optional[str] = None
    deployment_status: str = "pending"  # 'pending', 'active', 'failed', 'retired'
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    deployed_at: datetime = field(default_factory=datetime.now)
    deployed_by: str = "kedro_pipeline"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'deployment_id': self.deployment_id,
            'environment': self.environment,
            'endpoint_url': self.endpoint_url,
            'deployment_status': self.deployment_status,
            'deployment_config': self.deployment_config,
            'deployed_at': self.deployed_at.isoformat() if isinstance(self.deployed_at, datetime) else self.deployed_at,
            'deployed_by': self.deployed_by
        }


# Utility functions for validation and conversion

def validate_model_type(model_type: str) -> bool:
    """Validate model type."""
    return model_type in [e.value for e in ModelType]


def validate_model_status(status: str) -> bool:
    """Validate model status."""
    return status in [e.value for e in ModelStatus]


def parse_version(version: str) -> tuple:
    """Parse semantic version string."""
    try:
        parts = version.split('.')
        if len(parts) != 3:
            raise ValueError("Version must be in format 'major.minor.patch'")
        return tuple(map(int, parts))
    except ValueError as e:
        raise ValueError(f"Invalid version format: {version}. {str(e)}")


def increment_version(current_version: str, increment_type: str = "patch") -> str:
    """
    Increment version number.
    
    Args:
        current_version: Current version string (e.g., "1.0.0")
        increment_type: Type of increment ('major', 'minor', 'patch')
        
    Returns:
        str: New version string
    """
    major, minor, patch = parse_version(current_version)
    
    if increment_type == "major":
        return f"{major + 1}.0.0"
    elif increment_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif increment_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError("increment_type must be 'major', 'minor', or 'patch'")