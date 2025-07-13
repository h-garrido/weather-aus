"""
Model Registry Implementation for Weather Australia MLOps
=========================================================

Central registry for managing ML models with versioning, metadata tracking,
and lineage management integrated with Kedro framework.
"""

import hashlib
import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine
import joblib
import logging

from .models import ModelMetadata, ModelVersion, ModelComparison
from .exceptions import ModelRegistryError, ModelNotFoundError, DuplicateModelError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Central registry for managing ML models with versioning, metadata tracking,
    and lineage management.
    
    Integrated with Kedro data catalog and follows MLOps best practices.
    """
    
    def __init__(self, connection_params: Dict[str, str], base_path: str = "data/09_model_registry"):
        """
        Initialize Model Registry with database connection and file paths.
        
        Args:
            connection_params: Database connection parameters
            base_path: Base path for storing model registry files
        """
        self.connection_params = connection_params
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "model_metadata").mkdir(exist_ok=True)
        (self.base_path / "model_versions").mkdir(exist_ok=True)
        (self.base_path / "comparisons").mkdir(exist_ok=True)
        
        # Database connection
        self.engine = create_engine(
            f"postgresql://{connection_params['user']}:"
            f"{connection_params['password']}@{connection_params['host']}:"
            f"{connection_params['port']}/{connection_params['database']}"
        )
        
        logger.info(f"Model Registry initialized with base path: {self.base_path}")
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.connection_params)
    
    def _calculate_model_hash(self, model: Any) -> str:
        """
        Calculate SHA256 hash of model for uniqueness verification.
        
        Args:
            model: Trained model object
            
        Returns:
            str: SHA256 hash of the model
        """
        model_bytes = pickle.dumps(model)
        return hashlib.sha256(model_bytes).hexdigest()
    
    def _calculate_data_hash(self, data: Union[pd.DataFrame, Any]) -> str:
        """
        Calculate hash for training/test data.
        
        Args:
            data: Dataset to hash
            
        Returns:
            str: SHA256 hash of the data
        """
        if isinstance(data, pd.DataFrame):
            data_string = pd.util.hash_pandas_object(data).sum()
            return hashlib.sha256(str(data_string).encode()).hexdigest()
        else:
            data_bytes = pickle.dumps(data)
            return hashlib.sha256(data_bytes).hexdigest()
    
    def _get_next_version(self, model_name: str) -> str:
        """
        Get next version number for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Next version number (e.g., "1.0.1")
        """
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT version FROM model_registry 
                    WHERE model_name = %s 
                    ORDER BY created_at DESC 
                    LIMIT 1
                    """,
                    (model_name,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return "1.0.0"
                
                # Parse version and increment
                current_version = result[0]
                major, minor, patch = map(int, current_version.split('.'))
                return f"{major}.{minor}.{patch + 1}"
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,  # 'regression' or 'classification'
        algorithm: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data: Optional[Any] = None,
        test_data: Optional[Any] = None,
        pipeline_info: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: Trained model object
            model_name: Name of the model
            model_type: Type of model ('regression' or 'classification')
            algorithm: Algorithm used (e.g., 'logistic_regression')
            metrics: Performance metrics dict
            hyperparameters: Model hyperparameters
            training_data: Training dataset (optional)
            test_data: Test dataset (optional)
            pipeline_info: Pipeline metadata (optional)
            description: Model description (optional)
            tags: Additional tags (optional)
            
        Returns:
            str: Model ID (UUID)
            
        Raises:
            DuplicateModelError: If model with same hash already exists
            ModelRegistryError: If registration fails
        """
        try:
            # Calculate hashes
            model_hash = self._calculate_model_hash(model)
            training_data_hash = self._calculate_data_hash(training_data) if training_data is not None else None
            test_data_hash = self._calculate_data_hash(test_data) if test_data is not None else None
            
            # Check if model already exists
            if self._model_exists_by_hash(model_hash):
                raise DuplicateModelError(f"Model with hash {model_hash} already exists")
            
            # Get next version
            version = self._get_next_version(model_name)
            
            # Generate model ID
            model_id = str(uuid.uuid4())
            
            # Save model to file
            model_path = self.base_path / "model_versions" / f"{model_name}_v{version}_{model_id}.pkl"
            joblib.dump(model, model_path)
            
            # Prepare pipeline info
            pipeline_info = pipeline_info or {}
            git_commit = pipeline_info.get('git_commit_hash')
            kedro_version = pipeline_info.get('kedro_version')
            pipeline_name = pipeline_info.get('pipeline_name')
            node_name = pipeline_info.get('node_name')
            
            # Insert into database
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # Insert model registry record
                    cursor.execute(
                        """
                        INSERT INTO model_registry (
                            model_name, model_type, algorithm, version, model_hash,
                            training_data_hash, test_data_hash, model_path,
                            description, tags
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            model_name, model_type, algorithm, version, model_hash,
                            training_data_hash, test_data_hash, str(model_path),
                            description, json.dumps(tags) if tags else None
                        )
                    )
                    
                    db_model_id = cursor.fetchone()[0]
                    
                    # Insert metrics
                    for metric_name, metric_value in metrics.items():
                        cursor.execute(
                            """
                            INSERT INTO model_metrics (model_id, metric_name, metric_value, dataset_type)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (db_model_id, metric_name, float(metric_value), 'test')
                        )
                    
                    # Insert hyperparameters
                    for param_name, param_value in hyperparameters.items():
                        param_type = type(param_value).__name__
                        param_value_str = json.dumps(param_value) if isinstance(param_value, (dict, list)) else str(param_value)
                        
                        cursor.execute(
                            """
                            INSERT INTO model_hyperparameters (model_id, parameter_name, parameter_value, parameter_type)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (db_model_id, param_name, param_value_str, param_type)
                        )
                    
                    # Insert lineage info if available
                    if pipeline_info:
                        cursor.execute(
                            """
                            INSERT INTO model_lineage (
                                model_id, pipeline_name, node_name, git_commit_hash, kedro_version
                            ) VALUES (%s, %s, %s, %s, %s)
                            """,
                            (db_model_id, pipeline_name, node_name, git_commit, kedro_version)
                        )
                    
                    conn.commit()
            
            logger.info(f"Model {model_name} v{version} registered successfully with ID: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {str(e)}")
            raise ModelRegistryError(f"Model registration failed: {str(e)}")
    
    def _model_exists_by_hash(self, model_hash: str) -> bool:
        """Check if model exists by hash."""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM model_registry WHERE model_hash = %s",
                    (model_hash,)
                )
                return cursor.fetchone() is not None
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, ModelMetadata]:
        """
        Get model by name and version.
        
        Args:
            model_name: Name of the model
            version: Specific version (if None, gets latest)
            
        Returns:
            Tuple of (model_object, metadata)
            
        Raises:
            ModelNotFoundError: If model not found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if version:
                        cursor.execute(
                            """
                            SELECT * FROM model_registry 
                            WHERE model_name = %s AND version = %s
                            """,
                            (model_name, version)
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT * FROM model_registry 
                            WHERE model_name = %s 
                            ORDER BY created_at DESC 
                            LIMIT 1
                            """,
                            (model_name,)
                        )
                    
                    record = cursor.fetchone()
                    if not record:
                        raise ModelNotFoundError(f"Model {model_name} (version: {version}) not found")
                    
                    # Load model from file
                    model_path = Path(record['model_path'])
                    model = joblib.load(model_path)
                    
                    # Create metadata object
                    metadata = ModelMetadata(
                        id=record['id'],
                        model_name=record['model_name'],
                        model_type=record['model_type'],
                        algorithm=record['algorithm'],
                        version=record['version'],
                        model_hash=record['model_hash'],
                        created_at=record['created_at'],
                        status=record['status'],
                        description=record['description'],
                        tags=json.loads(record['tags']) if record['tags'] else None
                    )
                    
                    return model, metadata
                    
        except Exception as e:
            logger.error(f"Failed to get model {model_name}: {str(e)}")
            raise ModelRegistryError(f"Failed to retrieve model: {str(e)}")
    
    def list_models(self, model_type: Optional[str] = None, algorithm: Optional[str] = None) -> pd.DataFrame:
        """
        List all models in registry.
        
        Args:
            model_type: Filter by model type (optional)
            algorithm: Filter by algorithm (optional)
            
        Returns:
            DataFrame with model information
        """
        query = "SELECT * FROM model_summary WHERE status = 'active'"
        params = []
        
        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)
        
        if algorithm:
            query += " AND algorithm = %s"
            params.append(algorithm)
        
        query += " ORDER BY created_at DESC"
        
        return pd.read_sql(query, self.engine, params=params)
    
    def get_model_leaderboard(self, model_type: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get leaderboard for models of specific type.
        
        Args:
            model_type: Type of models ('regression' or 'classification')
            top_n: Number of top models to return
            
        Returns:
            DataFrame with ranked models
        """
        query = """
            SELECT * FROM model_leaderboard 
            WHERE model_type = %s 
            ORDER BY rank 
            LIMIT %s
        """
        
        return pd.read_sql(query, self.engine, params=(model_type, top_n))
    
    def compare_models(self, model_ids: List[int], comparison_name: str) -> Dict[str, Any]:
        """
        Compare multiple models and store results.
        
        Args:
            model_ids: List of model IDs to compare
            comparison_name: Name for this comparison
            
        Returns:
            Comparison results
        """
        # Implementation for model comparison logic
        # This would include loading models, running predictions, computing metrics, etc.
        pass
    
    def archive_model(self, model_name: str, version: str) -> bool:
        """
        Archive a model (mark as inactive).
        
        Args:
            model_name: Name of the model
            version: Version to archive
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        UPDATE model_registry 
                        SET status = 'archived' 
                        WHERE model_name = %s AND version = %s
                        """,
                        (model_name, version)
                    )
                    conn.commit()
                    
            logger.info(f"Model {model_name} v{version} archived successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive model {model_name} v{version}: {str(e)}")
            return False
    
    def get_model_metrics(self, model_id: int) -> pd.DataFrame:
        """
        Get all metrics for a specific model.
        
        Args:
            model_id: Database ID of the model
            
        Returns:
            DataFrame with metrics
        """
        query = """
            SELECT metric_name, metric_value, dataset_type, created_at
            FROM model_metrics 
            WHERE model_id = %s
            ORDER BY metric_name, dataset_type
        """
        
        return pd.read_sql(query, self.engine, params=(model_id,))
    
    def get_model_hyperparameters(self, model_id: int) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model.
        
        Args:
            model_id: Database ID of the model
            
        Returns:
            Dictionary with hyperparameters
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT parameter_name, parameter_value, parameter_type
                    FROM model_hyperparameters 
                    WHERE model_id = %s
                    """,
                    (model_id,)
                )
                
                params = {}
                for record in cursor.fetchall():
                    name = record['parameter_name']
                    value = record['parameter_value']
                    param_type = record['parameter_type']
                    
                    # Convert back to original type
                    if param_type in ['dict', 'list']:
                        params[name] = json.loads(value)
                    elif param_type == 'int':
                        params[name] = int(value)
                    elif param_type == 'float':
                        params[name] = float(value)
                    elif param_type == 'bool':
                        params[name] = value.lower() == 'true'
                    else:
                        params[name] = value
                
                return params