"""
Custom exceptions for Model Registry
===================================

Specific exceptions for better error handling and debugging.
"""


class ModelRegistryError(Exception):
    """Base exception for Model Registry operations."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotFoundError(ModelRegistryError):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_name: str, version: str = None):
        self.model_name = model_name
        self.version = version
        
        if version:
            message = f"Model '{model_name}' version '{version}' not found in registry"
        else:
            message = f"Model '{model_name}' not found in registry"
            
        super().__init__(message, "MODEL_NOT_FOUND")


class DuplicateModelError(ModelRegistryError):
    """Raised when trying to register a model that already exists."""
    
    def __init__(self, model_hash: str = None, model_name: str = None, version: str = None):
        self.model_hash = model_hash
        self.model_name = model_name
        self.version = version
        
        if model_hash:
            message = f"Model with hash '{model_hash}' already exists in registry"
        elif model_name and version:
            message = f"Model '{model_name}' version '{version}' already exists"
        else:
            message = "Duplicate model detected in registry"
            
        super().__init__(message, "DUPLICATE_MODEL")


class InvalidModelError(ModelRegistryError):
    """Raised when model validation fails."""
    
    def __init__(self, model_name: str, validation_errors: list):
        self.model_name = model_name
        self.validation_errors = validation_errors
        
        message = f"Model '{model_name}' validation failed: {', '.join(validation_errors)}"
        super().__init__(message, "INVALID_MODEL")


class DatabaseConnectionError(ModelRegistryError):
    """Raised when database connection fails."""
    
    def __init__(self, connection_details: str = None):
        self.connection_details = connection_details
        
        message = "Failed to connect to model registry database"
        if connection_details:
            message += f": {connection_details}"
            
        super().__init__(message, "DB_CONNECTION_ERROR")


class ModelFileError(ModelRegistryError):
    """Raised when model file operations fail."""
    
    def __init__(self, file_path: str, operation: str, details: str = None):
        self.file_path = file_path
        self.operation = operation
        self.details = details
        
        message = f"Model file {operation} failed for '{file_path}'"
        if details:
            message += f": {details}"
            
        super().__init__(message, "MODEL_FILE_ERROR")


class VersionError(ModelRegistryError):
    """Raised when version operations fail."""
    
    def __init__(self, version: str, operation: str, details: str = None):
        self.version = version
        self.operation = operation
        self.details = details
        
        message = f"Version {operation} failed for '{version}'"
        if details:
            message += f": {details}"
            
        super().__init__(message, "VERSION_ERROR")


class MetricValidationError(ModelRegistryError):
    """Raised when model metrics validation fails."""
    
    def __init__(self, metric_name: str, metric_value: any, reason: str = None):
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.reason = reason
        
        message = f"Metric '{metric_name}' with value '{metric_value}' is invalid"
        if reason:
            message += f": {reason}"
            
        super().__init__(message, "METRIC_VALIDATION_ERROR")


class HyperparameterError(ModelRegistryError):
    """Raised when hyperparameter operations fail."""
    
    def __init__(self, parameter_name: str, parameter_value: any, operation: str):
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.operation = operation
        
        message = f"Hyperparameter '{parameter_name}' {operation} failed"
        super().__init__(message, "HYPERPARAMETER_ERROR")


class LineageError(ModelRegistryError):
    """Raised when model lineage tracking fails."""
    
    def __init__(self, model_name: str, lineage_type: str, details: str = None):
        self.model_name = model_name
        self.lineage_type = lineage_type
        self.details = details
        
        message = f"Lineage tracking for '{model_name}' failed: {lineage_type}"
        if details:
            message += f" - {details}"
            
        super().__init__(message, "LINEAGE_ERROR")


class ModelComparisonError(ModelRegistryError):
    """Raised when model comparison operations fail."""
    
    def __init__(self, model_ids: list, comparison_type: str, details: str = None):
        self.model_ids = model_ids
        self.comparison_type = comparison_type
        self.details = details
        
        message = f"Model comparison ({comparison_type}) failed for models: {model_ids}"
        if details:
            message += f" - {details}"
            
        super().__init__(message, "MODEL_COMPARISON_ERROR")


class ModelArchiveError(ModelRegistryError):
    """Raised when model archiving operations fail."""
    
    def __init__(self, model_name: str, version: str, operation: str):
        self.model_name = model_name
        self.version = version
        self.operation = operation
        
        message = f"Model archiving {operation} failed for '{model_name}' v{version}"
        super().__init__(message, "MODEL_ARCHIVE_ERROR")


class ConfigurationError(ModelRegistryError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_item: str, details: str = None):
        self.config_item = config_item
        self.details = details
        
        message = f"Configuration error: {config_item}"
        if details:
            message += f" - {details}"
            
        super().__init__(message, "CONFIGURATION_ERROR")


# Utility functions for exception handling

def handle_registry_exception(func):
    """
    Decorator for handling registry exceptions with logging.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ModelRegistryError:
            # Re-raise registry-specific exceptions
            raise
        except Exception as e:
            # Convert general exceptions to registry errors
            raise ModelRegistryError(f"Unexpected error in {func.__name__}: {str(e)}")
    
    return wrapper


def validate_and_raise(condition: bool, exception_class: type, *args, **kwargs):
    """
    Validate condition and raise exception if false.
    
    Args:
        condition: Boolean condition to validate
        exception_class: Exception class to raise
        *args, **kwargs: Arguments for exception
    """
    if not condition:
        raise exception_class(*args, **kwargs)


# Exception mapping for different error scenarios

EXCEPTION_MAPPING = {
    'model_not_found': ModelNotFoundError,
    'duplicate_model': DuplicateModelError,
    'invalid_model': InvalidModelError,
    'db_connection': DatabaseConnectionError,
    'model_file': ModelFileError,
    'version_error': VersionError,
    'metric_validation': MetricValidationError,
    'hyperparameter': HyperparameterError,
    'lineage': LineageError,
    'comparison': ModelComparisonError,
    'archive': ModelArchiveError,
    'configuration': ConfigurationError
}


def get_exception_by_type(error_type: str) -> type:
    """
    Get exception class by error type string.
    
    Args:
        error_type: String identifier for error type
        
    Returns:
        Exception class
    """
    return EXCEPTION_MAPPING.get(error_type, ModelRegistryError)