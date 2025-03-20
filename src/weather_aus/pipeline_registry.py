"""Project pipelines."""
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from weather_aus.pipelines import data_processing as dp



def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "__default__": data_processing_pipeline,
    }
    
    # Si quieres usar el método alternativo de find_pipelines, descomenta correctamente:
    # pipelines = find_pipelines()
    # pipelines["__default__"] = sum(pipelines.values())
    # return pipelines
