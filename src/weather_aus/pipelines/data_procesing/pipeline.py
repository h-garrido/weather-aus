# src/weather_aus/pipelines/data_processing/pipeline.py
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import load_weather_data, clean_weather_data, calculate_weather_metrics

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de procesamiento de datos meteorológicos.
    
    Returns:
        Pipeline de Kedro configurado
    """
    return Pipeline(
        [
            node(
                func=load_weather_data,
                inputs="params:weather_data_filepath",  # Esto vendría del archivo parameters.yml
                outputs="raw_weather_data",
                name="load_raw_weather_data_node",
            ),
            node(
                func=clean_weather_data,
                inputs="raw_weather_data",
                outputs="clean_weather_data",
                name="clean_weather_data_node",
            ),
            node(
                func=calculate_weather_metrics,
                inputs="clean_weather_data",
                outputs={"metrics": "weather_metrics", "processed_data": "processed_weather_data"},
                name="calculate_weather_metrics_node",
            ),
        ]
    )