from kedro.pipeline import Pipeline, node, pipeline
from .nodes import transform_weather_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=transform_weather_data,
                inputs="raw_data_postgres",
                outputs="transformed_data",
                name="transform_weather_data_node",
            ),
        ]
    )