from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_and_save_to_postgres

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline para limpiar datos y guardarlos en PostgreSQL.
    
    Returns:
        Una instancia de Pipeline.
    """
    return pipeline(
        [
            node(
                func=clean_and_save_to_postgres,
                inputs="raw_data",
                outputs=None,  # No necesitamos una salida ya que los datos se guardan directamente en PostgreSQL
                name="clean_and_save_to_postgres",
            ),
        ]
    )