"""
This is a boilerplate pipeline 'exploracion_inicial'
generated using Kedro 0.19.11
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    preprocesar_datos,
    analizar_nulos,
    generar_descriptivos,
    graficar_valores_faltantes,
    graficar_histogramas,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(preprocesar_datos, inputs="csv_weather_aus", outputs="datos_procesados", name="preprocesamiento"),
        node(analizar_nulos, inputs="datos_procesados", outputs="reporte_nulos", name="nulos"),
        node(generar_descriptivos, inputs="datos_procesados", outputs="reporte_descriptivos", name="descriptivos"),
        node(graficar_valores_faltantes, inputs="datos_procesados", outputs=None, name="graficos_nulos"),
        node(graficar_histogramas, inputs="datos_procesados", outputs=None, name="graficos_histogramas"),
    ])