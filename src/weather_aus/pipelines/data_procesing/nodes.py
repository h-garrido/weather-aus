# src/weather_aus/pipelines/data_processing/nodes.py
import pandas as pd
import numpy as np
from typing import Dict, Any

def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Carga el dataset de meteorología desde un archivo CSV.
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        DataFrame con los datos meteorológicos cargados
    """
    return pd.read_csv(filepath)

def clean_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset meteorológico, maneja valores nulos, etc.
    
    Args:
        weather_df: DataFrame con datos meteorológicos crudos
        
    Returns:
        DataFrame con datos meteorológicos limpios
    """
    # Ejemplo de limpieza
    clean_df = weather_df.copy()
    
    # Convertir fechas a formato datetime
    if 'Date' in clean_df.columns:
        clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
    # Manejar valores nulos en variables numéricas
    numeric_cols = clean_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        clean_df[col] = clean_df[col].fillna(clean_df[col].median())
    
    return clean_df

def calculate_weather_metrics(clean_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula métricas y KPIs sobre los datos meteorológicos.
    
    Args:
        clean_df: DataFrame con datos meteorológicos limpios
        
    Returns:
        Diccionario con métricas calculadas
    """
    metrics = {}
    
    # Ejemplo de métricas
    if 'Rainfall' in clean_df.columns:
        metrics['avg_rainfall'] = clean_df['Rainfall'].mean()
        metrics['max_rainfall'] = clean_df['Rainfall'].max()
    
    if 'Temperature' in clean_df.columns:
        metrics['avg_temperature'] = clean_df['Temperature'].mean()
        metrics['temp_variance'] = clean_df['Temperature'].var()
    
    # Si hay datos por ubicación, puedes agrupar
    if 'Location' in clean_df.columns:
        metrics['location_summary'] = clean_df.groupby('Location').agg({
            'Rainfall': 'mean',
            'Temperature': ['mean', 'std']
        })
    
    return {
        "metrics": metrics,
        "processed_data": clean_df
    }