import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def create_feature_interactions_regression(df):
    """
    Crea variables de interacción y derivadas para modelos de regresión.
    
    Args:
        df: DataFrame con datos transformados
    
    Returns:
        DataFrame con características adicionales
    """
    logger.info(f"Creando interacciones y características derivadas para regresión con {len(df)} filas")
    
    # Hacer una copia para no modificar el original
    df_features = df.copy()
    
    # Crear interacciones y variables derivadas
    # RainTempInteraction: Interacción entre lluvia y temperatura
    if all(col in df_features.columns for col in ['rainfall', 'max_temp']):
        df_features['RainTempInteraction'] = df_features['rainfall'] * df_features['max_temp']
    
    # HumidityInteraction: Interacción entre humedad mañana y tarde
    if all(col in df_features.columns for col in ['humidity_9am', 'humidity_3pm']):
        df_features['HumidityInteraction'] = df_features['humidity_9am'] * df_features['humidity_3pm']
    
    # WindSpeedAvg: Promedio de velocidad de viento mañana y tarde
    if all(col in df_features.columns for col in ['wind_speed_9am', 'wind_speed_3pm']):
        df_features['WindSpeedAvg'] = (df_features['wind_speed_9am'] + df_features['wind_speed_3pm']) / 2
    
    # TempRange: Diferencia entre temperatura máxima y mínima
    if all(col in df_features.columns for col in ['max_temp', 'min_temp']):
        df_features['TempRange'] = df_features['max_temp'] - df_features['min_temp']
    
    # Renombrar algunas columnas para que coincidan con los nombres especificados
    column_mapping = {
        'humidity_3pm': 'Humidity3pm',
        'pressure_3pm': 'Pressure3pm',
        'wind_gust_speed': 'WindGustSpeed',
        'sunshine': 'Sunshine',
        'temp_9am': 'Temp9am',
        'pressure_9am': 'Pressure9am',
        'wind_speed_9am': 'WindSpeed9am',
        'cloud_3pm': 'Cloud3pm',
        'cloud_9am': 'Cloud9am',
        'evaporation': 'Evaporation',
        'max_temp': 'MaxTemp',
        'min_temp': 'MinTemp'
    }
    
    # Aplicar el mapeo de columnas solo a las que existen
    for old_col, new_col in column_mapping.items():
        if old_col in df_features.columns:
            df_features.rename(columns={old_col: new_col}, inplace=True)
    
    logger.info(f"Características derivadas creadas para regresión. Total columnas: {df_features.shape[1]}")
    
    return df_features

def select_regression_features(df):
    """
    Selecciona las variables más importantes para el modelado de regresión,
    utilizando RISK_MM como variable objetivo de regresión.
    
    Args:
        df: DataFrame con todas las características
    
    Returns:
        DataFrame con solo las características seleccionadas y risk_mm como target
    """
    # Lista de características importantes para predecir risk_mm
    important_features = [
        'Humidity3pm',     
        'Pressure3pm',     
        'WindGustSpeed',     
        'Sunshine',     
        'RainTempInteraction',     
        'Temp9am',     
        'HumidityInteraction',     
        'WindSpeedAvg',     
        'Pressure9am',     
        'TempRange',     
        'WindSpeed9am',     
        'Cloud3pm',     
        'Cloud9am',     
        'Evaporation',     
        'MaxTemp',     
        'MinTemp',
        'rainfall'  # Incluimos rainfall como predictor
    ]
    
    # Variable objetivo para regresión
    target_var = 'risk_mm'
    
    # Verificar que la variable objetivo existe
    if target_var not in df.columns:
        logger.error(f"Variable objetivo {target_var} no está presente en el DataFrame")
        raise ValueError(f"Variable objetivo {target_var} no encontrada")
    
    # Filtrar solo las columnas que existen en el DataFrame
    existing_features = [col for col in important_features if col in df.columns]
    
    if len(existing_features) < len(important_features):
        missing_features = set(important_features) - set(existing_features)
        logger.warning(f"Algunas características importantes no están presentes: {missing_features}")
    
    # Seleccionar características y variable objetivo
    columns_to_select = existing_features + [target_var]
    df_selected = df[columns_to_select].copy()
    
    logger.info(f"Seleccionadas {len(existing_features)} características para regresión")
    logger.info(f"Características seleccionadas: {', '.join(existing_features)}")
    logger.info(f"Variable objetivo: {target_var}")
    
    # Eliminar filas con valores nulos en las características seleccionadas o en la variable objetivo
    rows_before = len(df_selected)
    df_selected.dropna(inplace=True)
    rows_after = len(df_selected)
    logger.info(f"Eliminadas {rows_before - rows_after} filas con valores nulos")
    
    # Winsorización para tratar outliers (aplicado a todas las columnas numéricas)
    numeric_cols = df_selected.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    for col in numeric_cols:
        # Calculamos los percentiles 1 y 99
        p01 = df_selected[col].quantile(0.01)
        p99 = df_selected[col].quantile(0.99)
        
        # Aplicamos winsorización
        df_selected[col] = df_selected[col].clip(lower=p01, upper=p99)
    
    logger.info(f"Dataset final para modelado de regresión: {df_selected.shape[0]} filas, {df_selected.shape[1]} columnas")
    
    return df_selected