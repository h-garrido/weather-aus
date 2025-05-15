import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def transform_weather_data(df):
    """
    Transforma los datos meteorológicos para prepararlos para el modelado.
    
    Args:
        df: DataFrame con los datos cargados desde PostgreSQL.
        
    Returns:
        DataFrame con los datos transformados.
    """
    logger.info(f"Transformando {len(df)} filas de datos")
    
    # Hacer una copia para evitar modificar el original
    df_transformed = df.copy()
    
    # Convertir la columna de fecha a tipo fecha
    if 'date' in df_transformed.columns:
        df_transformed['date'] = pd.to_datetime(df_transformed['date'])
        
        # Extraer características de la fecha
        df_transformed['year'] = df_transformed['date'].dt.year
        df_transformed['month'] = df_transformed['date'].dt.month
        df_transformed['day'] = df_transformed['date'].dt.day
        df_transformed['day_of_week'] = df_transformed['date'].dt.dayofweek
    
    # Convertir variables categóricas en numéricas
    categorical_cols = df_transformed.select_dtypes(include=['object']).columns.tolist()
    if 'date' in categorical_cols:
        categorical_cols.remove('date')
    if 'location' in categorical_cols:
        categorical_cols.remove('location')
    if 'rain_today' in categorical_cols:
        categorical_cols.remove('rain_today')
    if 'rain_tomorrow' in categorical_cols:
        categorical_cols.remove('rain_tomorrow')
    
    for col in categorical_cols:
        dummies = pd.get_dummies(df_transformed[col], prefix=col, drop_first=True)
        df_transformed = pd.concat([df_transformed, dummies], axis=1)
        df_transformed.drop(columns=[col], inplace=True)
    
    # Convertir rain_today y rain_tomorrow a binario (0/1)
    if 'rain_today' in df_transformed.columns:
        df_transformed['rain_today_binary'] = np.where(df_transformed['rain_today'] == 'Yes', 1, 0)
        df_transformed.drop(columns=['rain_today'], inplace=True)
    
    if 'rain_tomorrow' in df_transformed.columns:
        df_transformed['rain_tomorrow_binary'] = np.where(df_transformed['rain_tomorrow'] == 'Yes', 1, 0)
        df_transformed.drop(columns=['rain_tomorrow'], inplace=True)
    
    # Manejar valores faltantes en columnas numéricas
    numerical_cols = df_transformed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_cols:
        df_transformed[col].fillna(df_transformed[col].median(), inplace=True)
    
    # Crear algunas características derivadas
    if all(col in df_transformed.columns for col in ['max_temp', 'min_temp']):
        df_transformed['temp_diff'] = df_transformed['max_temp'] - df_transformed['min_temp']
    
    if all(col in df_transformed.columns for col in ['humidity_9am', 'humidity_3pm']):
        df_transformed['humidity_diff'] = df_transformed['humidity_9am'] - df_transformed['humidity_3pm']
    
    if all(col in df_transformed.columns for col in ['pressure_9am', 'pressure_3pm']):
        df_transformed['pressure_diff'] = df_transformed['pressure_9am'] - df_transformed['pressure_3pm']
    
    if all(col in df_transformed.columns for col in ['temp_9am', 'temp_3pm']):
        df_transformed['temp_diff_9am_3pm'] = df_transformed['temp_3pm'] - df_transformed['temp_9am']
    
    # Eliminar la columna de fecha después de extraer características
    if 'date' in df_transformed.columns:
        df_transformed.drop(columns=['date'], inplace=True)
    
    # Convertir la columna location en dummies
    if 'location' in df_transformed.columns:
        location_dummies = pd.get_dummies(df_transformed['location'], prefix='location', drop_first=True)
        df_transformed = pd.concat([df_transformed, location_dummies], axis=1)
        df_transformed.drop(columns=['location'], inplace=True)
    
    logger.info(f"Transformación completada. Resultado: {df_transformed.shape[1]} columnas")
    
    return df_transformed