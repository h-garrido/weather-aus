import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
from typing import Dict, Any

logger = logging.getLogger(__name__)

def clean_and_save_to_postgres(df: pd.DataFrame) -> None:
    """
    Limpia y prepara los datos meteorológicos de Australia con imputación inteligente
    y los guarda en PostgreSQL en Docker.
    
    Args:
        df: Dataset de entrada con datos meteorológicos.
    """
    # Registra información sobre los datos de entrada
    logger.info(f"Datos de entrada: {len(df)} filas y {df.columns.size} columnas")
    
    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    # Convertir columna de fecha a formato datetime
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    
    # Renombrar columnas para que coincidan con la base de datos (snake_case)
    column_mapping = {
        'Date': 'date',
        'Location': 'location',
        'MinTemp': 'min_temp',
        'MaxTemp': 'max_temp',
        'Rainfall': 'rainfall',
        'Evaporation': 'evaporation',
        'Sunshine': 'sunshine',
        'WindGustDir': 'wind_gust_dir',
        'WindGustSpeed': 'wind_gust_speed',
        'WindDir9am': 'wind_dir_9am',
        'WindDir3pm': 'wind_dir_3pm',
        'WindSpeed9am': 'wind_speed_9am',
        'WindSpeed3pm': 'wind_speed_3pm',
        'Humidity9am': 'humidity_9am',
        'Humidity3pm': 'humidity_3pm',
        'Pressure9am': 'pressure_9am',
        'Pressure3pm': 'pressure_3pm',
        'Cloud9am': 'cloud_9am',
        'Cloud3pm': 'cloud_3pm',
        'Temp9am': 'temp_9am',
        'Temp3pm': 'temp_3pm',
        'RainToday': 'rain_today',
        'RISK_MM': 'risk_mm',
        'RainTomorrow': 'rain_tomorrow'
    }
    
    df_clean = df_clean.rename(columns=column_mapping)
    
    # 📊 ANÁLISIS DE VALORES FALTANTES
    logger.info("Análisis de valores faltantes por columna:")
    missing_info = df_clean.isnull().sum()
    missing_percent = (missing_info / len(df_clean)) * 100
    
    for col in df_clean.columns:
        if missing_info[col] > 0:
            logger.info(f"  {col}: {missing_info[col]} ({missing_percent[col]:.1f}%)")
    
    # 🔧 ESTRATEGIAS DE IMPUTACIÓN POR TIPO DE VARIABLE
    
    # Variables categóricas: Usar moda (valor más frecuente)
    categorical_cols = ['location', 'wind_gust_dir', 'wind_dir_9am', 'wind_dir_3pm', 'rain_today', 'rain_tomorrow']
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            mode_value = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_value)
            logger.info(f"Imputación categórica - {col}: {mode_value}")
    
    # Variables numéricas: Estrategias específicas
    numeric_cols = ['min_temp', 'max_temp', 'rainfall', 'evaporation', 'sunshine', 
                   'wind_gust_speed', 'wind_speed_9am', 'wind_speed_3pm',
                   'humidity_9am', 'humidity_3pm', 'pressure_9am', 'pressure_3pm',
                   'cloud_9am', 'cloud_3pm', 'temp_9am', 'temp_3pm', 'risk_mm']
    
    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().any():
            if col in ['rainfall', 'evaporation', 'sunshine']:
                # Para lluvia, evaporación y sol: usar 0 si falta (no ocurrió)
                df_clean[col] = df_clean[col].fillna(0)
                logger.info(f"Imputación específica - {col}: 0 (no ocurrió)")
            elif col in ['wind_gust_speed', 'wind_speed_9am', 'wind_speed_3pm']:
                # Para vientos: usar mediana (menos sensible a outliers)
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
                logger.info(f"Imputación numérica - {col}: {median_value:.2f} (mediana)")
            elif col in ['humidity_9am', 'humidity_3pm', 'pressure_9am', 'pressure_3pm']:
                # Para humedad y presión: usar interpolación por ubicación
                df_clean[col] = df_clean.groupby('location')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Si aún quedan nulos, usar mediana global
                if df_clean[col].isnull().any():
                    global_median = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(global_median)
                logger.info(f"Imputación por ubicación - {col}: mediana por location")
            elif col in ['temp_9am', 'temp_3pm', 'min_temp', 'max_temp']:
                # Para temperaturas: interpolación temporal si es posible
                df_clean = df_clean.sort_values(['location', 'date'])
                df_clean[col] = df_clean.groupby('location')[col].transform(
                    lambda x: x.interpolate(method='linear')
                )
                # Si aún quedan nulos, usar mediana por ubicación
                df_clean[col] = df_clean.groupby('location')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Si aún quedan nulos, usar mediana global
                if df_clean[col].isnull().any():
                    global_median = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(global_median)
                logger.info(f"Imputación temporal - {col}: interpolación + mediana")
            else:
                # Para otras variables: usar mediana
                median_value = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_value)
                logger.info(f"Imputación numérica - {col}: {median_value:.2f} (mediana)")
    
    # 🔍 VALIDACIONES FINALES
    
    # Eliminar solo filas donde falten datos críticos (targets)
    critical_cols = ['rain_tomorrow']  # Solo eliminar si falta el target
    initial_rows = len(df_clean)
    
    for col in critical_cols:
        if col in df_clean.columns:
            df_clean = df_clean.dropna(subset=[col])
    
    dropped_rows = initial_rows - len(df_clean)
    logger.info(f"Filas eliminadas por falta de target: {dropped_rows}")
    
    # Verificar que no queden nulos
    remaining_nulls = df_clean.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"Quedan {remaining_nulls} valores nulos después de la limpieza")
        # Como último recurso, eliminar filas con cualquier nulo restante
        df_clean = df_clean.dropna()
        logger.info(f"Filas finales después de eliminar nulos restantes: {len(df_clean)}")
    
    # Registramos info de los datos limpios
    logger.info(f"Datos limpios finales: {len(df_clean)} filas de {len(df)} originales")
    logger.info(f"Porcentaje de datos preservados: {(len(df_clean)/len(df)*100):.1f}%")
    
    # Guardar directamente en PostgreSQL en Docker
    try:
        # Conexión a PostgreSQL usando el nombre del servicio en Docker
        logger.info("Conectando a PostgreSQL en Docker...")
        engine = create_engine('postgresql://kedro:kedro@postgres:5432/kedro_db')
        
        # Verificamos que la tabla exista, y si no, la creamos
        logger.info("Verificando si la tabla existe...")
        with engine.connect() as conn:
            # Verificar si la tabla existe
            result = conn.execute(text("SELECT to_regclass('public.clean_data')"))
            table_exists = result.scalar() is not None
            
            # Si la tabla no existe, crearla con la estructura adecuada
            if not table_exists:
                logger.info("La tabla no existe, creándola...")
                create_table_sql = """
                CREATE TABLE clean_data (
                    date DATE,
                    location VARCHAR(255),
                    min_temp FLOAT,
                    max_temp FLOAT,
                    rainfall FLOAT,
                    evaporation FLOAT,
                    sunshine FLOAT,
                    wind_gust_dir VARCHAR(50),
                    wind_gust_speed FLOAT,
                    wind_dir_9am VARCHAR(50),
                    wind_dir_3pm VARCHAR(50),
                    wind_speed_9am FLOAT,
                    wind_speed_3pm FLOAT,
                    humidity_9am FLOAT,
                    humidity_3pm FLOAT,
                    pressure_9am FLOAT,
                    pressure_3pm FLOAT,
                    cloud_9am FLOAT,
                    cloud_3pm FLOAT,
                    temp_9am FLOAT,
                    temp_3pm FLOAT,
                    rain_today VARCHAR(10),
                    risk_mm FLOAT,
                    rain_tomorrow VARCHAR(10)
                )
                """
                conn.execute(text(create_table_sql))
                conn.commit()
                logger.info("Tabla creada exitosamente")
        
        # Guardamos los datos
        logger.info("Guardando datos en PostgreSQL...")
        df_clean.to_sql('clean_data', engine, if_exists='replace', index=False)
        logger.info(f"Se guardaron {len(df_clean)} filas en la tabla clean_data")
        
    except Exception as e:
        logger.error(f"Error al guardar en PostgreSQL: {e}")
        raise
    
    # Devolver None para que el pipeline funcione correctamente
    return None