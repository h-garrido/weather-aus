import pandas as pd
import logging
from sqlalchemy import create_engine, text
from typing import Dict, Any

logger = logging.getLogger(__name__)

def clean_and_save_to_postgres(df: pd.DataFrame) -> None:
    """
    Limpia y prepara los datos meteorológicos de Australia y los guarda en PostgreSQL en Docker.
    
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
    
    # Para demostración, vamos a eliminar filas con valores nulos
    df_clean = df_clean.dropna()
    
    # Registramos info de los datos limpios
    logger.info(f"Datos limpios: {len(df_clean)} filas")
    
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