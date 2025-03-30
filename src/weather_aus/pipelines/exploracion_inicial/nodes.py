"""
This is a boilerplate pipeline 'exploracion_inicial'
generated using Kedro 0.19.11
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


def preprocesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    cat_cols = ['Location', 'RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    num_cols = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed',
        'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 
        'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RISK_MM'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def analizar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    missing_pct = df.isnull().mean() * 100
    return pd.DataFrame({'Porcentaje de valores nulos': missing_pct}).sort_values(by='Porcentaje de valores nulos', ascending=False)


def generar_descriptivos(df: pd.DataFrame) -> dict:
    return {
        "numerico": df.describe(),
        "categorico": df.describe(include=['category'])
    }


def graficar_valores_faltantes(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Mapa de valores faltantes")
    plt.show()

    try:
        msno.matrix(df, figsize=(12, 6))
        plt.show()
    except ImportError:
        print("Falta missingno. Instálalo con 'pip install missingno'")


def graficar_histogramas(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(18, 16))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(5, 4, i)
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Histograma de {col}')
    plt.tight_layout()
    plt.show()