"""
# Script para verificar
import pickle

# Cargar un modelo
with open('data/06_models/logistic_regression_model.pkl', 'rb') as f:
    model_dict = pickle.load(f)
    
print("Keys:", model_dict.keys())
if 'features_used' in model_dict:
    print("Features used:", model_dict['features_used'])
    print("Number of features:", len(model_dict['features_used']))
"""

# check_features.py
import pandas as pd

# Cargar X_test
X_test = pd.read_parquet('data/05_model_input/X_test_classification.parquet')
print(f"X_test shape: {X_test.shape}")
print(f"X_test columns: {list(X_test.columns)}")

# Características necesarias
needed = ['risk_mm', 'humidity_3pm', 'temp_diff', 'temp_diff_9am_3pm', 
          'rain_today_binary', 'humidity_9am', 'rainfall', 'humidity_diff', 
          'cloud_3pm', 'sunshine', 'cloud_9am', 'wind_gust_speed', 'pressure_3pm']

missing = [f for f in needed if f not in X_test.columns]
print(f"\nMissing features: {missing}")