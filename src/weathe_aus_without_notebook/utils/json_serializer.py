import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from decimal import Decimal

def json_serial(obj):
    """Serializador JSON para objetos no serializables por defecto"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.datetime64):
        return pd.Timestamp(obj).isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, 'isoformat'):  # Para cualquier objeto datetime-like
        return obj.isoformat()
    else:
        return str(obj)  # Último recurso: convertir a string

# Para usar en catalog.yml, también necesitamos esta función
def safe_json_dump(data, filepath, **kwargs):
    """Función segura para guardar JSON con timestamp handling"""
    with open(filepath, 'w') as f:
        json.dump(data, f, default=json_serial, **kwargs)