-- Crear tabla para datos limpios
CREATE TABLE IF NOT EXISTS clean_data (
    id SERIAL PRIMARY KEY,
    data_value VARCHAR(255),
    category VARCHAR(100),
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crear índice para mejorar el rendimiento de las consultas
CREATE INDEX IF NOT EXISTS idx_clean_data_timestamp ON clean_data(timestamp);

-- Otorgar permisos al usuario de Kedro
GRANT ALL PRIVILEGES ON TABLE clean_data TO kedro_user;
GRANT USAGE, SELECT ON SEQUENCE clean_data_id_seq TO kedro_user;