en este apartado abra codigo que no existe a lo mejor en la bd pero se necesita o no existe en kedro en el apartado de init-scripts
-- 1. Crear tabla específica para resultados de leaderboard
DROP TABLE IF EXISTS model_leaderboard_results CASCADE;

CREATE TABLE model_leaderboard_results (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50),
    algorithm VARCHAR(100),
    model_name VARCHAR(100),
    version VARCHAR(20),
    primary_metric DECIMAL(15, 8),
    rank INTEGER,
    created_at TIMESTAMP,
    status VARCHAR(20),
    last_updated VARCHAR(50)
);

-- 2. Otorgar permisos completos al usuario kedro
GRANT ALL PRIVILEGES ON TABLE model_leaderboard_results TO kedro;
GRANT USAGE, SELECT ON SEQUENCE model_leaderboard_results_id_seq TO kedro;

-- 3. Crear índices para optimizar consultas
CREATE INDEX idx_leaderboard_results_rank ON model_leaderboard_results(rank);
CREATE INDEX idx_leaderboard_results_metric ON model_leaderboard_results(primary_metric DESC);
CREATE INDEX idx_leaderboard_results_type ON model_leaderboard_results(model_type);

-- 4. Verificar estructura
SELECT column_name, data_type, is_nullable, column_default
FROM information_schema.columns 
WHERE table_name = 'model_leaderboard_results' 
ORDER BY ordinal_position;