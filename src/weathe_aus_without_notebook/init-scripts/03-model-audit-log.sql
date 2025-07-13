-- Crear tabla model_audit_log que falta
-- Guarda este archivo como: init-scripts/03-model-audit-log.sql

CREATE TABLE IF NOT EXISTS model_audit_log (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    action VARCHAR(50) NOT NULL,
    new_values JSONB,
    old_values JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT
);

-- Crear índices para mejor rendimiento
CREATE INDEX IF NOT EXISTS idx_model_audit_log_model_id ON model_audit_log(model_id);
CREATE INDEX IF NOT EXISTS idx_model_audit_log_action ON model_audit_log(action);
CREATE INDEX IF NOT EXISTS idx_model_audit_log_timestamp ON model_audit_log(timestamp);

-- Crear función para audit log automático
CREATE OR REPLACE FUNCTION create_audit_log_entry()
RETURNS TRIGGER AS $$
BEGIN
    -- Solo para operaciones INSERT, UPDATE, DELETE
    IF TG_OP = 'INSERT' THEN
        INSERT INTO model_audit_log (model_id, action, new_values)
        VALUES (NEW.id, 'created', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO model_audit_log (model_id, action, new_values, old_values)
        VALUES (NEW.id, 'updated', to_jsonb(NEW), to_jsonb(OLD));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO model_audit_log (model_id, action, old_values)
        VALUES (OLD.id, 'deleted', to_jsonb(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Verificar si existe la tabla models antes de crear el trigger
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'models') THEN
        -- Crear trigger para auditoría automática en la tabla models
        DROP TRIGGER IF EXISTS model_audit_trigger ON models;
        CREATE TRIGGER model_audit_trigger
            AFTER INSERT OR UPDATE OR DELETE ON models
            FOR EACH ROW EXECUTE FUNCTION create_audit_log_entry();
        
        RAISE NOTICE 'Audit trigger created for models table';
    ELSE
        RAISE NOTICE 'Models table does not exist yet. Trigger will need to be created later.';
    END IF;
END $$;

-- Mensaje de confirmación
SELECT 'model_audit_log table and functions created successfully' as status;