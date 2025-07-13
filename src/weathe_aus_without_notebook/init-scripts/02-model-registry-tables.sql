-- ==================================================
-- MODEL REGISTRY DATABASE SCHEMA
-- Weather Australia MLOps Project
-- ==================================================

-- Drop existing tables if they exist (for development)
-- Comment out in production
DROP TABLE IF EXISTS model_comparisons CASCADE;
DROP TABLE IF EXISTS model_lineage CASCADE;
DROP TABLE IF EXISTS model_hyperparameters CASCADE;
DROP TABLE IF EXISTS model_metrics CASCADE;
DROP TABLE IF EXISTS model_registry CASCADE;

-- ==================================================
-- MAIN TABLES
-- ==================================================

-- Tabla principal de modelos registrados
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('regression', 'classification', 'clustering', 'other')),
    algorithm VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_hash VARCHAR(64) UNIQUE NOT NULL,
    
    -- Metadata del modelo
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'kedro_pipeline',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'deprecated', 'archived', 'failed')),
    
    -- Referencias a datos
    training_data_hash VARCHAR(64),
    test_data_hash VARCHAR(64),
    feature_set_version VARCHAR(20),
    
    -- Ubicación del modelo
    model_path TEXT NOT NULL,
    artifacts_path TEXT,
    
    -- Descripción y metadata
    description TEXT,
    tags JSONB,
    
    -- Constraints
    UNIQUE(model_name, version),
    
    -- Audit fields
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de métricas de performance
CREATE TABLE model_metrics (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15, 8) NOT NULL, -- Increased precision for metrics
    dataset_type VARCHAR(20) NOT NULL CHECK (dataset_type IN ('train', 'validation', 'test')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure no duplicate metrics per model/dataset combination
    UNIQUE(model_id, metric_name, dataset_type)
);

-- Tabla de hyperparámetros
CREATE TABLE model_hyperparameters (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    parameter_name VARCHAR(100) NOT NULL,
    parameter_value TEXT NOT NULL,
    parameter_type VARCHAR(20) NOT NULL CHECK (parameter_type IN ('int', 'float', 'str', 'bool', 'dict', 'list')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure no duplicate parameters per model
    UNIQUE(model_id, parameter_name)
);

-- Tabla de lineage (trazabilidad)
CREATE TABLE model_lineage (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    pipeline_name VARCHAR(100) NOT NULL,
    pipeline_version VARCHAR(20),
    node_name VARCHAR(100),
    input_datasets TEXT[],
    parent_model_ids INTEGER[],
    git_commit_hash VARCHAR(40),
    kedro_version VARCHAR(20),
    environment_info JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de comparaciones entre modelos
CREATE TABLE model_comparisons (
    id SERIAL PRIMARY KEY,
    comparison_name VARCHAR(100) NOT NULL,
    model_ids INTEGER[] NOT NULL,
    comparison_type VARCHAR(50) NOT NULL,
    winner_model_id INTEGER REFERENCES model_registry(id),
    comparison_results JSONB,
    comparison_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'kedro_pipeline',
    
    -- Validation: at least 2 models to compare
    CHECK (array_length(model_ids, 1) >= 2)
);

-- ==================================================
-- AUDIT AND TRACKING TABLES
-- ==================================================

-- Tabla de cambios de estado (audit log)
CREATE TABLE model_audit_log (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL, -- 'created', 'updated', 'archived', 'deployed', etc.
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(100) DEFAULT 'kedro_pipeline',
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT,
    
    -- Index for performance
    INDEX (model_id, changed_at)
);

-- Tabla de deployments (para futuro)
CREATE TABLE model_deployments (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES model_registry(id) ON DELETE CASCADE,
    deployment_id VARCHAR(100) UNIQUE NOT NULL,
    environment VARCHAR(50) NOT NULL, -- 'dev', 'staging', 'production'
    endpoint_url TEXT,
    deployment_status VARCHAR(20) DEFAULT 'pending' 
        CHECK (deployment_status IN ('pending', 'active', 'failed', 'retired')),
    deployment_config JSONB,
    deployed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_by VARCHAR(100) DEFAULT 'kedro_pipeline',
    retired_at TIMESTAMP,
    
    -- Only one active deployment per environment per model
    UNIQUE(model_id, environment) WHERE deployment_status = 'active'
);

-- ==================================================
-- PERFORMANCE INDEXES
-- ==================================================

-- Primary indexes for model_registry
CREATE INDEX idx_model_registry_name_version ON model_registry(model_name, version);
CREATE INDEX idx_model_registry_type_algorithm ON model_registry(model_type, algorithm);
CREATE INDEX idx_model_registry_created_at ON model_registry(created_at DESC);
CREATE INDEX idx_model_registry_status ON model_registry(status);
CREATE INDEX idx_model_registry_hash ON model_registry(model_hash);
CREATE INDEX idx_model_registry_tags ON model_registry USING GIN (tags);

-- Indexes for model_metrics
CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX idx_model_metrics_name_type ON model_metrics(metric_name, dataset_type);
CREATE INDEX idx_model_metrics_value ON model_metrics(metric_value DESC);

-- Indexes for model_hyperparameters
CREATE INDEX idx_model_hyperparameters_model_id ON model_hyperparameters(model_id);
CREATE INDEX idx_model_hyperparameters_name ON model_hyperparameters(parameter_name);

-- Indexes for model_lineage
CREATE INDEX idx_model_lineage_model_id ON model_lineage(model_id);
CREATE INDEX idx_model_lineage_pipeline ON model_lineage(pipeline_name);
CREATE INDEX idx_model_lineage_git_commit ON model_lineage(git_commit_hash);

-- Indexes for model_comparisons
CREATE INDEX idx_model_comparisons_created_at ON model_comparisons(created_at DESC);
CREATE INDEX idx_model_comparisons_type ON model_comparisons(comparison_type);

-- Indexes for audit_log
CREATE INDEX idx_model_audit_log_model_id_time ON model_audit_log(model_id, changed_at DESC);
CREATE INDEX idx_model_audit_log_action ON model_audit_log(action);

-- ==================================================
-- USEFUL VIEWS
-- ==================================================

-- Vista para ver modelos con sus mejores métricas
CREATE OR REPLACE VIEW model_summary AS
SELECT 
    mr.id,
    mr.model_name,
    mr.model_type,
    mr.algorithm,
    mr.version,
    mr.status,
    mr.created_at,
    mr.created_by,
    mr.description,
    mr.tags,
    
    -- Primary metric based on model type
    CASE 
        WHEN mr.model_type = 'regression' THEN
            (SELECT metric_value FROM model_metrics mm 
             WHERE mm.model_id = mr.id 
             AND mm.metric_name = 'r2_score' 
             AND mm.dataset_type = 'test' 
             LIMIT 1)
        WHEN mr.model_type = 'classification' THEN
            (SELECT metric_value FROM model_metrics mm 
             WHERE mm.model_id = mr.id 
             AND mm.metric_name = 'accuracy' 
             AND mm.dataset_type = 'test' 
             LIMIT 1)
    END AS primary_metric,
    
    -- Secondary metrics
    (SELECT metric_value FROM model_metrics mm 
     WHERE mm.model_id = mr.id 
     AND mm.metric_name = 'mean_squared_error' 
     AND mm.dataset_type = 'test' 
     LIMIT 1) AS mse,
     
    (SELECT metric_value FROM model_metrics mm 
     WHERE mm.model_id = mr.id 
     AND mm.metric_name = 'f1_score' 
     AND mm.dataset_type = 'test' 
     LIMIT 1) AS f1_score,
    
    -- Metadata counts
    (SELECT COUNT(*) FROM model_metrics mm WHERE mm.model_id = mr.id) as metrics_count,
    (SELECT COUNT(*) FROM model_hyperparameters mh WHERE mh.model_id = mr.id) as hyperparameters_count
    
FROM model_registry mr
ORDER BY mr.created_at DESC;

-- Vista para leaderboard de modelos por tipo
CREATE OR REPLACE VIEW model_leaderboard AS
SELECT 
    model_type,
    algorithm,
    model_name,
    version,
    primary_metric,
    RANK() OVER (
        PARTITION BY model_type 
        ORDER BY 
            CASE 
                WHEN model_type = 'regression' THEN primary_metric
                WHEN model_type = 'classification' THEN primary_metric
            END DESC NULLS LAST
    ) as rank,
    created_at,
    status
FROM model_summary 
WHERE status = 'active' 
  AND primary_metric IS NOT NULL;

-- Vista para comparaciones recientes
CREATE OR REPLACE VIEW recent_model_comparisons AS
SELECT 
    mc.comparison_name,
    mc.comparison_type,
    mc.created_at,
    mc.created_by,
    mr.model_name as winner_model_name,
    mr.algorithm as winner_algorithm,
    mr.version as winner_version,
    array_length(mc.model_ids, 1) as models_compared
FROM model_comparisons mc
LEFT JOIN model_registry mr ON mc.winner_model_id = mr.id
ORDER BY mc.created_at DESC;

-- Vista para audit trail
CREATE OR REPLACE VIEW model_audit_summary AS
SELECT 
    mr.model_name,
    mr.version,
    mal.action,
    mal.changed_by,
    mal.changed_at,
    mal.reason
FROM model_audit_log mal
JOIN model_registry mr ON mal.model_id = mr.id
ORDER BY mal.changed_at DESC;

-- ==================================================
-- FUNCTIONS AND TRIGGERS
-- ==================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for model_registry updated_at
CREATE TRIGGER update_model_registry_updated_at 
    BEFORE UPDATE ON model_registry 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Function to create audit log entries
CREATE OR REPLACE FUNCTION create_audit_log_entry()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO model_audit_log (model_id, action, new_values)
        VALUES (NEW.id, 'created', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO model_audit_log (model_id, action, old_values, new_values)
        VALUES (NEW.id, 'updated', to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO model_audit_log (model_id, action, old_values)
        VALUES (OLD.id, 'deleted', to_jsonb(OLD));
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger for audit logging
CREATE TRIGGER model_registry_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON model_registry
    FOR EACH ROW
    EXECUTE FUNCTION create_audit_log_entry();

-- ==================================================
-- INITIAL DATA AND PERMISSIONS
-- ==================================================

-- Create a default admin user (if needed)
-- INSERT INTO model_registry_users (username, role) VALUES ('kedro_pipeline', 'admin');

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO kedro_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO kedro_user;

-- ==================================================
-- COMMENTS FOR DOCUMENTATION
-- ==================================================

COMMENT ON TABLE model_registry IS 'Central registry for all ML models with versioning and metadata';
COMMENT ON TABLE model_metrics IS 'Performance metrics for each model';
COMMENT ON TABLE model_hyperparameters IS 'Hyperparameters used for each model';
COMMENT ON TABLE model_lineage IS 'Lineage and provenance information for models';
COMMENT ON TABLE model_comparisons IS 'Results of comparing multiple models';
COMMENT ON TABLE model_audit_log IS 'Audit trail for all model registry changes';
COMMENT ON TABLE model_deployments IS 'Deployment information for models';

COMMENT ON VIEW model_summary IS 'Summary view of all models with key metrics';
COMMENT ON VIEW model_leaderboard IS 'Ranked models by performance within each type';
COMMENT ON VIEW recent_model_comparisons IS 'Recent model comparison results';

-- Success message
SELECT 'Model Registry tables created successfully!' as status;