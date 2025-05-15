"""
Registro de pipelines.
"""

from typing import Dict

from kedro.pipeline import Pipeline
# Corregir esta línea para usar el nombre real de tu paquete
from weathe_aus_without_notebook.pipelines import data_to_postgres, data_transform, feature_selection_regression
from weathe_aus_without_notebook.pipelines.regression_models.linear_regression import pipeline as lr_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.random_forest import pipeline as rf_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.knn_regressor import pipeline as knn_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.bayes_regression import pipeline as bayes_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.gradient_boosting import pipeline as gb_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.ridge import pipeline as ridge_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.lasso import pipeline as lasso_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_models.svr import pipeline as svr_pipeline_module
from weathe_aus_without_notebook.pipelines.regression_evaluation import pipeline as eval_pipeline_module




def register_pipelines() -> Dict[str, Pipeline]:
    """
    Registra los pipelines del proyecto.
    
    Returns:
        Un diccionario con los pipelines.
    """
    data_to_postgres_pipeline = data_to_postgres.create_pipeline()
    data_transform_pipeline = data_transform.create_pipeline()
    feature_selection_regression_pipeline = feature_selection_regression.create_pipeline()
    linear_regression_pipeline = lr_pipeline_module.create_pipeline()
    random_forest_pipeline = rf_pipeline_module.create_pipeline()
    knn_regressor_pipeline = knn_pipeline_module.create_pipeline()
    gaussian_nb_regressor_pipeline = bayes_pipeline_module.create_pipeline()
    gradient_boosting_regressor_pipeline = gb_pipeline_module.create_pipeline()
    ridge_regressor_pipeline = ridge_pipeline_module.create_pipeline()
    lasso_regressor_pipeline = lasso_pipeline_module.create_pipeline()
    svr_pipeline = svr_pipeline_module.create_pipeline()
    regression_evaluation_pipeline = eval_pipeline_module.create_pipeline()
    
    
    return {
        "data_to_postgres": data_to_postgres_pipeline,
        "data_transform": data_transform_pipeline,
        "feature_selection_regression": feature_selection_regression_pipeline,
        "linear_regression": linear_regression_pipeline,
        "random_forest": random_forest_pipeline,
        "knn_regressor": knn_regressor_pipeline,
        "gaussian_nb_regressor": gaussian_nb_regressor_pipeline,
        "gradient_boosting_regressor": gradient_boosting_regressor_pipeline,
        "ridge_regressor": ridge_regressor_pipeline,
        "lasso_regressor": lasso_regressor_pipeline,
        "svr": svr_pipeline,
        "regression_evaluation": regression_evaluation_pipeline,
        # "regression_preparation": data_to_postgres_pipeline + data_transform_pipeline + feature_selection_regression_pipeline,
        "__default__": data_to_postgres_pipeline + data_transform_pipeline + feature_selection_regression_pipeline
    }