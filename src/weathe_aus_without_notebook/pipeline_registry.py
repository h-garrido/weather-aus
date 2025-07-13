"""
Registro de pipelines.
"""

from typing import Dict
from kedro.pipeline import Pipeline

"""
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
#from de clasidicacion
from weathe_aus_without_notebook.pipelines import feature_selection_classification
from weathe_aus_without_notebook.pipelines.classification_models.bayes_classification import pipeline as bayes_class_pipeline_module
from weathe_aus_without_notebook.pipelines.classification_models.random_forest_classifier import create_pipeline as random_forest_classifier_pipeline
from weathe_aus_without_notebook.pipelines.classification_models.svm_classifier import pipeline as svm_class_pipeline_module
from weathe_aus_without_notebook.pipelines.classification_models.knn_classifier import pipeline as knn_class_pipeline_module
from weathe_aus_without_notebook.pipelines.classification_models.logistic_regression import pipeline as lr_class_pipeline_module
from weathe_aus_without_notebook.pipelines.classification_models.decision_tree import pipeline as dt_class_pipeline_module
from weathe_aus_without_notebook.pipelines.classification_models.gradient_boosting_classifier import pipeline as gb_class_pipeline_module

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
    
#clasificacion
feature_selection_classification_pipeline = feature_selection_classification.create_pipeline()
bayes_classification_pipeline = bayes_class_pipeline_module.create_pipeline()
random_forest_classifier_pipeline_instance = random_forest_classifier_pipeline()
svm_classifier_pipeline = svm_class_pipeline_module.create_pipeline()
knn_classifier_pipeline = knn_class_pipeline_module.create_pipeline()
logistic_regression_classifier_pipeline = lr_class_pipeline_module.create_pipeline()
decision_tree_classifier_pipeline = dt_class_pipeline_module.create_pipeline()
gradient_boosting_classifier_pipeline = gb_class_pipeline_module.create_pipeline()

return {
        "data_to_postgres": data_to_postgres_pipeline,
        "data_transform": data_transform_pipeline,
        "feature_selection_regression": feature_selection_regression_pipeline,
        "feature_selection_classification": feature_selection_classification_pipeline,
        #regresion
        "linear_regression": linear_regression_pipeline,
        "random_forest": random_forest_pipeline,
        "knn_regressor": knn_regressor_pipeline,
        "gaussian_nb_regressor": gaussian_nb_regressor_pipeline,
        "gradient_boosting_regressor": gradient_boosting_regressor_pipeline,
        "ridge_regressor": ridge_regressor_pipeline,
        "lasso_regressor": lasso_regressor_pipeline,
        "svr": svr_pipeline,
        "regression_evaluation": regression_evaluation_pipeline,
        #clasificacion
        "bayes_classification": bayes_classification_pipeline,
        "random_forest_classifier": random_forest_classifier_pipeline_instance,
        "svm_classifier": svm_classifier_pipeline,
        "knn_classifier": knn_classifier_pipeline,
        "logistic_regression_classifier": logistic_regression_classifier_pipeline,
        "decision_tree_classifier": decision_tree_classifier_pipeline,
        "gradient_boosting_classifier": gradient_boosting_classifier_pipeline,
        
        # "regression_preparation": data_to_postgres_pipeline + data_transform_pipeline + feature_selection_regression_pipeline,
        "__default__": data_to_postgres_pipeline + data_transform_pipeline + feature_selection_regression_pipeline
    }
       
"""

# ===== IMPORT EXISTING PIPELINES =====
from weathe_aus_without_notebook.pipelines import data_to_postgres
from weathe_aus_without_notebook.pipelines import data_transform
from weathe_aus_without_notebook.pipelines import feature_selection_classification
from weathe_aus_without_notebook.pipelines import feature_selection_regression
from weathe_aus_without_notebook.pipelines import regression_evaluation
from weathe_aus_without_notebook.pipelines import classification_evaluation
from .pipelines.model_training import create_pipeline as mt
from weathe_aus_without_notebook.pipelines import model_comparison
from weathe_aus_without_notebook.pipelines import model_evaluation
from weathe_aus_without_notebook.pipelines import model_registry

# Import regression models
from weathe_aus_without_notebook.pipelines.regression_models import (
    linear_regression,
    random_forest as regression_random_forest,
    knn_regressor,
    lasso,
    ridge,
    svr,
    bayes_regression,
    gradient_boosting as regression_gradient_boosting
)

# Import classification models
from weathe_aus_without_notebook.pipelines.classification_models import (
    bayes_classification,
    decision_tree,
    gradient_boosting_classifier,
    knn_classifier,
    logistic_regression,
    random_forest_classifier,
    svm_classifier
)

# ===== IMPORT NEW UNIFIED PIPELINE =====
from weathe_aus_without_notebook.pipelines import data_management


def register_pipelines() -> Dict[str, Pipeline]:
    """
    Registra los pipelines del proyecto.
    
    Returns:
        Un diccionario con los pipelines.
    """

    # ===== NUEVA UNIFICACION DATA MANAGEMENT PIPELINE =====
    data_management_pipeline = data_management.create_full_data_management_pipeline()

    # ===== NEW MODEL TRAINING PIPELINE ===== 
    model_training_pipeline = mt()  

    # ===== model comparision ======
    model_comparison_pipeline = model_comparison.create_pipeline()

    #==== model evaluation ====
    model_evaluation_pipeline = model_evaluation.create_pipeline()

    # ===== MODEL REGISTRY PIPELINES =====
    model_registry_main_pipeline = model_registry.create_pipeline()
    model_registry_registration_pipeline = model_registry.create_model_registration_pipeline()
    model_registry_comparison_pipeline = model_registry.create_model_comparison_pipeline()
    model_registry_monitoring_pipeline = model_registry.create_model_monitoring_pipeline()


    # ===== PIPELINES CONTROL DE DATOS =====
    data_postgres_pipeline = data_to_postgres.create_pipeline()
    data_transform_pipeline = data_transform.create_pipeline()

    # Feature pipelines
    feature_selection_class_pipeline = feature_selection_classification.create_pipeline()
    feature_selection_reg_pipeline = feature_selection_regression.create_pipeline()

    # Evaluation pipelines
    regression_eval_pipeline = regression_evaluation.create_pipeline()
    classification_eval_pipeline = classification_evaluation.create_pipeline()
    
    # ===== MODELOS DE REGRESION PIPELINES =====
    linear_regression_pipeline = linear_regression.create_pipeline()
    random_forest_regression_pipeline = regression_random_forest.create_pipeline()
    knn_regressor_pipeline = knn_regressor.create_pipeline()
    lasso_pipeline = lasso.create_pipeline()
    ridge_pipeline = ridge.create_pipeline()
    svr_pipeline = svr.create_pipeline()
    bayes_regression_pipeline = bayes_regression.create_pipeline()
    gradient_boosting_regression_pipeline = regression_gradient_boosting.create_pipeline()
    
    # ===== MODELOS DE CLASIFICACION PIPELINES =====
    bayes_classification_pipeline = bayes_classification.create_pipeline()
    decision_tree_pipeline = decision_tree.create_pipeline()
    gradient_boosting_classification_pipeline = gradient_boosting_classifier.create_pipeline()
    knn_classification_pipeline = knn_classifier.create_pipeline()
    logistic_regression_pipeline = logistic_regression.create_pipeline()
    random_forest_classification_pipeline = random_forest_classifier.create_pipeline()
    svm_classification_pipeline = svm_classifier.create_pipeline()

    # ===== COMBINED PIPELINES FOR MLOPS =====
    
    # Full regression pipeline (unified data + all regression models + evaluation)
    full_regression_pipeline = (
        data_management_pipeline +
        linear_regression_pipeline +
        random_forest_regression_pipeline +
        knn_regressor_pipeline +
        lasso_pipeline +
        ridge_pipeline +
        svr_pipeline +
        bayes_regression_pipeline +
        gradient_boosting_regression_pipeline +
        regression_eval_pipeline
    )
    
    # Full classification pipeline (unified data + all classification models + evaluation)  
    full_classification_pipeline = (
        data_management_pipeline +
        bayes_classification_pipeline +
        decision_tree_pipeline +
        gradient_boosting_classification_pipeline +
        knn_classification_pipeline +
        logistic_regression_pipeline +
        random_forest_classification_pipeline +
        svm_classification_pipeline +
        classification_eval_pipeline
    )
    
    # Complete MLOps pipeline (everything)
    complete_mlops_pipeline = (
        data_postgres_pipeline +
        data_management_pipeline +
        full_regression_pipeline +
        full_classification_pipeline
    )
    
    complete_mlops_with_registry_pipeline = (
        data_postgres_pipeline +
        data_management_pipeline +
        full_regression_pipeline +
        full_classification_pipeline +
        model_registry_main_pipeline
    )
    
    return {
        # ===== NEW UNIFIED PIPELINES =====
        "data_management": data_management_pipeline,
        "model_training": model_training_pipeline,
        "full_regression": full_regression_pipeline,
        "full_classification": full_classification_pipeline,
        "complete_mlops": complete_mlops_pipeline,
        "model_comparison": model_comparison_pipeline,
        "model_evaluation": model_evaluation_pipeline,
        "model_registry": model_registry_main_pipeline,
        "model_registry_registration": model_registry_registration_pipeline,
        "model_registry_comparison": model_registry_comparison_pipeline,
        "model_registry_monitoring": model_registry_monitoring_pipeline,
        
        # ===== LEGACY DATA PIPELINES =====
        "data_postgres": data_postgres_pipeline,
        "data_transform": data_transform_pipeline,
        "feature_selection_classification": feature_selection_class_pipeline,
        "feature_selection_regression": feature_selection_reg_pipeline,
        
        # ===== REGRESSION MODELS =====
        "linear_regression": linear_regression_pipeline,
        "random_forest_regression": random_forest_regression_pipeline,
        "knn_regressor": knn_regressor_pipeline,
        "lasso": lasso_pipeline,
        "ridge": ridge_pipeline,
        "svr": svr_pipeline,
        "bayes_regression": bayes_regression_pipeline,
        "gradient_boosting_regression": gradient_boosting_regression_pipeline,
        
        # ===== CLASSIFICATION MODELS =====
        "bayes_classification": bayes_classification_pipeline,
        "decision_tree": decision_tree_pipeline,
        "gradient_boosting_classification": gradient_boosting_classification_pipeline,
        "knn_classification": knn_classification_pipeline,
        "logistic_regression": logistic_regression_pipeline,
        "random_forest_classification": random_forest_classification_pipeline,
        "svm_classification": svm_classification_pipeline,
        
        # ===== EVALUATION PIPELINES =====
        "regression_evaluation": regression_eval_pipeline,
        "classification_evaluation": classification_eval_pipeline,
        
        # ===== QUICK TESTING PIPELINES =====
        "quick_test_regression": (
            data_management_pipeline + 
            linear_regression_pipeline + 
            regression_eval_pipeline
        ),
        "quick_test_classification": (
            data_management_pipeline + 
            logistic_regression_pipeline + 
            classification_eval_pipeline
        ),
        
        # ===== LEGACY COMPATIBILITY =====
        "__default__": complete_mlops_pipeline,  # Default when running `kedro run`
        "complete_mlops_with_registry": complete_mlops_with_registry_pipeline,
        
    }


# ===== MLOPS UTILITY FUNCTIONS =====

def get_pipeline_by_task(task: str) -> Pipeline:
    """
    Get pipeline by ML task type.
    
    Args:
        task: Either 'regression', 'classification', or 'both'
    
    Returns:
        Appropriate pipeline for the task
    """
    pipelines = register_pipelines()
    
    if task.lower() == "regression":
        return pipelines["full_regression"]
    elif task.lower() == "classification":
        return pipelines["full_classification"] 
    elif task.lower() == "both":
        return pipelines["complete_mlops"]
    else:
        raise ValueError(f"Unknown task: {task}. Use 'regression', 'classification', or 'both'")


def get_model_pipeline(model_name: str, task: str) -> Pipeline:
    """
    Get specific model pipeline with unified data processing.
    
    Args:
        model_name: Name of the model (e.g., 'random_forest', 'svm')
        task: Either 'regression' or 'classification'
    
    Returns:
        Pipeline with data management + specific model + evaluation
    """
    pipelines = register_pipelines()
    data_mgmt = pipelines["data_management"]
    
    if task.lower() == "regression":
        if model_name.lower() == "random_forest":
            return data_mgmt + pipelines["random_forest_regression"] + pipelines["regression_evaluation"]
        elif model_name.lower() == "linear":
            return data_mgmt + pipelines["linear_regression"] + pipelines["regression_evaluation"]
        # Add more model mappings as needed
        
    elif task.lower() == "classification":
        if model_name.lower() == "random_forest":
            return data_mgmt + pipelines["random_forest_classification"] + pipelines["classification_evaluation"]
        elif model_name.lower() == "logistic":
            return data_mgmt + pipelines["logistic_regression"] + pipelines["classification_evaluation"]
        # Add more model mappings as needed
    
    raise ValueError(f"Unknown model/task combination: {model_name}/{task}")