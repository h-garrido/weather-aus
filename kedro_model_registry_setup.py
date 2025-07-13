#!/usr/bin/env python3
"""
Sistema de Model Registry para proyecto Kedro + Docker
Uso: python kedro_model_registry_setup.py --setup
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import requests
from typing import Dict, List, Optional

class KedroModelRegistry:
    def __init__(self, config_path="model_registry_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Carga configuración del registry"""
        default_config = {
            "storage_type": "huggingface",  # o "drive", "s3"
            "repository": "tu-usuario/weather-aus-models",
            "local_cache_dir": "data/.model_cache",
            "size_threshold_mb": 10,  # Archivos >10MB van al storage remoto
            "max_versions": 3,  # Máximo versiones locales
            "critical_models": [
                "classification_models_dict.pkl",
                "random_forest_classification_model.pkl",
                "gradient_boosting_regressor_model.pickle"
            ],
            "model_groups": {
                "classification": [
                    "logistic_regression_model.pkl",
                    "random_forest_classification_model.pkl", 
                    "svm_classification_model.pkl",
                    "decision_tree_model.pkl",
                    "bayes_classification_model.pkl",
                    "knn_model.pkl",
                    "gradient_boosting_model.pkl"
                ],
                "regression": [
                    "gaussian_nb_regressor_model.pickle",
                    "gradient_boosting_regressor_model.pickle",
                    "ridge_regressor_model.pickle", 
                    "lasso_regressor_model.pickle",
                    "svr_model.pickle"
                ]
            }
        }
        
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def save_config(self):
        """Guarda configuración"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
    
    def generate_file_hash(self, filepath: str) -> str:
        """Genera hash SHA256 de un archivo"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_file_size_mb(self, filepath: str) -> float:
        """Obtiene tamaño de archivo en MB"""
        return os.path.getsize(filepath) / (1024 * 1024)
    
    def scan_model_files(self) -> Dict[str, Dict]:
        """Escanea todos los archivos de modelo en el proyecto"""
        model_files = {}
        
        # Directorios a escanear
        scan_dirs = [
            "data/06_models",
            "data/05_model_input", 
            "data/02_intermediate",
            "data/09_model_registry"
        ]
        
        for scan_dir in scan_dirs:
            if not os.path.exists(scan_dir):
                continue
                
            for root, dirs, files in os.walk(scan_dir):
                for file in files:
                    if file.endswith(('.pkl', '.pickle', '.joblib')):
                        filepath = os.path.join(root, file)
                        rel_path = os.path.relpath(filepath)
                        
                        size_mb = self.get_file_size_mb(filepath)
                        file_hash = self.generate_file_hash(filepath)
                        
                        model_files[rel_path] = {
                            "filename": file,
                            "size_mb": round(size_mb, 2),
                            "hash": file_hash,
                            "last_modified": datetime.fromtimestamp(
                                os.path.getmtime(filepath)
                            ).isoformat(),
                            "needs_remote_storage": size_mb > self.config["size_threshold_mb"],
                            "is_critical": file in self.config["critical_models"],
                            "model_group": self.get_model_group(file)
                        }
        
        return model_files
    
    def get_model_group(self, filename: str) -> Optional[str]:
        """Determina grupo del modelo"""
        for group, models in self.config["model_groups"].items():
            if filename in models:
                return group
        return None
    
    def create_gitignore_entries(self, model_files: Dict) -> List[str]:
        """Crea entradas para .gitignore"""
        entries = [
            "# Archivos de modelo pesados (gestionados por model registry)",
            ""
        ]
        
        for filepath, info in model_files.items():
            if info["needs_remote_storage"]:
                entries.append(f"{filepath}")
                
        # Agregar directorios con versioning excesivo
        entries.extend([
            "",
            "# Versiones antiguas de datos",
            "data/02_intermediate/versioned_data.pkl/*/",
            "data/06_models/*/2025-05-*",  # Versiones viejas
            "data/06_models/*/2025-06-2[0-2]*",  # Versiones de hace días
            "",
            "# Cache local del model registry", 
            "data/.model_cache/",
            "model_registry_config.json"
        ])
        
        return entries
    
    def generate_download_script(self, model_files: Dict) -> str:
        """Genera script de descarga"""
        script = '''#!/usr/bin/env python3
"""
Script para descargar modelos necesarios antes de kedro run
Uso: python download_models.py [--group=classification] [--force]
"""

import os
import json
import argparse
from pathlib import Path

def download_from_huggingface(repo, filepath, local_path):
    """Descarga archivo desde Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
        
        print(f"[DOWNLOAD] Descargando {filepath}...")
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=filepath,
            local_dir=".",
            local_dir_use_symlinks=False
        )
        print(f"[SUCCESS] Descargado: {local_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Error descargando {filepath}: {e}")
        return False

def download_models(group=None, force=False):
    """Descarga modelos segun grupo"""
    
    # Cargar registry
    if not os.path.exists("model_registry.json"):
        print("[ERROR] model_registry.json no encontrado")
        return False
    
    with open("model_registry.json", "r", encoding='utf-8') as f:
        registry = json.load(f)
    
    repo = registry["config"]["repository"]
    models_to_download = []
    
    # Filtrar modelos por grupo
    for filepath, info in registry["models"].items():
        if not info["needs_remote_storage"]:
            continue
            
        if group and info["model_group"] != group:
            continue
            
        if not force and os.path.exists(filepath):
            print(f"[SKIP] Ya existe: {filepath}")
            continue
            
        models_to_download.append((filepath, info))
    
    if not models_to_download:
        print("[SUCCESS] Todos los modelos estan disponibles")
        return True
    
    print(f"[INFO] Descargando {len(models_to_download)} modelos...")
    
    success_count = 0
    for filepath, info in models_to_download:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if download_from_huggingface(repo, filepath, filepath):
            success_count += 1
    
    print(f"[SUCCESS] Descargados: {success_count}/{len(models_to_download)}")
    return success_count == len(models_to_download)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["classification", "regression"], 
                       help="Solo descargar modelos de un grupo")
    parser.add_argument("--force", action="store_true",
                       help="Forzar descarga aunque archivo exista")
    
    args = parser.parse_args()
    
    success = download_models(args.group, args.force)
    exit(0 if success else 1)
'''
        return script
    
    def generate_docker_integration(self) -> str:
        """Genera integración para Docker"""
        return '''# Agregar al Dockerfile ANTES de kedro run:

# Instalar dependencias del model registry
RUN pip install huggingface_hub

# Copiar script de descarga
COPY scripts/download_models.py /app/
COPY model_registry.json /app/

# Descargar modelos necesarios
RUN python download_models.py --group=classification
RUN python download_models.py --group=regression

# Ahora kedro run funcionará con todos los archivos
CMD ["kedro", "run"]
'''
    
    def create_model_registry(self) -> Dict:
        """Crea el registry completo"""
        print("[SCAN] Escaneando archivos de modelo...")
        model_files = self.scan_model_files()
        
        # Estadísticas
        total_files = len(model_files)
        large_files = sum(1 for info in model_files.values() if info["needs_remote_storage"])
        total_size_mb = sum(info["size_mb"] for info in model_files.values())
        remote_size_mb = sum(info["size_mb"] for info in model_files.values() 
                           if info["needs_remote_storage"])
        
        registry = {
            "created_at": datetime.now().isoformat(),
            "config": self.config,
            "stats": {
                "total_files": total_files,
                "large_files": large_files,
                "total_size_mb": round(total_size_mb, 2),
                "remote_size_mb": round(remote_size_mb, 2),
                "git_size_reduction": f"{round((remote_size_mb/total_size_mb)*100, 1)}%"
            },
            "models": model_files
        }
        
        return registry
    
    def setup_project(self):
        """Setup completo del proyecto"""
        print("[SETUP] Configurando Model Registry para Kedro...")
        
        # 1. Crear registry
        registry = self.create_model_registry()
        
        # 2. Guardar registry
        with open("model_registry.json", "w", encoding='utf-8') as f:
            json.dump(registry, f, indent=2)
        
        # 3. Crear .gitignore entries
        gitignore_entries = self.create_gitignore_entries(registry["models"])
        
        # 4. Crear script de descarga
        download_script = self.generate_download_script(registry["models"])
        
        # 5. Crear directorio scripts
        os.makedirs("scripts", exist_ok=True)
        
        with open("scripts/download_models.py", "w", encoding='utf-8') as f:
            f.write(download_script)
        
        # 6. Crear docker integration
        docker_info = self.generate_docker_integration()
        
        with open("DOCKER_INTEGRATION.md", "w", encoding='utf-8') as f:
            f.write(docker_info)
        
        # 7. Mostrar resultados
        print(f"""
[SUCCESS] Model Registry configurado!

[STATS] Estadisticas:
   • Total archivos: {registry['stats']['total_files']}
   • Archivos grandes: {registry['stats']['large_files']}
   • Tamaño total: {registry['stats']['total_size_mb']} MB
   • Tamaño remoto: {registry['stats']['remote_size_mb']} MB
   • Reduccion Git: {registry['stats']['git_size_reduction']}

[FILES] Archivos creados:
   • model_registry.json
   • scripts/download_models.py
   • DOCKER_INTEGRATION.md

[NEXT] Proximos pasos:
   1. Agregar a .gitignore:
""")
        
        for entry in gitignore_entries:
            print(f"      {entry}")
        
        print(f"""
   2. Configurar Hugging Face Hub:
      pip install huggingface_hub
      huggingface-cli login
      
   3. Crear repositorio: https://huggingface.co/new
   
   4. Subir modelos iniciales:
      python scripts/upload_models.py
      
   5. Probar descarga:
      python scripts/download_models.py --group=classification
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Setup completo")
    
    args = parser.parse_args()
    
    registry = KedroModelRegistry()
    
    if args.setup:
        registry.setup_project()
    else:
        print("Uso: python kedro_model_registry_setup.py --setup")