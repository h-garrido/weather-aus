#!/usr/bin/env python3
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
