#!/usr/bin/env python3
"""
Script para subir modelos pesados a Hugging Face Hub
Uso: python scripts/upload_models.py [--dry-run] [--group=classification]
"""

import os
import json
import argparse
from pathlib import Path
import time

def upload_to_huggingface(repo_id, filepath, commit_message=None):
    """Sube archivo a Hugging Face Hub"""
    try:
        from huggingface_hub import upload_file
        
        if not commit_message:
            filename = os.path.basename(filepath)
            commit_message = f"Upload {filename}"
        
        print(f"[UPLOAD] Subiendo {filepath}...")
        
        result = upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filepath,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message
        )
        
        print(f"[SUCCESS] Subido: {filepath}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error subiendo {filepath}: {e}")
        return False

def upload_models(repo_id, group=None, dry_run=False):
    """Sube modelos según configuración"""
    
    # Cargar registry
    if not os.path.exists("model_registry.json"):
        print("[ERROR] model_registry.json no encontrado")
        return False
    
    with open("model_registry.json", "r", encoding='utf-8') as f:
        registry = json.load(f)
    
    models_to_upload = []
    
    # Filtrar modelos por grupo y tamaño
    for filepath, info in registry["models"].items():
        if not info["needs_remote_storage"]:
            continue
            
        if group and info["model_group"] != group:
            continue
            
        if not os.path.exists(filepath):
            print(f"[SKIP] Archivo no existe: {filepath}")
            continue
            
        models_to_upload.append((filepath, info))
    
    if not models_to_upload:
        print("[INFO] No hay modelos para subir")
        return True
    
    print(f"[INFO] Preparando subir {len(models_to_upload)} archivos a {repo_id}")
    
    # Mostrar resumen
    total_size_mb = sum(info["size_mb"] for _, info in models_to_upload)
    print(f"[INFO] Tamaño total: {total_size_mb:.2f} MB")
    
    if dry_run:
        print("[DRY-RUN] Archivos que se subirían:")
        for filepath, info in models_to_upload:
            print(f"  • {filepath} ({info['size_mb']:.2f} MB)")
        return True
    
    # Confirmar subida
    response = input(f"¿Continuar con la subida? (y/N): ")
    if response.lower() != 'y':
        print("[CANCELLED] Subida cancelada")
        return False
    
    # Subir archivos
    success_count = 0
    for i, (filepath, info) in enumerate(models_to_upload, 1):
        print(f"[PROGRESS] {i}/{len(models_to_upload)}")
        
        commit_msg = f"Upload {os.path.basename(filepath)} ({info['size_mb']:.1f}MB)"
        
        if upload_to_huggingface(repo_id, filepath, commit_msg):
            success_count += 1
            # Pequeña pausa para no saturar
            time.sleep(1)
        else:
            print(f"[ERROR] Falló la subida de {filepath}")
    
    print(f"[SUMMARY] Subidos: {success_count}/{len(models_to_upload)}")
    
    if success_count == len(models_to_upload):
        print("[SUCCESS] Todos los archivos se subieron correctamente!")
        print(f"[INFO] Repositorio: https://huggingface.co/datasets/{repo_id}")
        return True
    else:
        print("[WARNING] Algunos archivos fallaron en la subida")
        return False

def update_registry_config(repo_id):
    """Actualiza configuración del registry con el repo"""
    config_path = "model_registry_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    config["repository"] = repo_id
    config["last_upload"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"[CONFIG] Configuración actualizada: {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="TU-USUARIO/weather-aus-models", 
                       help="Repositorio de Hugging Face (usuario/nombre)")
    parser.add_argument("--group", choices=["classification", "regression"], 
                       help="Solo subir modelos de un grupo")
    parser.add_argument("--dry-run", action="store_true",
                       help="Solo mostrar qué se subiría, sin subir")
    
    args = parser.parse_args()
    
    if "TU-USUARIO" in args.repo:
        print("[ERROR] Debes especificar tu repositorio real:")
        print("python scripts/upload_models.py --repo tu-usuario/weather-aus-models")
        exit(1)
    
    print(f"[START] Iniciando subida a {args.repo}")
    
    success = upload_models(args.repo, args.group, args.dry_run)
    
    if success and not args.dry_run:
        update_registry_config(args.repo)
    
    exit(0 if success else 1)