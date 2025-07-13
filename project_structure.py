#!/usr/bin/env python3
"""
Script mejorado para mostrar TODA la estructura del proyecto
Uso: python project_structure_complete.py
"""

import os

def print_complete_structure(startpath, max_depth=10):
    """
    Imprime la estructura completa de directorios sin filtros restrictivos
    """
    print(f"📁 Estructura COMPLETA del proyecto: {os.path.basename(startpath)}")
    print("=" * 80)
    
    for root, dirs, files in os.walk(startpath):
        # Filtrar solo .git para evitar spam masivo
        dirs[:] = [d for d in dirs if not d.startswith('.git')]
        
        level = root.replace(startpath, '').count(os.sep)
        if level < max_depth:
            indent = '│   ' * level
            folder_name = os.path.basename(root)
            
            if level == 0:
                print(f"📁 {folder_name}/")
            else:
                print(f"{indent[:-4]}├── 📁 {folder_name}/")
            
            # Mostrar TODOS los archivos
            subindent = '│   ' * (level + 1)
            
            # Separar archivos por tipo
            python_files = [f for f in files if f.endswith('.py')]
            data_files = [f for f in files if f.endswith(('.csv', '.json', '.pkl', '.pickle', '.joblib', '.parquet'))]
            config_files = [f for f in files if f.endswith(('.yml', '.yaml', '.toml', '.cfg', '.ini', '.conf'))]
            doc_files = [f for f in files if f.endswith(('.md', '.txt', '.rst', '.html'))]
            other_files = [f for f in files if f not in python_files + data_files + config_files + doc_files 
                          and not f.startswith('.') and not f.endswith(('.pyc', '.pyo'))]
            
            # Mostrar archivos por categoría
            all_files = [
                ('🐍', python_files),
                ('📊', data_files), 
                ('⚙️', config_files),
                ('📝', doc_files),
                ('📄', other_files)
            ]
            
            file_count = 0
            for icon, file_list in all_files:
                for file in file_list:
                    file_count += 1
                    if file_count <= 25:  # Mostrar hasta 25 archivos por directorio
                        print(f"{subindent[:-4]}├── {icon} {file}")
                    elif file_count == 26:
                        remaining = sum(len(fl) for _, fl in all_files) - 25
                        print(f"{subindent[:-4]}└── ... y {remaining} archivos más")
                        break
                if file_count > 25:
                    break
            
            # Mostrar conteo total si hay muchos archivos
            total_files = sum(len(fl) for _, fl in all_files)
            if total_files > 25:
                print(f"{subindent[:-4]}📊 Total: {total_files} archivos en este directorio")
    
    print("\n" + "=" * 80)
    print("Leyenda:")
    print("🐍 Python | 📊 Datos/Modelos | ⚙️ Configuración | 📝 Documentación | 📄 Otros")

def show_specific_extensions(startpath, extensions=['.pkl', '.pickle', '.joblib']):
    """
    Muestra específicamente archivos con ciertas extensiones
    """
    print(f"\n🔍 Archivos de modelos encontrados:")
    print("-" * 50)
    
    model_files = []
    for root, dirs, files in os.walk(startpath):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                rel_path = os.path.relpath(os.path.join(root, file), startpath)
                size = os.path.getsize(os.path.join(root, file))
                size_mb = size / (1024 * 1024)
                model_files.append((rel_path, size_mb))
    
    if model_files:
        for file_path, size_mb in sorted(model_files):
            print(f"📊 {file_path} ({size_mb:.2f} MB)")
        print(f"\n📈 Total: {len(model_files)} archivos de modelo")
    else:
        print("❌ No se encontraron archivos de modelo")

def show_directory_sizes(startpath):
    """
    Muestra el tamaño de los directorios principales
    """
    print(f"\n💾 Tamaños de directorios:")
    print("-" * 30)
    
    dir_sizes = {}
    for root, dirs, files in os.walk(startpath):
        if root == startpath:  # Solo directorios de primer nivel
            for directory in dirs:
                if not directory.startswith('.'):
                    dir_path = os.path.join(root, directory)
                    total_size = 0
                    file_count = 0
                    for subroot, subdirs, subfiles in os.walk(dir_path):
                        for file in subfiles:
                            try:
                                file_path = os.path.join(subroot, file)
                                total_size += os.path.getsize(file_path)
                                file_count += 1
                            except:
                                pass
                    dir_sizes[directory] = (total_size / (1024 * 1024), file_count)
    
    for directory, (size_mb, count) in sorted(dir_sizes.items(), key=lambda x: x[1][0], reverse=True):
        print(f"📁 {directory:<20} {size_mb:>8.2f} MB ({count} archivos)")

if __name__ == "__main__":
    current_dir = os.getcwd()
    
    # Mostrar estructura completa
    print_complete_structure(current_dir, max_depth=8)
    
    # Mostrar archivos de modelo específicamente
    show_specific_extensions(current_dir)
    
    # Mostrar tamaños de directorios
    show_directory_sizes(current_dir)
    
    print(f"\n💡 Para explorar un directorio específico:")
    print("python project_structure_complete.py [ruta_directorio]")