# Agregar al Dockerfile ANTES de kedro run:

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
