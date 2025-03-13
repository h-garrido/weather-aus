# Usa una imagen oficial de Python 3.9
FROM python:3.9

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias y lo instala
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Expone el puerto para Jupyter Notebook
EXPOSE 8888

# Comando para iniciar Jupyter Notebook cuando se ejecute el contenedor
CMD ["kedro", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]