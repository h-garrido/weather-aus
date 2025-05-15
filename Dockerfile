FROM python:3.9

WORKDIR /app

# Copiamos los archivos de requerimientos primero para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install -r requirements.txt

# Instalamos dependencias específicas para la conexión con PostgreSQL
RUN pip install kedro psycopg2-binary
RUN pip install "kedro-datasets[pandas.SQLTableDataSet]"

# Copiamos el resto del código del proyecto
COPY . .

# Comando para mantener el contenedor activo
CMD ["tail", "-f", "/dev/null"]