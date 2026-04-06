# Imagen base ligera de Python
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia e instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia lo imprescindible para la API (ajusta según tu estructura de archivos)
COPY api/ ./api/
COPY core/ ./core/
COPY services/ ./services/
COPY main.py .

# Expone el puerto de FastAPI
EXPOSE 8000

# Arranca la API (ajusta si tu archivo principal se llama diferente)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]