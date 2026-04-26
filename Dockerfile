FROM python:3.11-slim

# Evita archivos .pyc y activa stdout sin buffer (mejora logs en Render)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /api

# Dependencias del sistema requeridas por OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (capa cacheada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY app/ ./app/

# Carpeta donde irá el .pth en producción
# En Render la poblarás via Disk o subiendo el archivo al repo
RUN mkdir -p weights

# Puerto que expone Render por defecto
EXPOSE 10000

# Render pasa PORT como variable de entorno
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]