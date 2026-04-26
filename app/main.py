"""
=============================================================================
API — Detección de enfermedades foliares en banano
Modelo : MobileNetV2 (PyTorch)
Clases : cordana | pestalotiopsis | sana | sigatoka
=============================================================================
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.schemas import PredictionResponse
from app.model import get_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: carga el modelo una sola vez al arrancar el servidor
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Cargando modelo MobileNetV2...")
    get_predictor()          # warm-up — instancia el singleton
    logger.info("Modelo listo ✅")
    yield
    logger.info("Apagando servidor...")


app = FastAPI(
    title="Banana Leaf Disease API",
    description="Predicción de enfermedades foliares en banano con Grad-CAM.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ajusta a tu dominio de Expo en producción
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Render usa este endpoint para saber si el servicio está vivo."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen (JPG / PNG) y devuelve:
      - clase_predicha   : nombre de la enfermedad
      - confianza        : probabilidad de la clase ganadora (0-1)
      - probabilidades   : dict {clase: probabilidad} para las 4 clases
      - gradcam_b64      : imagen PNG con heatmap superpuesto en base64
    """
    # Validar tipo de archivo
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   "Usa JPG o PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    predictor = get_predictor()
    result = predictor.predict(image_bytes)
    return result