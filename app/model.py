"""
Model singleton + predictor.

Arquitectura idéntica a la del entrenamiento (exp_03):
  - MobileNetV2 con backbone congelado
  - Classifier: Dropout(0.5) -> Linear(1280, 4)
  - Grad-CAM sobre model.features[18]  (ultimo ConvBNActivation antes del pooling)

NOTA sobre GradCAM en MobileNetV2:
  features[-1][0] puede apuntar a una BatchNorm2d en algunas versiones de
  torchvision, haciendo que grads llegue como None en tiempo de ejecucion.
  La capa segura es features[18]: el bloque ConvBNActivation completo (320->1280).
  requires_grad debe estar activo en esa capa para que los hooks funcionen;
  se reactiva selectivamente despues de cargar los pesos.
"""

from __future__ import annotations

import base64
import io
import logging
import os
from functools import lru_cache
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes (deben coincidir exactamente con el entrenamiento)
# ---------------------------------------------------------------------------
CLASES: list[str] = ["cordana", "pestalotiopsis", "sana", "sigatoka"]
NUM_CLASES = len(CLASES)
IMG_SIZE = 224
DROPOUT = 0.5

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "weights", "model.pth"),
)


# ---------------------------------------------------------------------------
# Construccion del modelo
# ---------------------------------------------------------------------------
def _build_model(dropout: float = DROPOUT) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASES),
    )
    return model


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

_inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(MEAN, STD)],
    std=[1 / s for s in STD],
)


# ---------------------------------------------------------------------------
# Clase predictor
# ---------------------------------------------------------------------------
class BananaPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Dispositivo de inferencia: %s", self.device)

        self.model = _build_model().to(self.device)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"No se encontro el archivo de pesos en '{model_path}'. "
                "Coloca el .pth en la carpeta 'weights/' o define MODEL_PATH."
            )

        # weights_only=True es mas seguro (evita pickle arbitrario)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        logger.info("Pesos cargados desde %s", model_path)

        # Congelar todo el modelo
        for param in self.model.parameters():
            param.requires_grad = False

        # Reactivar gradientes SOLO en la capa target de GradCAM.
        # Sin esto, grads llega como None y GradCAM lanza AttributeError.
        # features[18] = ConvBNActivation(320 -> 1280), ultima capa de features,
        # maxima semantica antes del pooling global.
        self._target_layers = [self.model.features[18]]
        for param in self.model.features[18].parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    def predict(self, image_bytes: bytes) -> dict:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        tensor = _preprocess(pil_img)               # (3, 224, 224)
        batch  = tensor.unsqueeze(0).to(self.device) # (1, 3, 224, 224)

        # Inferencia — sin no_grad para que GradCAM pueda calcular gradientes
        # en la capa target (el resto sigue congelado por requires_grad=False)
        logits = self.model(batch)                   # (1, 4)
        probs  = torch.softmax(logits, dim=1)[0]     # (4,)

        probs_np   = probs.detach().cpu().numpy()
        clase_idx  = int(probs_np.argmax())
        clase_pred = CLASES[clase_idx]
        confianza  = float(probs_np[clase_idx])
        prob_dict: Dict[str, float] = {
            CLASES[i]: round(float(probs_np[i]), 6) for i in range(NUM_CLASES)
        }

        gradcam_b64 = self._generate_gradcam(tensor, clase_idx)

        return {
            "clase_predicha": clase_pred,
            "confianza":      round(confianza, 6),
            "probabilidades": prob_dict,
            "gradcam_b64":    gradcam_b64,
        }

    # ------------------------------------------------------------------
    def _generate_gradcam(self, img_tensor: torch.Tensor, clase_idx: int) -> str:
        with GradCAM(model=self.model, target_layers=self._target_layers) as cam:
            mask = cam(
                input_tensor=img_tensor.unsqueeze(0).to(self.device),
                targets=[ClassifierOutputTarget(clase_idx)],
            )  # (1, H, W)  — valores float en [0, 1]

        img_vis = _inv_normalize(img_tensor)
        img_vis = img_vis.permute(1, 2, 0).cpu().numpy()
        img_vis = np.clip(img_vis, 0.0, 1.0).astype(np.float32)

        cam_image = show_cam_on_image(img_vis, mask[0], use_rgb=True)  # uint8 RGB

        pil_cam = Image.fromarray(cam_image)
        buffer  = io.BytesIO()
        pil_cam.save(buffer, format="PNG", optimize=True)
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_predictor() -> BananaPredictor:
    return BananaPredictor()