from pydantic import BaseModel
from typing import Dict


class PredictionResponse(BaseModel):
    clase_predicha: str
    confianza: float
    probabilidades: Dict[str, float]
    gradcam_b64: str          # "data:image/png;base64,<datos>"

    model_config = {
        "json_schema_extra": {
            "example": {
                "clase_predicha": "sigatoka",
                "confianza": 0.947,
                "probabilidades": {
                    "cordana": 0.021,
                    "pestalotiopsis": 0.018,
                    "sana": 0.014,
                    "sigatoka": 0.947,
                },
                "gradcam_b64": "data:image/png;base64,iVBORw0KGgo...",
            }
        }
    }