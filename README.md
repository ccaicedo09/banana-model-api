# 🌿 Banana Leaf Disease API

API REST construida con **FastAPI + PyTorch** para clasificar enfermedades foliares
en banano y generar mapas **Grad-CAM** de explicabilidad.

## Clases
| Índice | Clase |
|--------|-------|
| 0 | cordana |
| 1 | pestalotiopsis |
| 2 | sana |
| 3 | sigatoka |

---

## Estructura del proyecto

```
banana-disease-api/
├── app/
│   ├── __init__.py
│   ├── main.py        ← FastAPI app, endpoints
│   ├── model.py       ← carga modelo, inferencia, Grad-CAM
│   └── schemas.py     ← Pydantic I/O
├── weights/           ← NO se sube a git — coloca model.pth aquí
├── Dockerfile
├── render.yaml
├── requirements.txt
└── README.md
```

---

## Configuración local

### 1. Clonar y crear entorno

```bash
git clone <tu-repo>
cd banana-disease-api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Colocar los pesos del modelo

Copia el archivo `exp_03_best.pth` generado por el entrenamiento:

```bash
mkdir -p weights
cp /ruta/a/exp_03_best.pth weights/model.pth
```

O define la variable de entorno apuntando a otra ruta:

```bash
export MODEL_PATH=/ruta/absoluta/exp_03_best.pth
```

### 3. Levantar el servidor

```bash
uvicorn app.main:app --reload --port 8000
```

Abre **http://localhost:8000/docs** para la UI interactiva de Swagger.

---

## Endpoint principal

### `POST /predict`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `file` | `UploadFile` | Imagen JPG o PNG de la hoja |

#### Respuesta (`200 OK`)

```json
{
  "clase_predicha": "sigatoka",
  "confianza": 0.947,
  "probabilidades": {
    "cordana": 0.021,
    "pestalotiopsis": 0.018,
    "sana": 0.014,
    "sigatoka": 0.947
  },
  "gradcam_b64": "data:image/png;base64,iVBORw0KGgo..."
}
```

#### Ejemplo con `curl`

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@hoja.jpg"
```

#### Ejemplo con JavaScript (Expo)

```js
const formData = new FormData();
formData.append("file", {
  uri: imageUri,
  name: "hoja.jpg",
  type: "image/jpeg",
});

const res = await fetch("https://tu-api.onrender.com/predict", {
  method: "POST",
  body: formData,
});
const data = await res.json();
// data.gradcam_b64 → úsalo directamente en <Image source={{ uri: data.gradcam_b64 }} />
```

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `MODEL_PATH` | `weights/model.pth` | Ruta absoluta al archivo `.pth` |
| `PORT` | `10000` | Puerto HTTP (Render lo inyecta automáticamente) |
