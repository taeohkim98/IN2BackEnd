# IN2 Food Recognition API

A FastAPI backend that accepts food images from a Flutter app, classifies them using a TFLite (or Keras MobileNetV2) model, and returns nutrition data from the Edamam Food Database API.

## Features

- **Image classification** — runs a TFLite model if one is provided, otherwise falls back to ImageNet-pretrained MobileNetV2 via Keras
- **Nutrition lookup** — queries the Edamam Food Database API for calories, protein, fat, carbs, and fiber per 100 g
- **Flutter-ready response format** — structured JSON with a `success` flag and a `data` envelope
- **Graceful degradation** — Edamam rate limits (429) and outages (503) return a response without nutrition rather than an error

## Requirements

- Python 3.11+
- Dependencies listed in [requirements.txt](requirements.txt)

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy the environment template and fill in your values
cp .env.example .env
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `IN2 Food Recognition API` | API title shown in docs |
| `DEBUG` | `false` | Enable debug mode |
| `MODEL_PATH` | _(empty)_ | Path to `.tflite` model file; leave empty to use Keras fallback |
| `LABELS_PATH` | _(empty)_ | Path to labels text file (one label per line) |
| `TOP_K` | `5` | Number of top predictions to return |
| `INPUT_SIZE` | `224` | Input image size (width × height) in pixels |
| `EDAMAM_APP_ID` | _(empty)_ | Edamam API app ID — get one at [developer.edamam.com](https://developer.edamam.com/) |
| `EDAMAM_APP_KEY` | _(empty)_ | Edamam API app key |

### Model files

Place your TFLite model and labels file in the `models/` directory and set `MODEL_PATH` / `LABELS_PATH` accordingly. If no TFLite model is found, the server automatically downloads and uses the ImageNet MobileNetV2 weights from Keras on first startup.

A helper script is available to download a pre-trained model:

```bash
python scripts/download_model.py
```

## Running the server

```bash
uvicorn app.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

## API reference

### `GET /`
Returns a welcome message.

### `GET /health`
Returns server status, active model type, and whether Edamam is configured.

```json
{
  "status": "ok",
  "model": "tflite",
  "edamam_configured": true
}
```

### `POST /analyze`
Accepts a food image and returns classification predictions with optional nutrition data.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | file | JPEG, PNG, GIF, WebP, or BMP image |

**Response**

```json
{
  "success": true,
  "data": {
    "top_food": "pizza",
    "confidence": 0.8532,
    "predictions": [
      { "label": "pizza", "confidence": 0.8532 },
      { "label": "hamburger", "confidence": 0.0981 }
    ],
    "nutrition": {
      "food_id": "food_a1gb90bazibb6bapkfzjbaxxxx",
      "label": "Pizza",
      "category": "Generic foods",
      "calories_per_100g": 266,
      "protein_per_100g": 11.0,
      "fat_per_100g": 10.4,
      "carbs_per_100g": 33.0,
      "fiber_per_100g": 2.3
    },
    "model_type": "tflite"
  },
  "timestamp": null
}
```

`nutrition` is `null` if Edamam is not configured or the lookup fails.

## Running tests

```bash
pytest
```

Tests mock both the classifier and the Edamam service so no model files or API keys are required.

## Project structure

```
app/
  main.py              # FastAPI app, lifespan setup, route handlers
  config.py            # Settings loaded from environment variables
  classifier/
    food_classifier.py # TFLite / Keras MobileNetV2 inference
  services/
    edamam.py          # Edamam Food Database API client
models/
  food_classifier.tflite
  labels.txt
scripts/
  download_model.py    # Helper to download a pre-trained TFLite model
  train_food_model.py  # Training script
tests/
  test_api.py
  test_classifier.py
  test_edamam.py
  test_flutter.py
```
