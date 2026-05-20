from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.classifier.food_classifier import FoodClassifier
from app.config import settings
from app.services.edamam import EdamamService

classifier: FoodClassifier = None
edamam: EdamamService = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, edamam
    classifier = FoodClassifier(
        model_path=settings.model_path,
        labels_path=settings.labels_path,
        top_k=settings.top_k,
        input_size=settings.input_size,
    )
    edamam = EdamamService(app_id=settings.edamam_app_id, app_key=settings.edamam_app_key)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "tflite" if classifier.interpreter else "keras-mobilenetv2",
        "edamam_configured": edamam.is_configured,
    }


@app.post("/analyze")
async def analyze_food(file: UploadFile = File(...)):
    """
    이미지 파일을 받아 음식 분류 + 칼로리 정보를 반환.

    - **file**: JPEG / PNG 이미지
    - 반환: 예측 목록, 최상위 음식명, 신뢰도, 영양 정보(Edamam)
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="image/* 형식의 파일만 허용됩니다.")

    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    predictions = classifier.classify(image_data)
    if not predictions:
        raise HTTPException(status_code=422, detail="이미지를 분류할 수 없습니다.")

    top = predictions[0]
    food_name = top["label"]

    nutrition = None
    if edamam.is_configured:
        try:
            nutrition = await edamam.search_food(food_name)
        except Exception:
            pass  # Edamam 실패는 치명적이지 않음

    return {
        "top_food": food_name,
        "confidence": top["confidence"],
        "predictions": predictions,
        "nutrition": nutrition,
    }
