from contextlib import asynccontextmanager
from pathlib import Path
from PIL import Image
import io

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

@app.get("/")
def root():
    return {"message": "Welcome to the Food Classifier API."}

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
    Flutter 앱에서 음식 이미지를 분석하고 영양 정보를 반환.
    
    - **file**: JPEG / PNG 이미지 파일
    - 반환: 예측 목록, 최상위 음식명, 신뢰도, 영양 정보
    """
    # ===== 원래 코드 (참고용) =====
    # if not file.content_type or not file.content_type.startswith("image/"):
    #     raise HTTPException(status_code=400, detail="image/* 형식의 파일만 허용됩니다.")
    #
    # image_data = await file.read()
    # if not image_data:
    #     raise HTTPException(status_code=400, detail="빈 파일입니다.")
    #
    # predictions = classifier.classify(image_data)
    # if not predictions:
    #     raise HTTPException(status_code=422, detail="이미지를 분류할 수 없습니다.")
    #
    # top = predictions[0]
    # food_name = top["label"]
    #
    # nutrition = None
    # if edamam.is_configured:
    #     try:
    #         nutrition = await edamam.search_food(food_name)
    #     except Exception:
    #         pass  # Edamam 실패는 치명적이지 않음
    #
    # return {
    #     "top_food": food_name,
    #     "confidence": top["confidence"],
    #     "predictions": predictions,
    #     "nutrition": nutrition,
    # }
    # ===== Flutter 통신 버전 =====
    
    # 1단계: 파일명 기반 확장자 검증
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식입니다. ({file_ext}) 허용: {', '.join(valid_extensions)}"
        )

    # 2단계: 이미지 데이터 읽기
    image_data = await file.read()
    if not image_data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    
    # 3단계: Magic bytes 검증 (실제 이미지 파일 확인) - PIL 사용
    try:
        Image.open(io.BytesIO(image_data))
    except (IOError, OSError):
        raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일입니다.")

    # 음식 분류
    predictions = classifier.classify(image_data)
    if not predictions:
        raise HTTPException(status_code=422, detail="이미지를 분류할 수 없습니다.")

    top = predictions[0]
    food_name = top["label"]
    confidence = top["confidence"]

    # 영양 정보 조회
    nutrition = None
    if edamam.is_configured:
        try:
            nutrition = await edamam.search_food(food_name)
        except Exception as e:
            # Edamam API 실패 시 fallback (429 Rate Limit, 503 Service Unavailable 등)
            import logging
            logging.warning(f"Edamam API 호출 실패: {str(e)} - nutrition 정보 없이 응답 진행")
            nutrition = None

    # Flutter 앱용 응답 포맷
    return {
        "success": True,
        "data": {
            "top_food": food_name,
            "confidence": round(float(confidence), 4),
            "predictions": [
                {
                    "label": p["label"],
                    "confidence": round(float(p["confidence"]), 4)
                }
                for p in predictions
            ],
            "nutrition": nutrition,
            "model_type": "tflite" if classifier.interpreter else "keras-mobilenetv2",
        },
        "timestamp": None,  # 필요시 추가: datetime.utcnow().isoformat()
    }
