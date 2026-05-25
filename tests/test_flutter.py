"""
Flutter 클라이언트 관점의 통합 테스트.
Flutter 앱이 전송하는 요청 형식과 응답 스키마를 검증합니다.
TensorFlow 없이도 실행되도록 classifier와 edamam을 mock합니다.
"""
import io
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_jpeg_bytes(width: int, height: int) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 224, height: int = 224) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_PREDICTIONS = [
    {"label": "pizza", "confidence": 0.85},
    {"label": "hamburger", "confidence": 0.10},
    {"label": "sushi", "confidence": 0.05},
]

MOCK_NUTRITION = {
    "calories_per_100g": 266,
    "protein_g": 11.0,
    "fat_g": 10.4,
    "carbs_g": 33.0,
}


@pytest.fixture(scope="module")
def client():
    """Mock된 Classifier와 Edamam으로 테스트 클라이언트 생성."""
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = MOCK_PREDICTIONS
    mock_classifier.interpreter = None

    mock_edamam = MagicMock()
    mock_edamam.is_configured = True
    # search_food는 async이므로 AsyncMock 사용
    mock_edamam.search_food = AsyncMock(return_value=MOCK_NUTRITION)

    with patch("app.main.FoodClassifier", return_value=mock_classifier), \
         patch("app.main.EdamamService", return_value=mock_edamam):
        from app.main import app
        with TestClient(app) as c:
            yield c


class TestCORS:
    """Flutter 웹/모바일이 CORS 없이 API를 호출할 수 있는지 확인."""

    def test_cors_preflight(self, client):
        r = client.options(
            "/analyze",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert r.status_code in (200, 204)
        assert "access-control-allow-origin" in r.headers

    def test_cors_header_on_get(self, client):
        r = client.get("/health", headers={"Origin": "http://localhost:8080"})
        assert "access-control-allow-origin" in r.headers


class TestHealthSchema:
    """Flutter 앱이 파싱하는 /health 응답 스키마를 검증."""

    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_fields(self, client):
        body = client.get("/health").json()
        assert isinstance(body.get("status"), str)
        assert isinstance(body.get("model"), str)
        assert isinstance(body.get("edamam_configured"), bool)


class TestAnalyzeWithFlutterImages:
    """Flutter 카메라/갤러리에서 오는 다양한 해상도 이미지를 처리할 수 있는지 확인."""

    @pytest.mark.parametrize("width,height", [
        (224, 224),    # 정사각형 (모델 입력 크기)
        (640, 480),    # 일반 카메라 해상도
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
    ])
    def test_analyze_various_resolutions(self, client, width, height):
        r = client.post(
            "/analyze",
            files={"file": ("photo.jpg", _make_jpeg_bytes(width, height), "image/jpeg")},
        )
        assert r.status_code == 200

    def test_analyze_png_from_gallery(self, client):
        """Flutter 갤러리에서 PNG 이미지도 처리 가능한지 확인."""
        r = client.post(
            "/analyze",
            files={"file": ("photo.png", _make_png_bytes(), "image/png")},
        )
        assert r.status_code == 200

    def test_analyze_response_schema(self, client):
        """Flutter dart 모델이 기대하는 응답 필드를 검증."""
        r = client.post(
            "/analyze",
            files={"file": ("photo.jpg", _make_jpeg_bytes(224, 224), "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        
        data = body.get("data")
        assert data is not None
        assert isinstance(data.get("top_food"), str)
        assert isinstance(data.get("confidence"), float)
        assert 0.0 <= data["confidence"] <= 1.0

        predictions = data.get("predictions")
        assert isinstance(predictions, list)
        assert len(predictions) > 0
        for pred in predictions:
            assert isinstance(pred.get("label"), str)
            assert isinstance(pred.get("confidence"), float)
            assert 0.0 <= pred["confidence"] <= 1.0

        assert data.get("nutrition") is None or isinstance(data.get("nutrition"), dict)

    def test_analyze_top_food_matches_highest_confidence(self, client):
        """top_food가 predictions 중 confidence가 가장 높은 항목인지 확인."""
        r = client.post(
            "/analyze",
            files={"file": ("photo.jpg", _make_jpeg_bytes(224, 224), "image/jpeg")},
        )
        body = r.json()
        data = body["data"]
        assert data["top_food"] == data["predictions"][0]["label"]
        assert data["confidence"] == data["predictions"][0]["confidence"]

    def test_analyze_nutrition_schema_when_present(self, client):
        """nutrition 필드가 있을 때 Flutter가 파싱할 수 있는 구조인지 확인."""
        r = client.post(
            "/analyze",
            files={"file": ("photo.jpg", _make_jpeg_bytes(224, 224), "image/jpeg")},
        )
        body = r.json()
        data = body.get("data", {})
        nutrition = data.get("nutrition")
        if nutrition is not None:
            assert isinstance(nutrition, dict)

    def test_analyze_rejects_non_image(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("file.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert r.status_code == 400

    def test_analyze_rejects_empty_file(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400
