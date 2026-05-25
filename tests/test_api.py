"""
FastAPI 엔드포인트 통합 테스트.
"""
import io
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
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


class TestHealthEndpoint:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "model" in body
        assert "edamam_configured" in body


class TestAnalyzeEndpoint:
    def test_analyze_dummy_image(self, client):
        image_bytes = _make_jpeg_bytes()
        r = client.post(
            "/analyze",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        data = body["data"]
        assert "top_food" in data
        assert "confidence" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0

    def test_analyze_rejects_non_image(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert r.status_code == 400

    def test_analyze_rejects_empty_file(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400

    def test_analyze_real_image_if_available(self, client):
        """images/ 폴더에 실제 음식 사진이 있으면 전체 파이프라인 검증 (Mock 사용)."""
        from pathlib import Path

        images = sorted(
            list(Path("images").glob("*.jpg"))
            + list(Path("images").glob("*.jpeg"))
            + list(Path("images").glob("*.png"))
        )
        if not images:
            pytest.skip("images/ 폴더에 이미지 없음")

        for img_path in images:
            data = img_path.read_bytes()
            # mock classifier를 사용하므로 실제 모델은 실행되지 않음
            r = client.post(
                "/analyze",
                files={"file": (img_path.name, data)},
            )
            if r.status_code != 200:
                print(f"\n[{img_path.name}] 업로드 실패: {r.json()}")
                continue
                
            body = r.json()
            result = body["data"]
            print(f"\n[{img_path.name}] -> {result['top_food']} ({result['confidence']:.4f})")
            if result["nutrition"]:
                kcal = result["nutrition"].get("calories_per_100g")
                print(f"  칼로리: {kcal} kcal/100g")

    def test_prediction_schema(self, client):
        image_bytes = _make_jpeg_bytes()
        r = client.post(
            "/analyze",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        body = r.json()
        data = body["data"]
        for pred in data["predictions"]:
            assert "label" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 1.0
