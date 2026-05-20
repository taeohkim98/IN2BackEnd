"""
FastAPI 엔드포인트 통합 테스트.
"""
import io

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


@pytest.fixture(scope="module")
def client():
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
        assert "top_food" in body
        assert "confidence" in body
        assert "predictions" in body
        assert isinstance(body["predictions"], list)
        assert len(body["predictions"]) > 0

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
        """test_images/ 폴더에 실제 음식 사진이 있으면 전체 파이프라인 검증."""
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
            r = client.post(
                "/analyze",
                files={"file": (img_path.name, data, "image/jpeg")},
            )
            assert r.status_code == 200
            body = r.json()
            print(f"\n[{img_path.name}] -> {body['top_food']} ({body['confidence']:.4f})")
            if body["nutrition"]:
                kcal = body["nutrition"].get("calories_per_100g")
                print(f"  칼로리: {kcal} kcal/100g")

    def test_prediction_schema(self, client):
        image_bytes = _make_jpeg_bytes()
        r = client.post(
            "/analyze",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        body = r.json()
        for pred in body["predictions"]:
            assert "label" in pred
            assert "confidence" in pred
            assert 0.0 <= pred["confidence"] <= 1.0
