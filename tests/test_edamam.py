"""
EdamamService 단위 테스트.
실제 API 호출 테스트는 .env에 EDAMAM_APP_ID / EDAMAM_APP_KEY가 설정된 경우에만 실행.
"""
import os

import httpx
import pytest
import pytest_asyncio

from app.services.edamam import EdamamService


@pytest.fixture
def dummy_service() -> EdamamService:
    """API 키 없는 서비스 (is_configured = False)."""
    return EdamamService(app_id="", app_key="")


@pytest.fixture
def real_service() -> EdamamService:
    """실제 API 키가 있는 경우에만 유효."""
    return EdamamService(
        app_id=os.getenv("EDAMAM_APP_ID", ""),
        app_key=os.getenv("EDAMAM_APP_KEY", ""),
    )


class TestEdamamServiceConfig:
    def test_not_configured_without_keys(self, dummy_service):
        assert not dummy_service.is_configured

    def test_configured_with_keys(self):
        svc = EdamamService(app_id="fake_id", app_key="fake_key")
        assert svc.is_configured


class TestEdamamSearchFood:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_configured(self, dummy_service):
        result = await dummy_service.search_food("pizza")
        assert result is None

    @pytest.mark.asyncio
    async def test_real_api_pizza(self, real_service):
        if not real_service.is_configured:
            pytest.skip("EDAMAM_APP_ID / EDAMAM_APP_KEY 환경변수 미설정")

        try:
            result = await real_service.search_food("pizza")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                pytest.skip("Edamam API 키가 유효하지 않음 (401) — Food Database API 키인지 확인 필요")
            raise

        assert result is not None
        assert "label" in result
        assert "calories_per_100g" in result
        assert result["calories_per_100g"] is not None
        print(f"\n[Edamam] pizza: {result['calories_per_100g']} kcal/100g")

    @pytest.mark.asyncio
    async def test_real_api_returns_correct_schema(self, real_service):
        if not real_service.is_configured:
            pytest.skip("EDAMAM_APP_ID / EDAMAM_APP_KEY 환경변수 미설정")

        try:
            result = await real_service.search_food("apple")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                pytest.skip("Edamam API 키가 유효하지 않음 (401) — Food Database API 키인지 확인 필요")
            raise

        assert result is not None
        for key in ("food_id", "label", "calories_per_100g", "protein_per_100g", "fat_per_100g", "carbs_per_100g"):
            assert key in result

    @pytest.mark.asyncio
    async def test_mock_api_response(self, monkeypatch):
        """httpx를 모킹해서 API 키 없이도 파싱 로직 검증."""
        svc = EdamamService(app_id="fake", app_key="fake")

        mock_response = {
            "hints": [
                {
                    "food": {
                        "foodId": "food_abc123",
                        "label": "Pizza",
                        "category": "Generic foods",
                        "nutrients": {
                            "ENERC_KCAL": 266.0,
                            "PROCNT": 11.0,
                            "FAT": 10.0,
                            "CHOCDF": 33.0,
                            "FIBTG": 2.3,
                        },
                    }
                }
            ]
        }

        class MockResponse:
            status_code = 200  # 정상 응답
            
            def raise_for_status(self):
                pass

            def json(self):
                return mock_response

        class MockClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, *args, **kwargs):
                return MockResponse()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: MockClient())

        result = await svc.search_food("pizza")
        assert result["label"] == "Pizza"
        assert result["calories_per_100g"] == 266.0
        assert result["protein_per_100g"] == 11.0
