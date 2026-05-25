import httpx
import logging
from typing import Optional


class EdamamService:
    """
    Edamam Food Database API v2 연동.
    https://developer.edamam.com/food-database-api-docs
    
    Rate Limit 또는 서버 에러 시 None 반환 (fallback).
    """

    BASE_URL = "https://api.edamam.com"

    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key
        self.logger = logging.getLogger(__name__)

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def search_food(self, food_name: str) -> Optional[dict]:
        """
        음식 이름으로 Edamam에서 영양 정보 조회.
        
        실패 시 None 반환:
        - API 키 미설정
        - 429 Too Many Requests (Rate Limit)
        - 503 Service Unavailable
        - 네트워크 오류
        """
        if not self.is_configured:
            return None

        url = f"{self.BASE_URL}/api/food-database/v2/parser"
        params = {
            "ingr": food_name,
            "app_id": self.app_id,
            "app_key": self.app_key,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                
                # Rate Limit 시 fallback
                if response.status_code == 429:
                    self.logger.warning(f"Edamam API Rate Limit (429): {food_name}")
                    return None
                
                # Service Unavailable 시 fallback
                if response.status_code == 503:
                    self.logger.warning(f"Edamam API Service Unavailable (503): {food_name}")
                    return None
                
                response.raise_for_status()
                data = response.json()

            hints = data.get("hints", [])
            if not hints:
                return None

            food = hints[0]["food"]
            nutrients = food.get("nutrients", {})

            return {
                "food_id": food.get("foodId"),
                "label": food.get("label"),
                "category": food.get("category"),
                "calories_per_100g": nutrients.get("ENERC_KCAL"),
                "protein_per_100g": nutrients.get("PROCNT"),
                "fat_per_100g": nutrients.get("FAT"),
                "carbs_per_100g": nutrients.get("CHOCDF"),
                "fiber_per_100g": nutrients.get("FIBTG"),
            }
        
        except httpx.TimeoutException:
            self.logger.warning(f"Edamam API Timeout: {food_name}")
            return None
        except httpx.RequestError as e:
            self.logger.warning(f"Edamam API Network Error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Edamam API Unexpected Error: {e}")
            return None
