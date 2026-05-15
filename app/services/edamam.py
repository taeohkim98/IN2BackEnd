import httpx
from typing import Optional


class EdamamService:
    """
    Edamam Food Database API v2 연동.
    https://developer.edamam.com/food-database-api-docs
    """

    BASE_URL = "https://api.edamam.com"

    def __init__(self, app_id: str, app_key: str):
        self.app_id = app_id
        self.app_key = app_key

    @property
    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_key)

    async def search_food(self, food_name: str) -> Optional[dict]:
        """
        음식 이름으로 Edamam에서 영양 정보 조회.
        API 키 미설정 시 None 반환.
        """
        if not self.is_configured:
            return None

        url = f"{self.BASE_URL}/api/food-database/v2/parser"
        params = {
            "ingr": food_name,
            "app_id": self.app_id,
            "app_key": self.app_key,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
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
