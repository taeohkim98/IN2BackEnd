import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    app_name: str = os.getenv("APP_NAME", "IN2 Food Recognition API")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    model_path: str = os.getenv("MODEL_PATH", "")
    labels_path: str = os.getenv("LABELS_PATH", "")
    top_k: int = int(os.getenv("TOP_K", "5"))
    input_size: int = int(os.getenv("INPUT_SIZE", "224"))
    edamam_app_id: str = os.getenv("EDAMAM_APP_ID", "")
    edamam_app_key: str = os.getenv("EDAMAM_APP_KEY", "")
    edamam_base_url: str = "https://api.edamam.com"


settings = Settings()
