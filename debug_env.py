# debug_env.py
from dotenv import load_dotenv
import os

loaded = load_dotenv()

print("loaded:", loaded)
print("cwd:", os.getcwd())
print("id:", os.getenv("EDAMAM_APP_ID"))
print("key:", os.getenv("EDAMAM_APP_KEY"))

print("APP_ID =", os.getenv("EDAMAM_APP_ID"))