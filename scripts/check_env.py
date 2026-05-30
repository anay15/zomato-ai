from pathlib import Path
from dotenv import load_dotenv
import os

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
print("env file exists:", (ROOT / ".env").exists())
print("provider:", os.getenv("LLM_PROVIDER"))
print("key set:", bool(os.getenv("GROQ_API_KEY", "").strip()))
print("key len:", len(os.getenv("GROQ_API_KEY", "").strip()))
print("model:", os.getenv("GROQ_MODEL"))
