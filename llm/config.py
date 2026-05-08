import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "512"))
CACHE_DIR = os.environ.get("CACHE_DIR", "/data/cache")
