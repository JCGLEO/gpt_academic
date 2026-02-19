from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    embed_model: str = os.getenv("EMBED_MODEL", "models/text-embedding-004")
    data_dir: str = os.getenv("DATA_DIR", "./data/raw")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
