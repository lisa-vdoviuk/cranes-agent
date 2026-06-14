from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"

    max_search_results: int = 5
    max_pages_to_scrape: int = 3
    request_timeout_seconds: int = 15

    cache_dir: Path = Path("data/cache")
    search_cache_path: Path = Path("data/cache/search_cache.json")
    page_cache_dir: Path = Path("data/cache/page_cache")

    min_page_text_chars: int = 300
    max_page_text_chars: int = 5000


def get_settings() -> Settings:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to your .env file before running the pipeline."
        )

    settings = Settings(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        max_search_results=int(os.getenv("MAX_SEARCH_RESULTS", "5")),
        max_pages_to_scrape=int(os.getenv("MAX_PAGES_TO_SCRAPE", "3")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
    )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.page_cache_dir.mkdir(parents=True, exist_ok=True)

    return settings