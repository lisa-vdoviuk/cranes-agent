from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_enabled: bool = True
    groq_cache_enabled: bool = True
    groq_skip_low_evidence: bool = True
    groq_skip_confident_heuristics: bool = True
    groq_max_search_items: int = 4
    groq_max_pages: int = 2
    groq_page_excerpt_chars: int = 900
    groq_record_notes_chars: int = 400
    groq_max_tokens: int = 700

    max_search_results: int = 5
    max_pages_to_scrape: int = 3
    request_timeout_seconds: int = 15

    cache_dir: Path = Path("data/cache")
    search_cache_path: Path = Path("data/cache/search_cache.json")
    llm_cache_path: Path = Path("data/cache/llm_cache.json")
    page_cache_dir: Path = Path("data/cache/page_cache")

    min_page_text_chars: int = 300
    max_page_text_chars: int = 5000


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def get_settings() -> Settings:
    groq_enabled = _env_bool("GROQ_ENABLED", True)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_enabled and not api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY. Add it to your .env file, or set GROQ_ENABLED=false to run heuristics only."
        )

    settings = Settings(
        groq_api_key=api_key,
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        groq_enabled=groq_enabled,
        groq_cache_enabled=_env_bool("GROQ_CACHE_ENABLED", True),
        groq_skip_low_evidence=_env_bool("GROQ_SKIP_LOW_EVIDENCE", True),
        groq_skip_confident_heuristics=_env_bool("GROQ_SKIP_CONFIDENT_HEURISTICS", True),
        groq_max_search_items=_env_int("GROQ_MAX_SEARCH_ITEMS", 4),
        groq_max_pages=_env_int("GROQ_MAX_PAGES", 2),
        groq_page_excerpt_chars=_env_int("GROQ_PAGE_EXCERPT_CHARS", 900),
        groq_record_notes_chars=_env_int("GROQ_RECORD_NOTES_CHARS", 400),
        groq_max_tokens=_env_int("GROQ_MAX_TOKENS", 700),
        max_search_results=_env_int("MAX_SEARCH_RESULTS", 5),
        max_pages_to_scrape=_env_int("MAX_PAGES_TO_SCRAPE", 3),
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 15),
    )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.page_cache_dir.mkdir(parents=True, exist_ok=True)

    return settings