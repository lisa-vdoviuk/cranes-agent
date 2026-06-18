from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_model: str = "llama3.1:8b-instruct-q4_K_M"
    ollama_base_url: str = "http://localhost:11434"
    ollama_enabled: bool = True
    ollama_cache_enabled: bool = True
    ollama_skip_low_evidence: bool = True
    ollama_skip_confident_heuristics: bool = True
    ollama_max_search_items: int = 4
    ollama_max_pages: int = 2
    ollama_page_excerpt_chars: int = 900
    ollama_record_notes_chars: int = 400
    ollama_max_tokens: int = 700

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
    ollama_enabled = _env_bool("OLLAMA_ENABLED", True)

    settings = Settings(
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip(),
        ollama_enabled=ollama_enabled,
        ollama_cache_enabled=_env_bool("OLLAMA_CACHE_ENABLED", True),
        ollama_skip_low_evidence=_env_bool("OLLAMA_SKIP_LOW_EVIDENCE", True),
        ollama_skip_confident_heuristics=_env_bool("OLLAMA_SKIP_CONFIDENT_HEURISTICS", True),
        ollama_max_search_items=_env_int("OLLAMA_MAX_SEARCH_ITEMS", 4),
        ollama_max_pages=_env_int("OLLAMA_MAX_PAGES", 2),
        ollama_page_excerpt_chars=_env_int("OLLAMA_PAGE_EXCERPT_CHARS", 900),
        ollama_record_notes_chars=_env_int("OLLAMA_RECORD_NOTES_CHARS", 400),
        ollama_max_tokens=_env_int("OLLAMA_MAX_TOKENS", 700),
        max_search_results=_env_int("MAX_SEARCH_RESULTS", 5),
        max_pages_to_scrape=_env_int("MAX_PAGES_TO_SCRAPE", 3),
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 15),
    )

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.page_cache_dir.mkdir(parents=True, exist_ok=True)

    return settings