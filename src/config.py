from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ------------------------------------------------------------------ #
    # Ollama text-LLM settings                                            #
    # ------------------------------------------------------------------ #
    llm_model: str = "llama3.1:8b-instruct-q4_K_M"
    llm_base_url: str = "http://localhost:11434"
    llm_enabled: bool = True
    llm_cache_enabled: bool = True
    llm_skip_low_evidence: bool = True
    llm_skip_confident_heuristics: bool = True
    llm_max_search_items: int = 4
    llm_max_pages: int = 2
    llm_page_excerpt_chars: int = 900
    llm_record_notes_chars: int = 400
    llm_max_tokens: int = 700

    # ------------------------------------------------------------------ #
    # Ollama vision-LLM settings (for crane image colour analysis)        #
    # ------------------------------------------------------------------ #
    vision_model: str = "llava:7b"
    vision_enabled: bool = True
    vision_max_images_per_company: int = 3
    # Images larger than this (bytes) are skipped to keep local inference fast.
    vision_max_image_bytes: int = 5_000_000
    vision_image_timeout_seconds: int = 20

    # ------------------------------------------------------------------ #
    # Web scraping                                                        #
    # ------------------------------------------------------------------ #
    max_search_results: int = 5
    max_pages_to_scrape: int = 3
    request_timeout_seconds: int = 15

    # ------------------------------------------------------------------ #
    # Cache paths (directories are created lazily at first use, not here) #
    # ------------------------------------------------------------------ #
    cache_dir: Path = Path("data/cache")
    search_cache_path: Path = Path("data/cache/search_cache.json")
    llm_cache_path: Path = Path("data/cache/llm_cache.json")
    page_cache_dir: Path = Path("data/cache/page_cache")
    image_cache_dir: Path = Path("data/cache/image_cache")

    # ------------------------------------------------------------------ #
    # Text processing                                                     #
    # ------------------------------------------------------------------ #
    min_page_text_chars: int = 300
    max_page_text_chars: int = 5000

    # ------------------------------------------------------------------ #
    # Official website verification / data cleanliness                    #
    # ------------------------------------------------------------------ #
    official_site_required: bool = True
    site_min_official_score: int = 60
    allow_profile_as_verified_url: bool = False
    max_profile_evidence_urls: int = 2


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
    return Settings(
        llm_model=os.getenv("LLM_MODEL", "llama3.1:8b-instruct-q4_K_M").strip(),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434").strip(),
        llm_enabled=_env_bool("LLM_ENABLED", True),
        llm_cache_enabled=_env_bool("LLM_CACHE_ENABLED", True),
        llm_skip_low_evidence=_env_bool("LLM_SKIP_LOW_EVIDENCE", True),
        llm_skip_confident_heuristics=_env_bool("LLM_SKIP_CONFIDENT_HEURISTICS", True),
        llm_max_search_items=_env_int("LLM_MAX_SEARCH_ITEMS", 4),
        llm_max_pages=_env_int("LLM_MAX_PAGES", 2),
        llm_page_excerpt_chars=_env_int("LLM_PAGE_EXCERPT_CHARS", 900),
        llm_record_notes_chars=_env_int("LLM_RECORD_NOTES_CHARS", 400),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 700),
        vision_model=os.getenv("VISION_MODEL", "llava:7b").strip(),
        vision_enabled=_env_bool("VISION_ENABLED", True),
        vision_max_images_per_company=_env_int("VISION_MAX_IMAGES_PER_COMPANY", 3),
        vision_max_image_bytes=_env_int("VISION_MAX_IMAGE_BYTES", 5_000_000),
        vision_image_timeout_seconds=_env_int("VISION_IMAGE_TIMEOUT_SECONDS", 20),
        max_search_results=_env_int("MAX_SEARCH_RESULTS", 5),
        max_pages_to_scrape=_env_int("MAX_PAGES_TO_SCRAPE", 3),
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 15),
        official_site_required=_env_bool("OFFICIAL_SITE_REQUIRED", True),
        site_min_official_score=_env_int("SITE_MIN_OFFICIAL_SCORE", 60),
        allow_profile_as_verified_url=_env_bool("ALLOW_PROFILE_AS_VERIFIED_URL", False),
        max_profile_evidence_urls=_env_int("MAX_PROFILE_EVIDENCE_URLS", 2),
    )


def ensure_cache_dirs(settings: Settings) -> None:
    """Create all cache directories. Call this once at pipeline startup."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.page_cache_dir.mkdir(parents=True, exist_ok=True)
    settings.image_cache_dir.mkdir(parents=True, exist_ok=True)
