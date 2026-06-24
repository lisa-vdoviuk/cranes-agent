from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import Settings, ensure_cache_dirs, get_settings
from src.excel_loader import DEFAULT_GERMANY_FILTER, DEFAULT_SHEET, load_legacy_excel
from src.llm_ollama import enrich_company_with_llm
from src.scraper import scrape_search_results
from src.search import search_company_web
from src.website_cleaner import clean_enriched_csv, default_clean_output_path

SCHEMA_VERSION = "v4"
LOGGER = logging.getLogger("crane_enrichment")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO", log_file: str | Path | None = None) -> Path:
    """Configure console + file logging and return the file path used."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"enrichment_{stamp}.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    LOGGER.setLevel(numeric_level)
    LOGGER.handlers.clear()
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    LOGGER.info("Logging started. File: %s", log_file)
    return Path(log_file)


def log_settings(settings: Settings) -> None:
    LOGGER.info("Runtime settings:")
    LOGGER.info("  LLM_ENABLED=%s", settings.llm_enabled)
    LOGGER.info("  LLM_MODEL=%s", settings.llm_model)
    LOGGER.info("  LLM_BASE_URL=%s", settings.llm_base_url)
    LOGGER.info("  VISION_ENABLED=%s", settings.vision_enabled)
    LOGGER.info("  VISION_MODEL=%s", settings.vision_model)
    LOGGER.info("  MAX_SEARCH_RESULTS=%s", settings.max_search_results)
    LOGGER.info("  MAX_PAGES_TO_SCRAPE=%s", settings.max_pages_to_scrape)
    LOGGER.info("  LLM_SKIP_LOW_EVIDENCE=%s", settings.llm_skip_low_evidence)
    LOGGER.info("  LLM_SKIP_CONFIDENT_HEURISTICS=%s", settings.llm_skip_confident_heuristics)
    LOGGER.info("  OFFICIAL_SITE_REQUIRED=%s", getattr(settings, "official_site_required", True))
    LOGGER.info("  SITE_MIN_OFFICIAL_SCORE=%s", getattr(settings, "site_min_official_score", 60))
    LOGGER.info("  ALLOW_PROFILE_AS_VERIFIED_URL=%s", getattr(settings, "allow_profile_as_verified_url", False))
    LOGGER.info("  CLEAN_MIN_IDENTITY_SCORE=%s", getattr(settings, "clean_min_identity_score", 50))
    LOGGER.info("  CLEAN_MAX_WEBSITE_CANDIDATES=%s", getattr(settings, "clean_max_website_candidates", 8))


# ---------------------------------------------------------------------------
# Ollama preflight
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OllamaHealth:
    package_installed: bool
    server_reachable: bool
    base_url: str
    available_models: tuple[str, ...] = ()
    llm_model_available: bool = False
    vision_model_available: bool = False
    error: str = ""

    @property
    def ok_for_text_llm(self) -> bool:
        return self.package_installed and self.server_reachable and self.llm_model_available

    @property
    def ok_for_vision_llm(self) -> bool:
        return self.package_installed and self.server_reachable and self.vision_model_available


def _ollama_package_installed() -> bool:
    try:
        import ollama  # noqa: F401
    except ImportError:
        return False
    return True


def _model_is_available(model_name: str, available_models: tuple[str, ...]) -> bool:
    """
    Ollama usually returns exact names like 'llama3.1:8b-instruct-q4_K_M'.
    We still allow a prefix match for cases where the local tag is a shorter alias.
    """
    model_name = (model_name or "").strip()
    if not model_name:
        return False
    if model_name in available_models:
        return True
    requested_base = model_name.split(":", 1)[0]
    return any(name == requested_base or name.startswith(requested_base + ":") for name in available_models)


def check_ollama_health(settings: Settings, timeout_seconds: int = 3) -> OllamaHealth:
    package_installed = _ollama_package_installed()
    if not package_installed:
        return OllamaHealth(
            package_installed=False,
            server_reachable=False,
            base_url=settings.llm_base_url,
            error="Python package 'ollama' is not installed. Run: pip install ollama",
        )

    tags_url = settings.llm_base_url.rstrip("/") + "/api/tags"
    try:
        response = requests.get(tags_url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return OllamaHealth(
            package_installed=True,
            server_reachable=False,
            base_url=settings.llm_base_url,
            error=f"Cannot reach Ollama at {tags_url}: {exc}",
        )

    model_names: list[str] = []
    for item in payload.get("models", []):
        name = str(item.get("name") or item.get("model") or "").strip()
        if name:
            model_names.append(name)

    available = tuple(sorted(set(model_names)))
    return OllamaHealth(
        package_installed=True,
        server_reachable=True,
        base_url=settings.llm_base_url,
        available_models=available,
        llm_model_available=_model_is_available(settings.llm_model, available),
        vision_model_available=_model_is_available(settings.vision_model, available),
        error="" if available else "Ollama is reachable, but no local models were returned by /api/tags.",
    )


def log_ollama_health(health: OllamaHealth, settings: Settings) -> None:
    LOGGER.info("Ollama preflight:")
    LOGGER.info("  package_installed=%s", health.package_installed)
    LOGGER.info("  server_reachable=%s", health.server_reachable)
    LOGGER.info("  base_url=%s", health.base_url)
    if health.available_models:
        LOGGER.info("  available_models=%s", ", ".join(health.available_models))
    else:
        LOGGER.warning("  available_models=<none>")
    LOGGER.info("  requested_text_model=%s available=%s", settings.llm_model, health.llm_model_available)
    LOGGER.info("  requested_vision_model=%s available=%s", settings.vision_model, health.vision_model_available)
    if health.error:
        LOGGER.warning("  health_error=%s", health.error)


def prepare_runtime_settings(
    settings: Settings,
    health: OllamaHealth,
    *,
    auto_disable_ollama_if_down: bool,
    strict_ollama: bool,
) -> tuple[Settings, str]:
    """
    Decide whether to keep Ollama enabled or disable unavailable pieces for this run.

    Why this exists:
      - If LLM_ENABLED=true but Ollama is not running, the original pipeline may either
        crash or save fallback rows.
      - For development runs, it is usually better to log the problem once, disable
        unavailable local models, and continue with deterministic heuristics.
    """
    updates: dict[str, Any] = {}
    notes: list[str] = []

    if settings.llm_enabled:
        if not health.package_installed:
            msg = "LLM requested, but the Python ollama package is not installed."
            notes.append(msg)
            if strict_ollama:
                raise RuntimeError(msg)
            if auto_disable_ollama_if_down:
                updates["llm_enabled"] = False
        elif not health.server_reachable:
            msg = f"LLM requested, but Ollama server is not reachable at {settings.llm_base_url}."
            notes.append(msg)
            if strict_ollama:
                raise RuntimeError(msg)
            if auto_disable_ollama_if_down:
                updates["llm_enabled"] = False
        elif not health.llm_model_available:
            msg = f"LLM requested, but model '{settings.llm_model}' is not installed in Ollama."
            notes.append(msg)
            if strict_ollama:
                raise RuntimeError(msg)
            if auto_disable_ollama_if_down:
                updates["llm_enabled"] = False

    if settings.vision_enabled:
        if not health.package_installed or not health.server_reachable:
            msg = "Vision requested, but Ollama is unavailable."
            notes.append(msg)
            if strict_ollama:
                raise RuntimeError(msg)
            if auto_disable_ollama_if_down:
                updates["vision_enabled"] = False
        elif not health.vision_model_available:
            msg = f"Vision requested, but model '{settings.vision_model}' is not installed in Ollama."
            notes.append(msg)
            if strict_ollama:
                raise RuntimeError(msg)
            if auto_disable_ollama_if_down:
                updates["vision_enabled"] = False

    runtime_settings = replace(settings, **updates) if updates else settings

    if updates:
        LOGGER.warning("Ollama is not fully available. Runtime overrides applied: %s", updates)
        LOGGER.warning("This run will continue, but rows may be heuristic/fallback instead of LLM/vision.")
    elif settings.llm_enabled or settings.vision_enabled:
        LOGGER.info("Ollama is available for all enabled local-model features.")
    else:
        LOGGER.info("LLM and vision are disabled by config; using deterministic enrichment only.")

    return runtime_settings, " | ".join(dict.fromkeys(notes))


# ---------------------------------------------------------------------------
# Enrichment helpers
# ---------------------------------------------------------------------------

def _completed_key(row: pd.Series | dict) -> tuple[str, str]:
    return (
        str(row.get("company_name", "")).strip().lower(),
        str(row.get("country", "")).strip().lower(),
    )


def _clean_optional(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "n/a"}:
        return ""
    return text


def _safe_reason(value: object, limit: int = 500) -> str:
    text = _clean_optional(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _load_existing_rows(
    output_path: Path,
    *,
    resume: bool,
    rerun_fallbacks: bool,
) -> tuple[list[dict], set[tuple[str, str]]]:
    """Load previous output and return rows to keep plus completed keys."""
    if not resume or not output_path.exists():
        return [], set()

    existing = pd.read_csv(output_path)
    if "company_name" not in existing.columns:
        LOGGER.warning("Existing output has no company_name column; ignoring resume file: %s", output_path)
        return [], set()

    if rerun_fallbacks and "enrichment_path" in existing.columns:
        keep = existing[existing["enrichment_path"].astype(str).str.lower() != "fallback"].copy()
        skipped = len(existing) - len(keep)
        LOGGER.warning("--rerun-fallbacks enabled: %s existing fallback rows will be recomputed.", skipped)
    else:
        keep = existing

    completed = {_completed_key(row) for _, row in keep.iterrows()}
    return keep.to_dict(orient="records"), completed


def enrich_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    resume: bool = True,
    rerun_fallbacks: bool = False,
    auto_disable_ollama_if_down: bool = True,
    strict_ollama: bool = False,
) -> pd.DataFrame:
    original_settings = get_settings()
    ensure_cache_dirs(original_settings)

    log_settings(original_settings)
    health = check_ollama_health(original_settings)
    log_ollama_health(health, original_settings)
    settings, runtime_note = prepare_runtime_settings(
        original_settings,
        health,
        auto_disable_ollama_if_down=auto_disable_ollama_if_down,
        strict_ollama=strict_ollama,
    )

    if settings != original_settings:
        LOGGER.info("Effective runtime settings after Ollama preflight:")
        log_settings(settings)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enriched_rows, completed_companies = _load_existing_rows(
        output_path,
        resume=resume,
        rerun_fallbacks=rerun_fallbacks,
    )

    total = len(df)
    LOGGER.info("Starting enrichment for %s input company records.", total)
    LOGGER.info("Resume=%s; already completed=%s; output=%s", resume, len(completed_companies), output_path)

    for idx, row in df.iterrows():
        company_record = row.to_dict()
        company_name = _clean_optional(company_record.get("company_name", ""))
        country = _clean_optional(company_record.get("country", ""))
        existing_web = _clean_optional(company_record.get("existing_web", ""))
        legacy_info = _clean_optional(company_record.get("legacy_info", ""))
        emails = _clean_optional(company_record.get("emails", ""))
        contacts = _clean_optional(company_record.get("contacts", ""))

        if not company_name:
            LOGGER.warning("[%s/%s] Empty company_name; row skipped.", idx + 1, total)
            continue

        key = _completed_key(company_record)
        if resume and key in completed_companies:
            LOGGER.info("[%s/%s] SKIP already enriched: %s / %s", idx + 1, total, company_name, country)
            continue

        LOGGER.info("[%s/%s] Enriching: %s / %s", idx + 1, total, company_name, country)
        LOGGER.debug("Input record: %s", json.dumps(company_record, ensure_ascii=False, default=str))

        search_results = search_company_web(
            company_name=company_name,
            country=country,
            existing_web=existing_web,
            legacy_info=legacy_info,
            emails=emails,
            contacts=contacts,
            settings=settings,
        )
        LOGGER.info("  Search/direct candidates: %s", len(search_results))
        for rank, result in enumerate(search_results[:5], start=1):
            LOGGER.info(
                "    search[%s] score=%.1f type=%s url=%s title=%s",
                rank,
                result.relevance_score,
                result.source_type,
                result.url,
                _safe_reason(result.title, 120),
            )

        scraped_pages = scrape_search_results(search_results, settings=settings)
        LOGGER.info("  Scraped pages: %s", len(scraped_pages))
        for rank, page in enumerate(scraped_pages[:5], start=1):
            LOGGER.info(
                "    page[%s] url=%s chars=%s crane_images=%s title=%s",
                rank,
                page.url,
                len(page.text or ""),
                len(page.crane_image_urls),
                _safe_reason(page.title, 120),
            )

        enrichment = enrich_company_with_llm(
            company_record=company_record,
            search_results=search_results,
            scraped_pages=scraped_pages,
            settings=settings,
        )

        LOGGER.info(
            "  RESULT path=%s status=%s conf=%.0f%% role=%s capacity=%s color=%s color_conf=%.0f%%",
            enrichment.enrichment_path,
            enrichment.ai_status,
            enrichment.status_confidence * 100,
            enrichment.market_role,
            enrichment.crane_capacity_range,
            enrichment.crane_color_scheme,
            enrichment.color_confidence * 100,
        )
        LOGGER.info("  RESULT website=%s verified=%s", enrichment.company_website_url, enrichment.verified_url)
        LOGGER.info("  RESULT reason=%s", _safe_reason(enrichment.reasoning_note, 1000))
        LOGGER.info("  RESULT color_note=%s", _safe_reason(enrichment.color_evidence_note, 500))

        if original_settings.llm_enabled and not health.ok_for_text_llm:
            LOGGER.warning(
                "  OLLAMA-DOWN explanation: LLM was enabled in config, but text model was unavailable. "
                "This row could not use Ollama text classification. reason=%s",
                health.error or f"model '{original_settings.llm_model}' unavailable",
            )
        if original_settings.vision_enabled and not health.ok_for_vision_llm:
            LOGGER.warning(
                "  OLLAMA-DOWN explanation: Vision was enabled in config, but vision model was unavailable. "
                "Crane colors may stay Unknown. reason=%s",
                health.error or f"model '{original_settings.vision_model}' unavailable",
            )

        enriched_row = dict(company_record)
        enriched_row.update(
            {
                "ai_status": enrichment.ai_status,
                "status_confidence": enrichment.status_confidence,
                "market_role": enrichment.market_role,
                "verified_url": enrichment.verified_url,
                "company_website_url": enrichment.company_website_url,
                "summary": enrichment.summary,
                "evidence_urls": " | ".join(enrichment.evidence_urls),
                "reasoning_note": enrichment.reasoning_note,
                "official_website_confidence": enrichment.official_website_confidence,
                "site_status": enrichment.site_status,
                "site_rejection_reason": enrichment.site_rejection_reason,
                "profile_urls": " | ".join(enrichment.profile_urls),
                "rejected_urls": " | ".join(enrichment.rejected_urls),
                "official_site_debug": enrichment.official_site_debug,
                "crane_capacity_range": enrichment.crane_capacity_range,
                "crane_capacity_details": enrichment.crane_capacity_details,
                "responsible_sales_contacts": enrichment.responsible_sales_contacts,
                "contact_confidence": enrichment.contact_confidence,
                "contact_source": enrichment.contact_source,
                "crane_color_scheme": enrichment.crane_color_scheme,
                "color_confidence": enrichment.color_confidence,
                "color_evidence_note": enrichment.color_evidence_note,
                "enrichment_path": enrichment.enrichment_path,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "llm_model": original_settings.llm_model,
                "vision_model": original_settings.vision_model,
                "schema_version": SCHEMA_VERSION,
                "ollama_base_url": original_settings.llm_base_url,
                "ollama_server_reachable": health.server_reachable,
                "ollama_text_model_available": health.llm_model_available,
                "ollama_vision_model_available": health.vision_model_available,
                "runtime_note": runtime_note,
                "crm_priority": "",
                "crm_next_action": "",
                "crm_owner_notes": "",
            }
        )

        enriched_rows.append(enriched_row)
        pd.DataFrame(enriched_rows).to_csv(output_path, index=False)
        LOGGER.info("  Saved -> %s", output_path)

    LOGGER.info("Finished. Total output rows: %s", len(enriched_rows))
    return pd.DataFrame(enriched_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich the crane CRM Excel workbook and optionally create a clean website-validated dataset."
    )

    parser.add_argument(
        "--input",
        required=False,
        help="Path to input .xlsx file, e.g. 'data/input/!E-MAIL PEDIDOS MUNDO.xlsx'. Required unless --clean-only is used.",
    )
    parser.add_argument(
        "--output",
        default="data/output/enriched_companies.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        default=None,
        help=(
            f"Excel sheet to process. Can be repeated. Default: {DEFAULT_SHEET}. "
            "MUNDO is the main CRM sheet."
        ),
    )
    parser.add_argument(
        "--country-contains",
        default=DEFAULT_GERMANY_FILTER,
        help=(
            "Only process rows whose PAIS contains this string. "
            "Default: 'Alemania' (Spanish for Germany)."
        ),
    )
    parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Disable country filtering and process all countries.",
    )
    parser.add_argument(
        "--company-column",
        default=None,
        help="Company column override (default: auto-detect EMPRESA).",
    )
    parser.add_argument(
        "--notes-column",
        default=None,
        help="Notes column override (default: auto-detect MENSAJE).",
    )
    parser.add_argument(
        "--no-combine-duplicates",
        action="store_true",
        help="Do not combine duplicate company/contact rows before enrichment.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N companies for testing, e.g. --limit 10.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing output CSV.",
    )
    parser.add_argument(
        "--rerun-fallbacks",
        action="store_true",
        help="When resuming, recompute rows whose previous enrichment_path was fallback.",
    )
    parser.add_argument(
        "--no-auto-disable-ollama-if-down",
        action="store_true",
        help=(
            "Do not automatically disable LLM/Vision when Ollama is unavailable. "
            "Use this if you want to see the raw failure from lower-level code."
        ),
    )
    parser.add_argument(
        "--strict-ollama",
        action="store_true",
        default=False,
        help=(
            "Fail before processing if LLM/Vision is enabled but Ollama or the requested model is unavailable."
        ),
    )
    parser.add_argument(
        "--clean-websites",
        action="store_true",
        help=(
            "After enrichment, live-validate company_website_url/verified_url/existing_web "
            "and write a second clean CSV where only active official websites remain. "
            "Profile, social, marketplace, parked, expired, unpaid-domain and default-hosting pages are rejected."
        ),
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help=(
            "Skip Excel enrichment and clean an existing enriched CSV. The CSV path is taken "
            "from --output; --clean-output controls where the cleaned CSV is written."
        ),
    )
    parser.add_argument(
        "--clean-output",
        default=None,
        help=(
            "Output path for --clean-websites/--clean-only. Default: same as --output "
            "with '_clean' before .csv, e.g. data/output/enriched_companies_clean.csv."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level. The log file always receives DEBUG+.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log file path. Default: logs/enrichment_YYYYMMDD_HHMMSS.log.",
    )

    return parser.parse_args()


def _run_clean_websites(output_path: str | Path, clean_output: str | Path | None = None) -> Path:
    settings = get_settings()
    input_path = Path(output_path)
    clean_path = Path(clean_output) if clean_output else default_clean_output_path(input_path)
    LOGGER.info("Running clean website validation: input=%s output=%s", input_path, clean_path)
    clean_df = clean_enriched_csv(input_path, clean_path, settings=settings, logger=LOGGER)
    LOGGER.info("Clean website validation finished. Rows=%s output=%s", len(clean_df), clean_path)
    if "clean_website_status" in clean_df.columns:
        counts = clean_df["clean_website_status"].value_counts(dropna=False).to_dict()
        LOGGER.info("Clean website statuses: %s", counts)
    return clean_path


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level, args.log_file)

    if args.clean_only:
        if not Path(args.output).exists():
            raise FileNotFoundError(
                f"--clean-only expects --output to point to an existing enriched CSV. Not found: {args.output}"
            )
        _run_clean_websites(args.output, args.clean_output)
        return

    if not args.input:
        raise SystemExit("--input is required unless --clean-only is used.")

    country_filter = None if args.all_countries else args.country_contains

    LOGGER.info("Loading workbook: %s", args.input)
    df = load_legacy_excel(
        input_path=args.input,
        sheet_names=args.sheet,
        company_column=args.company_column,
        notes_column=args.notes_column,
        country_contains=country_filter,
        combine_duplicates=not args.no_combine_duplicates,
        limit=args.limit,
    )

    LOGGER.info("Loaded %s company records from %s", len(df), args.input)
    if country_filter:
        LOGGER.info("Country filter: PAIS contains '%s'", country_filter)

    enrich_dataframe(
        df=df,
        output_path=args.output,
        resume=not args.no_resume,
        rerun_fallbacks=args.rerun_fallbacks,
        auto_disable_ollama_if_down=not args.no_auto_disable_ollama_if_down,
        strict_ollama=args.strict_ollama,
    )

    if args.clean_websites:
        _run_clean_websites(args.output, args.clean_output)


if __name__ == "__main__":
    main()
