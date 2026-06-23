from __future__ import annotations

"""
color_inference.py — Vision-first crane color detection.

Pipeline (preference order):
  1. Ollama vision model (LLaVA) analyses downloaded crane photographs.
     This is the only reliable way to detect actual physical equipment paint.
  2. Explicit structured part/color phrases in page or workbook text
     (e.g. "boom - yellow; chassis - black").
  3. Explicit colored crane/equipment phrases in text.
  4. Low-confidence manufacturer brand-livery fallback (text-only).

Old approach (Pillow dominant-color pixel counting on arbitrary web images) is
intentionally removed: it measured UI colours, backgrounds, and logos, not
physical equipment paint.
"""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.constants import CRANE_TERMS
from src.schemas import ScrapedPage, SearchResult

try:
    from ollama import Client as OllamaClient
except ImportError:
    OllamaClient = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColorInference:
    scheme: str = "Unknown"
    confidence: float = 0.0
    note: str = "No reliable color evidence was available."


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

COLOR_VARIANTS: dict[str, str] = {
    "blue": "blue", "blau": "blue", "blauer": "blue", "blaue": "blue",
    "blauen": "blue", "blauem": "blue",
    "red": "red", "rot": "red", "roter": "red", "rote": "red",
    "roten": "red", "rotem": "red",
    "yellow": "yellow", "gelb": "yellow", "gelber": "yellow", "gelbe": "yellow",
    "gelben": "yellow", "gelbem": "yellow",
    "orange": "orange", "oranger": "orange", "orangene": "orange", "orangenen": "orange",
    "white": "white", "weiß": "white", "weiss": "white", "weißer": "white",
    "weisser": "white", "weiße": "white", "weisse": "white",
    "weißen": "white", "weissen": "white",
    "black": "black", "schwarz": "black", "schwarzer": "black",
    "schwarze": "black", "schwarzen": "black", "schwarzem": "black",
    "green": "green", "grün": "green", "gruen": "green", "grüner": "green",
    "gruener": "green", "grüne": "green", "gruene": "green",
    "grünen": "green", "gruenen": "green",
    "gray": "grey", "grey": "grey", "grau": "grey", "grauer": "grey",
    "graue": "grey", "grauen": "grey",
    "silber": "silver", "silver": "silver",
}

_COLOR_WORD_PATTERN = "|".join(
    sorted((re.escape(k) for k in COLOR_VARIANTS), key=len, reverse=True)
)
_PART_PATTERN = (
    r"(?:boom|ausleger|jib|main(?: body)?|oberwagen|unterwagen|body"
    r"|counterweight|gegengewicht|chassis|cab|kabine)"
)
_CRANE_OBJECT_PATTERN = (
    r"(?:mobilkran(?:e|en)?|autokran(?:e|en)?|teleskopkran(?:e|en)?"
    r"|kran(?:e|en)?|crane(?:s)?|crawler crane(?:s)?|raupenkran(?:e|en)?"
    r"|fleet|fuhrpark|vehicle(?:s)?|fahrzeug(?:e|en)?|machine(?:s)?|maschine(?:n)?)"
)

_STRUCTURED_PATTERNS = [
    re.compile(
        rf"\b(?P<part>{_PART_PATTERN})\s*(?:-|:|=|is|ist|in)\s*(?P<color>{_COLOR_WORD_PATTERN})\b",
        re.I,
    ),
    re.compile(
        rf"\b(?P<color>{_COLOR_WORD_PATTERN})\s+(?P<part>{_PART_PATTERN})\b",
        re.I,
    ),
]

_CRANE_COLOR_PATTERNS = [
    re.compile(
        rf"\b(?P<color>{_COLOR_WORD_PATTERN})\s+(?P<object>{_CRANE_OBJECT_PATTERN})\b",
        re.I,
    ),
    re.compile(
        rf"\b(?P<object>{_CRANE_OBJECT_PATTERN})\s+(?:in|is|ist|painted|lackiert|farbe|color)"
        rf"\s+(?P<color>{_COLOR_WORD_PATTERN})\b",
        re.I,
    ),
]

# Brand livery is the lowest-confidence fallback (text-only evidence that a brand
# was mentioned, not an image of the actual fleet).
_BRAND_LIVERY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(?:liebherr|ltm\s*\d|ltr\s*\d|lrf\s*\d)\b", re.I),
        "typical Liebherr mobile crane livery: boom/main - yellow; counterweight/undercarriage - black or grey",
        "Tentative manufacturer livery inferred from Liebherr/LTM text — not a verified fleet photograph.",
    ),
    (
        re.compile(r"\b(?:sennebogen)\b", re.I),
        "typical Sennebogen livery: boom/main - green; accents/counterweight - grey or black",
        "Tentative manufacturer livery inferred from Sennebogen text — not a verified fleet photograph.",
    ),
    (
        re.compile(r"\b(?:demag|terex\s*demag|ac\s*\d|cc\s*\d)\b", re.I),
        "typical Demag/Terex-Demag livery: boom/main - white; accents - red/orange; undercarriage - black or grey",
        "Tentative manufacturer livery inferred from Demag/Terex-Demag text — not a verified fleet photograph.",
    ),
    (
        re.compile(r"\b(?:tadano|faun|atf\s*\d)\b", re.I),
        "typical Tadano/Faun livery: boom/main - blue or white; accents - red/black depending on model age",
        "Tentative manufacturer livery inferred from Tadano/Faun text — not a verified fleet photograph.",
    ),
    (
        re.compile(r"\b(?:grove|gmk\s*\d|manitowoc)\b", re.I),
        "typical Grove/GMK livery: boom/main - blue, grey, or white depending on model/fleet; undercarriage - dark grey/black",
        "Tentative manufacturer livery inferred from Grove/GMK text — not a verified fleet photograph.",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_text(value: object, max_chars: int = 3000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_chars]


def _has_crane_context(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CRANE_TERMS)


def _normalize_color(value: str) -> str:
    return COLOR_VARIANTS.get(value.lower().strip(), value.lower().strip())


def _text_sources(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
) -> list[tuple[str, str, str]]:
    """Return (source_label, source_type, text) tuples for text-based evidence."""
    out: list[tuple[str, str, str]] = []

    record_text = " ".join(
        _clean_text(company_record.get(key, ""), 1200)
        for key in ("company_name", "legacy_info", "original_notes", "top_tags", "existing_web")
    )
    if record_text.strip():
        out.append(("legacy workbook", "legacy_workbook", record_text))

    for result in search_results[:8]:
        text = _clean_text(f"{result.title}. {result.snippet}", 1200)
        if text:
            out.append((result.url or "search result", "search", text))

    for page in scraped_pages[:4]:
        text = _clean_text(f"{page.title}. {page.text}", 2500)
        if text:
            out.append((page.url, "page_text", text))

    return out


def _extract_structured_color(text: str) -> str:
    assignments: list[str] = []
    for pattern in _STRUCTURED_PATTERNS:
        for match in pattern.finditer(text):
            part = re.sub(r"\s+", " ", match.group("part").lower()).strip()
            color = _normalize_color(match.group("color"))
            item = f"{part} - {color}"
            if item not in assignments:
                assignments.append(item)
            if len(assignments) >= 4:
                break
        if assignments:
            break
    return "; ".join(assignments)


def _extract_crane_color_phrase(text: str) -> str:
    colors: list[str] = []
    for pattern in _CRANE_COLOR_PATTERNS:
        for match in pattern.finditer(text):
            color = _normalize_color(match.group("color"))
            if color not in colors:
                colors.append(color)
            if len(colors) >= 3:
                break
        if colors:
            break
    if not colors:
        return ""
    if len(colors) == 1:
        return f"crane/equipment - {colors[0]}"
    return "crane/equipment - " + "/".join(colors)


# ---------------------------------------------------------------------------
# Vision model (LLaVA via Ollama) — core new capability
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = (
    "You are an expert at identifying the paint colours of industrial mobile cranes "
    "from photographs. Focus only on the physical equipment, not the background, sky, "
    "ground, or website UI elements."
)

_VISION_USER_PROMPT = (
    "Examine this image carefully.\n\n"
    "If this image shows a real mobile crane or heavy lifting equipment, describe its "
    "physical paint colours. Use this JSON format:\n"
    '{"is_crane_image": true, "colors": [{"part": "boom", "color": "yellow"}, '
    '{"part": "chassis", "color": "black"}], "confidence": 0.85, '
    '"note": "Yellow Liebherr all-terrain crane on a construction site."}\n\n'
    "If this is NOT a crane image (it shows a logo, website screenshot, person, "
    "generic landscape, or anything other than physical crane equipment), return:\n"
    '{"is_crane_image": false, "colors": [], "confidence": 0.0, "note": "Not a crane image."}\n\n'
    "Return only valid JSON, no other text."
)

_VISION_CACHE_VERSION = "v1"


def _vision_cache_path(image_path: Path, vision_model: str, settings: Any) -> Path:
    from src.utils import safe_hash
    key = safe_hash(f"{_VISION_CACHE_VERSION}::{vision_model}::{image_path.name}")
    return settings.image_cache_dir / f"vision_{key}.json"


def _load_vision_cache(image_path: Path, vision_model: str, settings: Any) -> dict | None:
    path = _vision_cache_path(image_path, vision_model, settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_vision_cache(image_path: Path, vision_model: str, settings: Any, result: dict) -> None:
    path = _vision_cache_path(image_path, vision_model, settings)
    try:
        path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _analyse_image_with_vision(
    image_path: Path,
    settings: Any,
) -> dict | None:
    """
    Send a local crane image to the Ollama vision model (LLaVA) for colour analysis.
    Returns the parsed JSON dict from the model, or None on any error.

    The model is asked to return:
      {
        "is_crane_image": bool,
        "colors": [{"part": str, "color": str}, ...],
        "confidence": float,
        "note": str
      }
    """
    if OllamaClient is None:
        print("    [VISION] ollama package not installed; skipping vision analysis.")
        return None

    # Check vision cache first.
    cached = _load_vision_cache(image_path, settings.vision_model, settings)
    if cached is not None:
        return cached

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        print(f"    [VISION] Cannot read image {image_path}: {exc}")
        return None

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    try:
        client = OllamaClient(host=settings.llm_base_url)
        response = client.chat(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": _VISION_USER_PROMPT,
                    "images": [image_b64],
                }
            ],
            format="json",
            stream=False,
        )
        raw_content = response.get("message", {}).get("content", "")
        if not raw_content:
            return None

        result = json.loads(raw_content)
        _save_vision_cache(image_path, settings.vision_model, settings, result)
        return result

    except Exception as exc:
        print(f"    [VISION] Vision model error for {image_path.name}: {exc}")
        return None


def _colours_from_vision_results(
    vision_results: list[dict],
) -> tuple[str, float, str]:
    """
    Aggregate colour results across multiple vision calls into a single scheme string.
    Returns (scheme, confidence, note).
    """
    all_parts: list[dict] = []
    confidence_sum = 0.0
    valid_count = 0
    notes: list[str] = []

    for result in vision_results:
        if not result.get("is_crane_image"):
            continue
        confidence_sum += float(result.get("confidence", 0.0))
        valid_count += 1
        for entry in result.get("colors", []):
            part = str(entry.get("part", "")).strip().lower()
            color = _normalize_color(str(entry.get("color", "")).strip())
            if part and color and color != "unknown":
                all_parts.append({"part": part, "color": color})
        note_text = str(result.get("note", "")).strip()
        if note_text and note_text not in notes:
            notes.append(note_text)

    if not all_parts or valid_count == 0:
        return "Unknown", 0.0, "Vision model found no crane images among the downloaded photographs."

    # Merge duplicate part assignments (keep most frequent color per part).
    part_color_counts: dict[str, dict[str, int]] = {}
    for entry in all_parts:
        p, c = entry["part"], entry["color"]
        part_color_counts.setdefault(p, {})
        part_color_counts[p][c] = part_color_counts[p].get(c, 0) + 1

    scheme_parts: list[str] = []
    for part, color_counts in sorted(part_color_counts.items()):
        best_color = max(color_counts, key=lambda c: color_counts[c])
        scheme_parts.append(f"{part} - {best_color}")

    scheme = "; ".join(scheme_parts)
    avg_confidence = min(confidence_sum / valid_count, 1.0)
    note = "Vision model (LLaVA) analysis of actual crane photographs. " + " | ".join(notes[:3])

    return scheme, avg_confidence, note


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_crane_color_scheme(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Any | None = None,
) -> ColorInference:
    """
    Infer the physical crane/equipment color scheme using a four-tier pipeline:

    Tier 1 — Vision LLM on downloaded crane images (highest confidence, ~0.75–0.90).
              Requires settings.vision_enabled = True and the LLaVA model to be
              available in Ollama. The scraper must have populated page.crane_image_urls.

    Tier 2 — Structured part/color text patterns (e.g. "boom: yellow; chassis: black").

    Tier 3 — Unstructured crane color phrases (e.g. "yellow cranes").

    Tier 4 — Manufacturer brand-livery fallback (low confidence ~0.28).
              Only reached when no images or explicit text exist.
    """

    # ---- Tier 1: Vision model ----
    if settings is not None and getattr(settings, "vision_enabled", False):
        image_urls_to_analyse: list[str] = []
        for page in scraped_pages:
            for img_url in page.crane_image_urls:
                if img_url not in image_urls_to_analyse:
                    image_urls_to_analyse.append(img_url)
                if len(image_urls_to_analyse) >= settings.vision_max_images_per_company:
                    break
            if len(image_urls_to_analyse) >= settings.vision_max_images_per_company:
                break

        if image_urls_to_analyse:
            from src.scraper import download_crane_image
            vision_results: list[dict] = []

            for img_url in image_urls_to_analyse:
                image_path = download_crane_image(img_url, settings)
                if image_path is None:
                    continue
                result = _analyse_image_with_vision(image_path, settings)
                if result is not None:
                    vision_results.append(result)

            if vision_results:
                scheme, confidence, note = _colours_from_vision_results(vision_results)
                if scheme.lower() != "unknown" and confidence > 0.0:
                    return ColorInference(scheme=scheme, confidence=confidence, note=note)

    # ---- Tier 2 & 3: Text-based extraction ----
    text_sources = _text_sources(company_record, search_results, scraped_pages)
    combined_text = " ".join(text for _, _, text in text_sources)

    if not _has_crane_context(combined_text):
        return ColorInference()

    for source, source_type, text in text_sources:
        if not _has_crane_context(text):
            continue

        structured = _extract_structured_color(text)
        if structured:
            conf = 0.68 if source_type == "page_text" else 0.55
            return ColorInference(
                scheme=structured,
                confidence=conf,
                note=f"Explicit part/color phrase found in {source_type}: {source}",
            )

        phrase = _extract_crane_color_phrase(text)
        if phrase:
            conf = 0.58 if source_type == "page_text" else 0.45
            return ColorInference(
                scheme=phrase,
                confidence=conf,
                note=f"Explicit crane/equipment color phrase found in {source_type}: {source}",
            )

    # ---- Tier 4: Brand livery fallback ----
    for pattern, scheme, note in _BRAND_LIVERY_RULES:
        if pattern.search(combined_text):
            return ColorInference(scheme=scheme, confidence=0.28, note=note)

    return ColorInference()
