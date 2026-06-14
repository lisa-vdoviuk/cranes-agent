from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.schemas import ScrapedPage, SearchResult


@dataclass(frozen=True)
class ColorInference:
    scheme: str = "Unknown"
    confidence: float = 0.0
    note: str = "No reliable color evidence was available."


CRANE_CONTEXT_TERMS = [
    "kran", "crane", "mobilkran", "autokran", "teleskopkran", "fahrzeugkran",
    "crawler", "raupenkran", "all terrain", "ltm", "gmk", "atf", "ac ", "cc ",
    "demag", "tadano", "faun", "grove", "liebherr", "sennebogen", "terex",
    "kranverleih", "fuhrpark", "fleet", "used crane", "gebrauchtkran",
]

COLOR_VARIANTS = {
    "blue": "blue", "blau": "blue", "blauer": "blue", "blaue": "blue", "blauen": "blue", "blauem": "blue",
    "red": "red", "rot": "red", "roter": "red", "rote": "red", "roten": "red", "rotem": "red",
    "yellow": "yellow", "gelb": "yellow", "gelber": "yellow", "gelbe": "yellow", "gelben": "yellow", "gelbem": "yellow",
    "orange": "orange", "oranger": "orange", "orangene": "orange", "orangenen": "orange",
    "white": "white", "weiß": "white", "weiss": "white", "weißer": "white", "weisser": "white", "weiße": "white", "weisse": "white", "weißen": "white", "weissen": "white",
    "black": "black", "schwarz": "black", "schwarzer": "black", "schwarze": "black", "schwarzen": "black", "schwarzem": "black",
    "green": "green", "grün": "green", "gruen": "green", "grüner": "green", "gruener": "green", "grüne": "green", "gruene": "green", "grünen": "green", "gruenen": "green",
    "gray": "grey", "grey": "grey", "grau": "grey", "grauer": "grey", "graue": "grey", "grauen": "grey", "silber": "silver", "silver": "silver",
}

COLOR_WORD_PATTERN = "|".join(sorted((re.escape(key) for key in COLOR_VARIANTS), key=len, reverse=True))
PART_PATTERN = r"(?:boom|ausleger|jib|main(?: body)?|oberwagen|unterwagen|body|counterweight|gegengewicht|chassis|cab|kabine)"
CRANE_OBJECT_PATTERN = r"(?:mobilkran(?:e|en)?|autokran(?:e|en)?|teleskopkran(?:e|en)?|kran(?:e|en)?|crane(?:s)?|crawler crane(?:s)?|raupenkran(?:e|en)?|fleet|fuhrpark|vehicle(?:s)?|fahrzeug(?:e|en)?|machine(?:s)?|maschine(?:n)?)"

STRUCTURED_COLOR_PATTERNS = [
    re.compile(rf"\b(?P<part>{PART_PATTERN})\s*(?:-|:|=|is|ist|in)\s*(?P<color>{COLOR_WORD_PATTERN})\b", re.I),
    re.compile(rf"\b(?P<color>{COLOR_WORD_PATTERN})\s+(?P<part>{PART_PATTERN})\b", re.I),
]

CRANE_COLOR_PATTERNS = [
    re.compile(rf"\b(?P<color>{COLOR_WORD_PATTERN})\s+(?P<object>{CRANE_OBJECT_PATTERN})\b", re.I),
    re.compile(rf"\b(?P<object>{CRANE_OBJECT_PATTERN})\s+(?:in|is|ist|painted|lackiert|farbe|color)\s+(?P<color>{COLOR_WORD_PATTERN})\b", re.I),
]

# Conservative manufacturer/model fallback. These are not proof of a specific fleet paint scheme,
# but they are useful when the only evidence is used-crane model/brand text.
BRAND_LIVERY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"\b(?:liebherr|ltm\s*\d|ltr\s*\d|lrf\s*\d)\b", re.I),
        "typical Liebherr mobile crane livery: boom/main - yellow; counterweight/undercarriage - black or grey",
        "Tentative manufacturer/model livery inferred from Liebherr/LTM evidence, not verified fleet paint.",
    ),
    (
        re.compile(r"\b(?:sennebogen)\b", re.I),
        "typical Sennebogen livery: boom/main - green; accents/counterweight - grey or black",
        "Tentative manufacturer/model livery inferred from Sennebogen evidence, not verified fleet paint.",
    ),
    (
        re.compile(r"\b(?:demag|terex\s*demag|ac\s*\d|cc\s*\d)\b", re.I),
        "typical Demag/Terex-Demag livery: boom/main - white; accents - red/orange; undercarriage - black or grey",
        "Tentative manufacturer/model livery inferred from Demag/Terex-Demag evidence, not verified fleet paint.",
    ),
    (
        re.compile(r"\b(?:tadano|faun|atf\s*\d)\b", re.I),
        "typical Tadano/Faun livery: boom/main - blue or white; accents - red/black depending on model age",
        "Tentative manufacturer/model livery inferred from Tadano/Faun evidence, not verified fleet paint.",
    ),
    (
        re.compile(r"\b(?:grove|gmk\s*\d|manitowoc)\b", re.I),
        "typical Grove/GMK livery: boom/main - blue, grey, or white depending on model/fleet; undercarriage - dark grey/black",
        "Tentative manufacturer/model livery inferred from Grove/GMK evidence, not verified fleet paint.",
    ),
]


def _clean_text(value: object, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def _has_crane_context(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CRANE_CONTEXT_TERMS)


def _normalize_color(value: str) -> str:
    return COLOR_VARIANTS.get(value.lower().strip(), value.lower().strip())


def _sources(company_record: dict[str, Any], search_results: list[SearchResult], scraped_pages: list[ScrapedPage]) -> list[tuple[str, str, str]]:
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
        if page.image_color_hints:
            out.append((page.url, "image_hints", " | ".join(page.image_color_hints[:4])))
    return out


def _extract_structured_color(text: str) -> str:
    assignments: list[str] = []
    for pattern in STRUCTURED_COLOR_PATTERNS:
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
    for pattern in CRANE_COLOR_PATTERNS:
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


def infer_crane_color_scheme(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
) -> ColorInference:
    """Infer crane/equipment color without spending LLM tokens.

    Preference order:
    1. Explicit part/color phrases in page or workbook text.
    2. Explicit colored crane/equipment phrases in relevant text.
    3. Scraper image hints from crane-relevant images.
    4. Low-confidence manufacturer/model livery fallback.
    """
    evidence = _sources(company_record, search_results, scraped_pages)
    combined_text = " ".join(text for _, _, text in evidence)

    if not _has_crane_context(combined_text):
        return ColorInference()

    for source, source_type, text in evidence:
        if not _has_crane_context(text):
            continue
        structured = _extract_structured_color(text)
        if structured:
            return ColorInference(
                scheme=structured,
                confidence=0.68 if source_type == "page_text" else 0.55,
                note=f"Explicit part/color phrase found in {source_type}: {source}",
            )
        phrase = _extract_crane_color_phrase(text)
        if phrase:
            return ColorInference(
                scheme=phrase,
                confidence=0.58 if source_type == "page_text" else 0.45,
                note=f"Explicit crane/equipment color phrase found in {source_type}: {source}",
            )

    image_hints = [text for _, source_type, text in evidence if source_type == "image_hints"]
    if image_hints:
        # Keep this low-confidence: dominant colors may include sky/buildings/logo/background.
        first = image_hints[0]
        colors = re.findall(r"\b(black|white|gray|grey|red|orange|yellow|green|blue|navy|purple|brown)\b", first, flags=re.I)
        unique = []
        for color in colors:
            norm = _normalize_color(color)
            if norm not in unique:
                unique.append(norm)
        if unique:
            return ColorInference(
                scheme="visual hint from crane-related website image: " + "/".join(unique[:4]),
                confidence=0.32,
                note="Low-confidence dominant-color hint from crane-related website image; may include background/logo colors.",
            )

    for pattern, scheme, note in BRAND_LIVERY_RULES:
        if pattern.search(combined_text):
            return ColorInference(
                scheme=scheme,
                confidence=0.28,
                note=note,
            )

    return ColorInference()
