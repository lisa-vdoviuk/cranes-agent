from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared constants used across llm_ollama.py, color_inference.py, scraper.py
# ---------------------------------------------------------------------------

CRANE_TERMS: list[str] = [
    "mobilkran", "autokran", "kranverleih", "kran", "crane", "mobile crane",
    "crawler crane", "liebherr", "grove", "demag", "tadano", "faun", "sennebogen",
    "lifting", "heavy transport", "schwertransport", "arbeitsbühne", "hebebühne",
    "teleskopkran", "fahrzeugkran", "raupenkran", "all terrain",
    "ltm", "gmk", "atf", "terex", "fuhrpark", "fleet", "used crane", "gebrauchtkran",
]

# Image <img> alt/src/title terms that hint the image shows a crane or fleet.
IMAGE_RELEVANCE_TERMS: list[str] = [
    "kran", "crane", "mobilkran", "autokran", "mobile-crane", "mobile_crane",
    "crawler", "teleskop", "liebherr", "tadano", "demag", "grove", "faun",
    "sennebogen", "fleet", "fuhrpark", "vermietung", "equipment", "maschine",
    "ltm", "gmk", "atf",
]
