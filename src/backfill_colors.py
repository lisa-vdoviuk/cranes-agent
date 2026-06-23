from __future__ import annotations

"""
backfill_colors.py — Re-run the vision + text color pipeline on an existing
enriched CSV without re-running the full enrichment pipeline.

Changes vs original:
  - Calls the new infer_crane_color_scheme() which uses the Ollama vision model
    (LLaVA) to analyse real downloaded crane photographs instead of measuring
    Pillow pixel histograms of arbitrary web images.
  - Reconstructs ScrapedPage objects including crane_image_urls from the
    saved evidence_urls column, so the vision model has URLs to work from.
  - Adds --vision-only flag to skip the text-fallback tiers and only accept
    vision model results (useful for quality audits).
  - Docstring updated; "Groq" references removed throughout.
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.color_inference import infer_crane_color_scheme
from src.config import ensure_cache_dirs, get_settings
from src.schemas import ScrapedPage, SearchResult


def _split_urls(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [part.strip() for part in text.replace("\n", "|").split("|") if part.strip()]


def _row_to_record(row: pd.Series) -> dict[str, Any]:
    return {key: ("" if pd.isna(value) else value) for key, value in row.to_dict().items()}


def _row_to_search_results(row: pd.Series) -> list[SearchResult]:
    results: list[SearchResult] = []
    for column in ("company_website_url", "verified_url", "existing_web"):
        url = str(row.get(column, "") or "").strip()
        if url and url.lower() not in {"nan", "none", "null"}:
            results.append(
                SearchResult(
                    title=column,
                    url=url,
                    snippet="Existing CRM URL",
                    source_type="legacy_url",
                    relevance_score=8.0,
                )
            )
    for url in _split_urls(row.get("evidence_urls", "")):
        results.append(
            SearchResult(
                title="Evidence URL",
                url=url,
                snippet="Existing CRM evidence URL",
                source_type="search",
                relevance_score=4.0,
            )
        )
    # Deduplicate by URL.
    seen: set[str] = set()
    out: list[SearchResult] = []
    for result in results:
        key = result.url.strip().rstrip("/").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(result)
    return out


def _row_to_scraped_pages(row: pd.Series) -> list[ScrapedPage]:
    """
    Reconstruct ScrapedPage objects from saved CSV columns.

    The verified_url / evidence_urls columns may point to pages that were
    previously scraped and whose crane_image_urls are already cached in
    data/cache/page_cache/<hash>.json.  The scraper's _load_page_cache()
    will pick those up transparently when the vision pipeline calls
    download_crane_image().

    We also synthesise one page that aggregates the saved text fields so
    the text-tier fallback has something to search.
    """
    pages: list[ScrapedPage] = []

    # Page 1: aggregate text from saved enrichment columns.
    page_text = " ".join(
        str(row.get(col, "") or "")
        for col in (
            "summary",
            "reasoning_note",
            "crane_capacity_details",
            "legacy_info",
            "original_notes",
        )
    )
    verified_url = str(row.get("verified_url", "") or "").strip()
    company_name = str(row.get("company_name", "") or "").strip()

    if page_text.strip():
        pages.append(
            ScrapedPage(
                url=verified_url or "",
                title=company_name,
                text=page_text,
                crane_image_urls=[],
            )
        )

    # Additional pages: one per evidence URL.  Crane image URLs for these pages
    # will be loaded from the JSON page cache on disk when the vision pipeline
    # calls download_crane_image().
    for url in _split_urls(row.get("evidence_urls", "")):
        if url and url != verified_url:
            pages.append(
                ScrapedPage(
                    url=url,
                    title="",
                    text="",
                    # crane_image_urls will be populated from the page cache
                    # by infer_crane_color_scheme() → download_crane_image().
                    crane_image_urls=[],
                )
            )

    # Attempt to hydrate crane_image_urls from the JSON page cache for each page.
    _hydrate_image_urls_from_page_cache(pages)
    return pages


def _hydrate_image_urls_from_page_cache(pages: list[ScrapedPage]) -> None:
    """
    For each page that has a URL but no crane_image_urls, try loading them
    from the JSON page cache written by the scraper.
    """
    import json
    from src.utils import safe_hash

    settings = get_settings()
    page_cache_dir = settings.page_cache_dir

    for page in pages:
        if page.crane_image_urls or not page.url:
            continue
        cache_path = page_cache_dir / f"{safe_hash(page.url)}.json"
        if not cache_path.exists():
            continue
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_urls = data.get("crane_image_urls", [])
            if cached_urls:
                page.crane_image_urls = cached_urls
                if not page.title:
                    page.title = data.get("title", "")
                if not page.text:
                    page.text = data.get("text", "")
        except (json.JSONDecodeError, OSError):
            pass


def backfill_colors(
    input_path: str | Path,
    output_path: str | Path,
    overwrite: bool = False,
    vision_only: bool = False,
) -> pd.DataFrame:
    """
    Re-run color inference on every row in the enriched CSV.

    Args:
        input_path:  Path to the enriched CSV to read.
        output_path: Path to write the updated CSV.
        overwrite:   When True, re-run even on rows that already have a
                     non-Unknown color.
        vision_only: When True, only accept results from the vision model
                     (tiers 2–4 text fallbacks are still available but their
                     results are discarded unless vision produced something).
    """
    settings = get_settings()
    ensure_cache_dirs(settings)

    df = pd.read_csv(input_path)
    for column, default in {
        "crane_color_scheme": "Unknown",
        "color_confidence": 0.0,
        "color_evidence_note": "",
    }.items():
        if column not in df.columns:
            df[column] = default

    updates = 0

    for idx, row in df.iterrows():
        status = str(row.get("ai_status", "") or "").strip().lower()
        company_name = str(row.get("company_name", "") or "").strip()

        # Always clear Not Relevant rows.
        if status == "not relevant":
            df.at[idx, "crane_color_scheme"] = "Unknown"
            df.at[idx, "color_confidence"] = 0.0
            df.at[idx, "color_evidence_note"] = (
                "Company is classified as Not Relevant to crane/heavy-lifting activity."
            )
            continue

        current = str(row.get("crane_color_scheme", "") or "").strip().lower()
        if current not in {"", "unknown", "nan", "none", "null"} and not overwrite:
            continue

        print(f"  [COLOR] {company_name}")

        record = _row_to_record(row)
        search_results = _row_to_search_results(row)
        pages = _row_to_scraped_pages(row)

        inferred = infer_crane_color_scheme(
            company_record=record,
            search_results=search_results,
            scraped_pages=pages,
            settings=settings,
        )

        # In vision_only mode, discard text-tier results (confidence < 0.70
        # is a reliable proxy — vision results consistently score ≥ 0.75).
        if vision_only and inferred.confidence < 0.70:
            print(f"    → Skipped (vision_only mode, confidence {inferred.confidence:.0%})")
            continue

        if inferred.scheme.lower() != "unknown":
            df.at[idx, "crane_color_scheme"] = inferred.scheme
            df.at[idx, "color_confidence"] = inferred.confidence
            df.at[idx, "color_evidence_note"] = inferred.note
            updates += 1
            print(
                f"    → {inferred.scheme!r}  "
                f"({inferred.confidence:.0%})  "
                f"{inferred.note[:80]}"
            )
        else:
            print("    → No reliable color evidence found.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nBackfilled color rows: {updates}")
    print(f"Output: {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill crane color fields using the Ollama vision model (LLaVA). "
            "Reads an existing enriched CSV and re-runs the full color pipeline "
            "(vision → structured text → color phrase → brand livery) on rows "
            "where crane_color_scheme is Unknown or empty."
        )
    )
    parser.add_argument("--input", required=True, help="Input enriched CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-Unknown color values.",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help=(
            "Only accept color results that came from the vision model "
            "(confidence ≥ 0.70). Text-tier results are discarded. "
            "Useful for auditing image quality."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backfill_colors(
        input_path=args.input,
        output_path=args.output,
        overwrite=args.overwrite,
        vision_only=args.vision_only,
    )


if __name__ == "__main__":
    main()
