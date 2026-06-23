from __future__ import annotations

"""Audit an existing enriched CSV for bad CRM website links.

Usage:
  python -m src.audit_website_quality \
    --input data/output/enriched_companies.csv \
    --output data/output/website_quality_audit.csv \
    --live
"""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import ensure_cache_dirs, get_settings
from src.schemas import SearchResult
from src.scraper import scrape_search_results
from src.site_verifier import classify_url_category, normalize_url, resolve_official_site


def split_urls(value: object) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return []
    urls: list[str] = []
    for part in text.replace("\n", "|").split("|"):
        url = normalize_url(part.strip())
        if url and url not in urls:
            urls.append(url)
    return urls


def candidate_urls(row: pd.Series) -> list[str]:
    urls: list[str] = []
    for col in ("company_website_url", "verified_url", "existing_web", "evidence_urls", "profile_urls"):
        for url in split_urls(row.get(col, "")):
            if url not in urls:
                urls.append(url)
    return urls


def make_results(urls: Iterable[str]) -> list[SearchResult]:
    results: list[SearchResult] = []
    for url in urls:
        category = classify_url_category(url)
        source_type = category if category != "unknown" else "existing_output"
        results.append(SearchResult(title="Existing CSV candidate", url=url, snippet="", source_type=source_type))
    return results


def audit_csv(input_path: Path, output_path: Path, *, live: bool) -> pd.DataFrame:
    settings = get_settings()
    ensure_cache_dirs(settings)
    df = pd.read_csv(input_path)
    rows: list[dict] = []

    for idx, row in df.iterrows():
        company_name = str(row.get("company_name", "") or "").strip()
        if not company_name:
            continue
        record = row.to_dict()
        urls = candidate_urls(row)
        results = make_results(urls)
        pages = scrape_search_results(results, settings=settings) if live and results else []
        resolution = resolve_official_site(
            company_record=record,
            search_results=results,
            scraped_pages=pages,
            min_official_score=float(getattr(settings, "site_min_official_score", 60)),
        )
        old_url = normalize_url(row.get("company_website_url", ""))
        rows.append(
            {
                "row": idx + 1,
                "company_name": company_name,
                "country": row.get("country", ""),
                "old_company_website_url": old_url,
                "new_official_url": resolution.best_url,
                "official_website_confidence": resolution.confidence,
                "site_status": resolution.status,
                "site_rejection_reason": resolution.rejection_reason,
                "old_url_would_be_removed": bool(old_url and old_url != resolution.best_url),
                "profile_urls": " | ".join(resolution.profile_urls),
                "rejected_urls": " | ".join(resolution.rejected_urls),
                "official_site_debug": resolution.debug,
            }
        )

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CRM website URLs and reject profile/platform/dead links.")
    parser.add_argument("--input", default="data/output/enriched_companies.csv")
    parser.add_argument("--output", default="data/output/website_quality_audit.csv")
    parser.add_argument("--live", action="store_true", help="Fetch candidate pages for stronger official-site verification.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = audit_csv(Path(args.input), Path(args.output), live=args.live)
    print(f"Audited rows: {len(out)}")
    print(f"Output: {args.output}")
    if not out.empty:
        print(out["site_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
