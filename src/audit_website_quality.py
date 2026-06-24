from __future__ import annotations

"""Audit an existing enriched CSV for bad CRM website links.

Usage:
  python -m src.audit_website_quality \
    --input data/output/enriched_companies.csv \
    --output data/output/website_quality_audit.csv \
    --live

  # Re-audit only previously-accepted rows to catch parked/expired domains:
  python -m src.audit_website_quality \
    --input data/output/enriched_companies.csv \
    --output data/output/website_quality_audit.csv \
    --live --recheck-parked
"""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.config import ensure_cache_dirs, get_settings
from src.excel_loader import load_legacy_excel
from src.parked_domain_detector import check_domain_health
from src.schemas import SearchResult
from src.scraper import _request_headers, scrape_search_results
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


def _live_check_url_for_parking(url: str, timeout: int = 10) -> tuple[bool, str]:
    """Fetch *url* and run the parked-domain detector against the response.

    Returns ``(is_parked, evidence_string)``.  On any fetch error returns
    ``(False, "fetch_error:<exc>")`` so the caller can decide whether to
    treat the URL as dead.
    """
    try:
        resp = requests.get(
            url,
            headers=_request_headers(),
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text
        # Light text extraction: trafilatura not needed here; first 600 chars suffice.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text_sample = soup.get_text(separator=" ", strip=True)[:600]
        result = check_domain_health(str(resp.url), html, text_sample)
        return result.is_parked, f"{result.detection_method}:{result.evidence}" if result.is_parked else ""
    except Exception as exc:
        return False, f"fetch_error:{exc}"


def _load_input_dataframe(input_path: Path, *, limit: int | None = None) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm", ".ods"}:
        return load_legacy_excel(
            input_path,
            country_contains=None,
            combine_duplicates=False,
            limit=limit,
        )

    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding, dtype=str)
            if limit is not None and limit > 0:
                df = df.head(limit).copy()
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is not None:
        raise UnicodeDecodeError(
            last_error.encoding,
            last_error.object,
            last_error.start,
            last_error.end,
            last_error.reason,
        ) from last_error
    df = pd.read_csv(input_path, dtype=str)
    if limit is not None and limit > 0:
        df = df.head(limit).copy()
    return df


def audit_csv(
    input_path: Path,
    output_path: Path,
    *,
    live: bool,
    recheck_parked: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    settings = get_settings()
    ensure_cache_dirs(settings)
    df = _load_input_dataframe(input_path, limit=limit)
    rows: list[dict] = []

    for idx, row in df.iterrows():
        company_name = str(row.get("company_name", "") or "").strip()
        if not company_name:
            continue
        record = row.to_dict()
        urls = candidate_urls(row)
        results = make_results(urls)

        # ── Optional fast parking re-check on previously accepted URLs ────
        parked_urls: set[str] = set()
        parked_evidence: dict[str, str] = {}
        if recheck_parked and urls:
            for url in urls:
                is_parked, evidence = _live_check_url_for_parking(url, timeout=settings.request_timeout_seconds)
                if is_parked:
                    parked_urls.add(url)
                    parked_evidence[url] = evidence

        pages = scrape_search_results(results, settings=settings) if live and results else []
        resolution = resolve_official_site(
            company_record=record,
            search_results=results,
            scraped_pages=pages,
            min_official_score=float(getattr(settings, "site_min_official_score", 60)),
            parked_urls=parked_urls,
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
                "parked_urls_detected": " | ".join(sorted(parked_urls)),
                "parked_evidence": "; ".join(
                    f"{u}={e}" for u, e in sorted(parked_evidence.items())
                ),
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
    parser = argparse.ArgumentParser(description="Audit CRM website URLs and reject profile/platform/dead/parked links.")
    parser.add_argument("--input", default="data/output/enriched_companies.csv")
    parser.add_argument("--output", default="data/output/website_quality_audit.csv")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch candidate pages for stronger official-site verification.",
    )
    parser.add_argument(
        "--recheck-parked",
        action="store_true",
        help=(
            "For each candidate URL, perform a live HTTP fetch and run the "
            "parked-domain detector.  Use this to clean an existing enriched CSV "
            "without rerunning the full pipeline.  Implies network access."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = audit_csv(
        Path(args.input),
        Path(args.output),
        live=args.live,
        recheck_parked=args.recheck_parked,
        limit=args.limit,
    )
    print(f"Audited rows: {len(out)}")
    print(f"Output: {args.output}")
    if not out.empty:
        print(out["site_status"].value_counts(dropna=False).to_string())
        parked_count = out["parked_urls_detected"].astype(bool).sum()
        if parked_count:
            print(f"\nRows with parked/expired URLs detected: {parked_count}")


if __name__ == "__main__":
    main()
