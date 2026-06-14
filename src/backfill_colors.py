from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.color_inference import infer_crane_color_scheme
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
            results.append(SearchResult(title=column, url=url, snippet="Existing CRM URL", source_type="legacy_url", relevance_score=8.0))
    for url in _split_urls(row.get("evidence_urls", "")):
        results.append(SearchResult(title="Evidence URL", url=url, snippet="Existing CRM evidence URL", source_type="search", relevance_score=4.0))
    # De-dupe by URL.
    seen: set[str] = set()
    out: list[SearchResult] = []
    for result in results:
        key = result.url.strip().rstrip("/").lower()
        if key and key not in seen:
            seen.add(key)
            out.append(result)
    return out


def backfill_colors(input_path: str | Path, output_path: str | Path, overwrite: bool = False) -> pd.DataFrame:
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
        if status == "not relevant":
            df.at[idx, "crane_color_scheme"] = "Unknown"
            df.at[idx, "color_confidence"] = 0.0
            df.at[idx, "color_evidence_note"] = "Company is classified as Not Relevant to crane/heavy-lifting activity."
            continue

        current = str(row.get("crane_color_scheme", "") or "").strip().lower()
        if current not in {"", "unknown", "nan", "none", "null"} and not overwrite:
            continue
        record = _row_to_record(row)
        search_results = _row_to_search_results(row)
        page_text = " ".join(
            str(row.get(col, "") or "")
            for col in ("summary", "reasoning_note", "crane_capacity_details", "legacy_info", "original_notes")
        )
        pages = [ScrapedPage(url=str(row.get("verified_url", "") or ""), title=str(row.get("company_name", "") or ""), text=page_text)]
        inferred = infer_crane_color_scheme(record, search_results, pages)
        if inferred.scheme.lower() != "unknown":
            df.at[idx, "crane_color_scheme"] = inferred.scheme
            df.at[idx, "color_confidence"] = inferred.confidence
            df.at[idx, "color_evidence_note"] = inferred.note
            updates += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Backfilled color rows: {updates}")
    print(f"Output: {output_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill deterministic crane color fields without Groq.")
    parser.add_argument("--input", required=True, help="Input enriched CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-Unknown color values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backfill_colors(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
