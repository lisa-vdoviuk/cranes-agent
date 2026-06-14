from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import get_settings
from src.excel_loader import DEFAULT_GERMANY_FILTER, DEFAULT_SHEET, load_legacy_excel
from src.llm_groq import enrich_company_with_llm
from src.scraper import scrape_search_results
from src.search import search_company_web


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


def enrich_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    resume: bool = True,
) -> pd.DataFrame:
    settings = get_settings()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing: pd.DataFrame | None = None
    completed_companies: set[tuple[str, str]] = set()

    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        if "company_name" in existing.columns:
            completed_companies = {_completed_key(row) for _, row in existing.iterrows()}

    enriched_rows: list[dict] = []

    if existing is not None:
        enriched_rows.extend(existing.to_dict(orient="records"))

    total = len(df)

    for idx, row in df.iterrows():
        company_record = row.to_dict()
        company_name = _clean_optional(company_record.get("company_name", ""))
        country = _clean_optional(company_record.get("country", ""))
        existing_web = _clean_optional(company_record.get("existing_web", ""))
        legacy_info = _clean_optional(company_record.get("legacy_info", ""))
        emails = _clean_optional(company_record.get("emails", ""))
        contacts = _clean_optional(company_record.get("contacts", ""))

        if not company_name:
            continue

        key = _completed_key(company_record)
        if resume and key in completed_companies:
            print(f"[SKIP] {company_name} / {country} already enriched.")
            continue

        print(f"\n[{idx + 1}/{total}] Enriching: {company_name} / {country}")

        search_results = search_company_web(
            company_name=company_name,
            country=country,
            existing_web=existing_web,
            legacy_info=legacy_info,
            emails=emails,
            contacts=contacts,
            settings=settings,
        )
        print(f"  Search/direct candidates: {len(search_results)}")
        for result in search_results[:5]:
            print(f"   - score={result.relevance_score:.1f} type={result.source_type} {result.url}")

        scraped_pages = scrape_search_results(search_results, settings=settings)
        print(f"  Scraped pages: {len(scraped_pages)}")

        enrichment = enrich_company_with_llm(
            company_record=company_record,
            search_results=search_results,
            scraped_pages=scraped_pages,
            settings=settings,
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
                "crane_capacity_range": enrichment.crane_capacity_range,
                "crane_capacity_details": enrichment.crane_capacity_details,
                "responsible_sales_contacts": enrichment.responsible_sales_contacts,
                "contact_confidence": enrichment.contact_confidence,
                "contact_source": enrichment.contact_source,
                "crane_color_scheme": enrichment.crane_color_scheme,
                "color_confidence": enrichment.color_confidence,
                "color_evidence_note": enrichment.color_evidence_note,
                "last_checked": datetime.now(timezone.utc).isoformat(),
                "llm_model": settings.groq_model,
                "crm_priority": "",
                "crm_next_action": "",
                "crm_owner_notes": "",
            }
        )

        enriched_rows.append(enriched_row)

        # Save after every company so progress is not lost.
        pd.DataFrame(enriched_rows).to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")

    return pd.DataFrame(enriched_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich the uploaded Spanish/German crane CRM Excel workbook."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .xlsx file, for example data/input/!E-MAIL PEDIDOS MUNDO.xlsx",
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
            f"Excel sheet to process. Can be used multiple times. Default: {DEFAULT_SHEET}. "
            "For this workbook, MUNDO is the main CRM sheet."
        ),
    )
    parser.add_argument(
        "--country-contains",
        default=DEFAULT_GERMANY_FILTER,
        help=(
            "Only process rows whose PAIS contains this text. "
            "Default is 'Alemania' because the workbook stores Germany in Spanish."
        ),
    )
    parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Disable country filtering and process all countries in the selected sheet(s).",
    )
    parser.add_argument(
        "--company-column",
        default=None,
        help="Company column override. For the MUNDO sheet this is EMPRESA.",
    )
    parser.add_argument(
        "--notes-column",
        default=None,
        help="Notes column override. For the MUNDO sheet this is MENSAJE.",
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
        help="Optional test limit after filtering/grouping, for example --limit 10.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from an existing output CSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    country_filter = None if args.all_countries else args.country_contains

    df = load_legacy_excel(
        input_path=args.input,
        sheet_names=args.sheet,
        company_column=args.company_column,
        notes_column=args.notes_column,
        country_contains=country_filter,
        combine_duplicates=not args.no_combine_duplicates,
        limit=args.limit,
    )

    print(f"Loaded {len(df)} company records from {args.input}")
    if country_filter:
        print(f"Country filter: PAIS contains '{country_filter}'")
    print(f"Sheets: {args.sheet or [DEFAULT_SHEET]}")

    enriched = enrich_dataframe(
        df=df,
        output_path=args.output,
        resume=not args.no_resume,
    )

    print("\nDone.")
    print(f"Rows written: {len(enriched)}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
