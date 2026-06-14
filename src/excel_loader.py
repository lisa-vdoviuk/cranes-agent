from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


# This loader is tailored to the uploaded workbook:
# !E-MAIL PEDIDOS MUNDO.xlsx
# Primary CRM sheet: MUNDO
# Primary columns: EMAIL, EMPRESA, CONTACTO, TELEFONO, PAIS, TOP, INFO, WEB, MENSAJE,
#                  ADDED_TIME, MODIFIED_TIME

DEFAULT_SHEET = "MUNDO"
DEFAULT_GERMANY_FILTER = "Alemania"  # The workbook uses Spanish country names.

COMPANY_CANDIDATES = ["EMPRESA", "NOMBRE", "CLIENTES"]
EMAIL_CANDIDATES = ["EMAIL"]
CONTACT_CANDIDATES = ["CONTACTO", "APELLIDO", "NOMBRE"]
PHONE_CANDIDATES = ["TELEFONO", "SMS", "MOVIL"]
COUNTRY_CANDIDATES = ["PAIS"]
INFO_CANDIDATES = ["INFO", "Comentarios", "Unnamed: 5"]
WEB_CANDIDATES = ["WEB"]
NOTES_CANDIDATES = ["MENSAJE", "Comentarios"]
TOP_CANDIDATES = ["TOP"]
ADDED_TIME_CANDIDATES = ["ADDED_TIME"]
MODIFIED_TIME_CANDIDATES = ["MODIFIED_TIME"]


def _clean_header(value: object) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_text(value: object) -> str:
    if value is None:
        return ""

    # pandas can pass NaN-like values even with mixed Excel types.
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass

    text = str(value).strip()

    if text.lower() in {"nan", "nat", "none"}:
        return ""

    # Clean numbers that Excel/Pandas sometimes reads as 967443743.0
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_key(value: object) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9áéíóúüñäöß]+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized_to_original = {_clean_header(col).lower(): col for col in df.columns}
    for candidate in candidates:
        found = normalized_to_original.get(candidate.lower())
        if found is not None:
            return found
    return None


def _unique_join(values: Iterable[object], separator: str = " | ", max_items: int = 30) -> str:
    seen: set[str] = set()
    cleaned: list[str] = []

    for value in values:
        text = _clean_text(value)
        if not text:
            continue

        key = text.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(text)

        if len(cleaned) >= max_items:
            break

    return separator.join(cleaned)


def _drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove accidental header rows embedded inside the sheet data.

    The uploaded MUNDO sheet has a first data row that repeats the column labels.
    This vectorized implementation is much faster than row-wise apply on 64k+ rows.
    """
    header_words = {
        "EMAIL",
        "EMPRESA",
        "CONTACTO",
        "TELEFONO",
        "PAIS",
        "TOP",
        "INFO",
        "WEB",
        "MENSAJE",
        "ADDED_TIME",
        "MODIFIED_TIME",
        "APELLIDO",
        "SMS",
    }

    check_cols = [col for col in df.columns if _clean_header(col).upper() in header_words]
    if not check_cols:
        return df

    upper_values = df[check_cols].astype(str).apply(lambda col: col.str.strip().str.upper())
    repeated_header_score = upper_values.isin(header_words).sum(axis=1)
    return df.loc[repeated_header_score < 4].copy()


def _read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    df = df.rename(columns={col: _clean_header(col) for col in df.columns})

    # The uploaded MUNDO sheet contains a duplicated header row as its first data row.
    df = _drop_repeated_header_rows(df)

    df["source_sheet"] = sheet_name
    # Excel row number: header is row 1, data starts at row 2.
    df["source_excel_row"] = df.index + 2
    return df.reset_index(drop=True)


def _standardize_sheet(
    df: pd.DataFrame,
    *,
    company_column: str | None = None,
    notes_column: str | None = None,
) -> pd.DataFrame:
    company_col = company_column or _find_column(df, COMPANY_CANDIDATES)
    email_col = _find_column(df, EMAIL_CANDIDATES)
    contact_col = _find_column(df, CONTACT_CANDIDATES)
    phone_col = _find_column(df, PHONE_CANDIDATES)
    country_col = _find_column(df, COUNTRY_CANDIDATES)
    info_col = _find_column(df, INFO_CANDIDATES)
    web_col = _find_column(df, WEB_CANDIDATES)
    notes_col = notes_column or _find_column(df, NOTES_CANDIDATES)
    top_col = _find_column(df, TOP_CANDIDATES)
    added_col = _find_column(df, ADDED_TIME_CANDIDATES)
    modified_col = _find_column(df, MODIFIED_TIME_CANDIDATES)

    if company_col is None or company_col not in df.columns:
        raise ValueError(
            "Could not detect a company column in this sheet. "
            "For the uploaded workbook, use sheet 'MUNDO' or pass --company-column EMPRESA."
        )

    def get_col(col: str | None) -> pd.Series:
        if col is None or col not in df.columns:
            return pd.Series([""] * len(df), index=df.index)
        return df[col].map(_clean_text)

    output = pd.DataFrame(
        {
            "source_sheet": get_col("source_sheet"),
            "source_excel_row": get_col("source_excel_row"),
            "company_name": get_col(company_col),
            "country": get_col(country_col),
            "emails": get_col(email_col),
            "contacts": get_col(contact_col),
            "phones": get_col(phone_col),
            "top_tags": get_col(top_col),
            "legacy_info": get_col(info_col),
            "existing_web": get_col(web_col),
            "original_notes": get_col(notes_col),
            "added_time": get_col(added_col),
            "modified_time": get_col(modified_col),
        }
    )

    output["company_name"] = output["company_name"].map(_clean_text)
    output = output[output["company_name"] != ""].copy()

    output["company_key"] = output["company_name"].map(_normalize_key)
    output["country_key"] = output["country"].map(_normalize_key)

    return output.reset_index(drop=True)


def _combine_duplicate_companies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    grouped_rows: list[dict[str, object]] = []
    group_cols = ["company_key", "country_key"]

    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        first = group.iloc[0]
        row = {
            "source_sheets": _unique_join(group["source_sheet"], max_items=10),
            "source_row_count": int(len(group)),
            "source_excel_rows": _unique_join(group["source_excel_row"], max_items=50),
            "company_name": first["company_name"],
            "country": first["country"],
            "emails": _unique_join(group["emails"], max_items=50),
            "contacts": _unique_join(group["contacts"], max_items=50),
            "phones": _unique_join(group["phones"], max_items=50),
            "top_tags": _unique_join(group["top_tags"], max_items=20),
            "legacy_info": _unique_join(group["legacy_info"], max_items=30),
            "existing_web": _unique_join(group["existing_web"], max_items=10),
            "original_notes": _unique_join(group["original_notes"], max_items=20),
            "added_time": _unique_join(group["added_time"], max_items=10),
            "modified_time": _unique_join(group["modified_time"], max_items=10),
        }
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows)


def load_legacy_excel(
    input_path: str | Path,
    sheet_names: list[str] | None = None,
    company_column: str | None = None,
    notes_column: str | None = None,
    country_contains: str | None = DEFAULT_GERMANY_FILTER,
    combine_duplicates: bool = True,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load and normalize the uploaded Excel workbook.

    Defaults are intentionally tuned for this actual file:
    - sheet_names defaults to ["MUNDO"]
    - country_contains defaults to "Alemania" because the workbook stores Germany in Spanish
    - duplicate company rows are combined before enrichment to avoid repeated LLM calls
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input Excel file not found: {path}")

    xls = pd.ExcelFile(path)
    available_sheets = xls.sheet_names

    if sheet_names is None or not sheet_names:
        sheet_names = [DEFAULT_SHEET]

    missing = [sheet for sheet in sheet_names if sheet not in available_sheets]
    if missing:
        raise ValueError(
            f"Sheet(s) not found: {missing}. Available sheets: {available_sheets}"
        )

    standardized_frames: list[pd.DataFrame] = []

    for sheet_name in sheet_names:
        raw_df = _read_sheet(path, sheet_name)
        standardized = _standardize_sheet(
            raw_df,
            company_column=company_column,
            notes_column=notes_column,
        )
        standardized_frames.append(standardized)

    if not standardized_frames:
        return pd.DataFrame()

    df = pd.concat(standardized_frames, ignore_index=True)

    if country_contains:
        country_query = country_contains.strip().lower()
        if country_query:
            df = df[
                df["country"].fillna("").astype(str).str.lower().str.contains(
                    country_query,
                    na=False,
                    regex=False,
                )
            ].copy()

    if combine_duplicates:
        df = _combine_duplicate_companies(df)
    else:
        df = df.drop(columns=["company_key", "country_key"], errors="ignore")
        df = df.rename(
            columns={
                "source_sheet": "source_sheets",
                "source_excel_row": "source_excel_rows",
            }
        )
        df["source_row_count"] = 1

    # Keep stable, useful column order.
    ordered_columns = [
        "source_sheets",
        "source_row_count",
        "source_excel_rows",
        "company_name",
        "country",
        "emails",
        "contacts",
        "phones",
        "top_tags",
        "legacy_info",
        "existing_web",
        "original_notes",
        "added_time",
        "modified_time",
    ]
    existing_ordered = [col for col in ordered_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in existing_ordered]
    df = df[existing_ordered + remaining]

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    return df.reset_index(drop=True)
