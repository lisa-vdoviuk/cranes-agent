from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


DEFAULT_DATA_PATH = Path("data/output/enriched_companies.csv")

STATUS_OPTIONS = [
    "Active",
    "Acquired",
    "Defunct",
    "Merged",
    "Rebranded",
    "Unclear",
    "Not Relevant",
]

MARKET_ROLE_OPTIONS = [
    "Manufacturer",
    "Dealer",
    "Rental Company",
    "Service Provider",
    "Parts Supplier",
    "Parent Company",
    "Unknown",
]

CONTACT_SOURCE_OPTIONS = ["website", "legacy_workbook", "both", "none", ""]

CORE_COLUMNS = [
    "_row_id",
    "company_name",
    "company_website_url",
    "country",
    "emails",
    "contacts",
    "phones",
    "ai_status",
    "status_confidence",
    "market_role",
    "crane_capacity_range",
    "crane_capacity_details",
    "responsible_sales_contacts",
    "contact_confidence",
    "contact_source",
    "crane_color_scheme",
    "color_confidence",
    "color_evidence_note",
    "verified_url",
    "summary",
    "evidence_urls",
    "reasoning_note",
    "crm_priority",
    "crm_next_action",
    "crm_owner_notes",
    "source_row_count",
    "source_excel_rows",
    "last_checked",
    "llm_model",
]


st.set_page_config(
    page_title="German Mobile Crane CRM",
    page_icon="🏗️",
    layout="wide",
)


# ----------------------------
# Data helpers
# ----------------------------


def normalize_url(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return ""

    # Existing workbook values may contain several URLs separated by pipes/spaces.
    first = re.split(r"\s*\|\s*|\s+", text)[0].strip().strip(";,)]")
    if not first:
        return ""
    if not re.match(r"https?://", first, flags=re.I):
        first = "https://" + first
    return first


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make the dashboard resilient if some enrichment columns are missing."""
    df = df.copy()

    defaults = {
        "source_sheets": "",
        "source_row_count": 0,
        "source_excel_rows": "",
        "company_name": "",
        "country": "",
        "emails": "",
        "contacts": "",
        "phones": "",
        "top_tags": "",
        "legacy_info": "",
        "existing_web": "",
        "original_notes": "",
        "ai_status": "Unclear",
        "status_confidence": 0.0,
        "market_role": "Unknown",
        "verified_url": "",
        "company_website_url": "",
        "summary": "",
        "evidence_urls": "",
        "reasoning_note": "",
        "crane_capacity_range": "Unknown",
        "crane_capacity_details": "",
        "responsible_sales_contacts": "",
        "contact_confidence": 0.0,
        "contact_source": "none",
        "crane_color_scheme": "Unknown",
        "color_confidence": 0.0,
        "color_evidence_note": "",
        "crm_priority": "",
        "crm_next_action": "",
        "crm_owner_notes": "",
        "last_checked": "",
        "llm_model": "",
    }

    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value

    if "_row_id" not in df.columns:
        df.insert(0, "_row_id", range(1, len(df) + 1))

    number_columns = [
        "status_confidence",
        "contact_confidence",
        "color_confidence",
        "source_row_count",
    ]
    for column in number_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    text_columns = [col for col in df.columns if col not in number_columns]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    # Build a reliable website URL for clickable company names.
    company_urls: list[str] = []
    for _, row in df.iterrows():
        company_urls.append(
            normalize_url(row.get("company_website_url"))
            or normalize_url(row.get("verified_url"))
            or normalize_url(row.get("existing_web"))
        )
    df["company_website_url"] = company_urls

    # Keep selectbox values valid.
    df.loc[~df["ai_status"].isin(STATUS_OPTIONS), "ai_status"] = "Unclear"
    df.loc[~df["market_role"].isin(MARKET_ROLE_OPTIONS), "market_role"] = "Unknown"

    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        return ensure_columns(pd.DataFrame())

    return ensure_columns(pd.read_csv(csv_path))


def save_csv(df: pd.DataFrame, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_to_save = df.copy()

    if "_row_id" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["_row_id"])

    df_to_save.to_csv(output_path, index=False)


def split_evidence_urls(value: str) -> list[str]:
    if not value:
        return []

    separators = [" | ", "\n", ","]
    urls = [value]

    for sep in separators:
        if sep in value:
            urls = value.split(sep)
            break

    return [normalize_url(url) for url in urls if normalize_url(url)]


def filter_dataframe(
    df: pd.DataFrame,
    selected_statuses: Iterable[str],
    selected_roles: Iterable[str],
    selected_contact_sources: Iterable[str],
    min_confidence: float,
    min_contact_confidence: float,
    search_text: str,
    only_with_website: bool,
    only_with_capacity: bool,
    only_with_contacts: bool,
) -> pd.DataFrame:
    filtered = df.copy()

    if selected_statuses:
        filtered = filtered[filtered["ai_status"].isin(selected_statuses)]

    if selected_roles:
        filtered = filtered[filtered["market_role"].isin(selected_roles)]

    if selected_contact_sources:
        filtered = filtered[filtered["contact_source"].isin(selected_contact_sources)]

    filtered = filtered[filtered["status_confidence"] >= min_confidence]
    filtered = filtered[filtered["contact_confidence"] >= min_contact_confidence]

    if only_with_website:
        filtered = filtered[filtered["company_website_url"].fillna("").astype(str).str.strip() != ""]

    if only_with_capacity:
        filtered = filtered[
            filtered["crane_capacity_range"].fillna("").astype(str).str.lower().str.strip().ne("unknown")
            & filtered["crane_capacity_range"].fillna("").astype(str).str.strip().ne("")
        ]

    if only_with_contacts:
        filtered = filtered[filtered["responsible_sales_contacts"].fillna("").astype(str).str.strip() != ""]

    if search_text.strip():
        query = search_text.strip().lower()

        searchable_columns = [
            "company_name",
            "country",
            "emails",
            "contacts",
            "phones",
            "original_notes",
            "legacy_info",
            "summary",
            "reasoning_note",
            "verified_url",
            "company_website_url",
            "evidence_urls",
            "crane_capacity_range",
            "crane_capacity_details",
            "responsible_sales_contacts",
            "crane_color_scheme",
            "color_evidence_note",
        ]

        mask = pd.Series(False, index=filtered.index)

        for column in searchable_columns:
            if column in filtered.columns:
                mask = mask | filtered[column].fillna("").astype(str).str.lower().str.contains(
                    query,
                    na=False,
                    regex=False,
                )

        filtered = filtered[mask]

    return filtered


def merge_edits_back(master_df: pd.DataFrame, edited_df: pd.DataFrame) -> pd.DataFrame:
    """Merge edits from the filtered data editor back into the full dataframe."""
    master_df = master_df.copy()
    edited_df = edited_df.copy()

    if "_row_id" not in master_df.columns or "_row_id" not in edited_df.columns:
        st.error("Cannot save edits because row IDs are missing.")
        return master_df

    editable_columns = [
        "company_name",
        "company_website_url",
        "emails",
        "contacts",
        "phones",
        "original_notes",
        "ai_status",
        "status_confidence",
        "market_role",
        "crane_capacity_range",
        "crane_capacity_details",
        "responsible_sales_contacts",
        "contact_confidence",
        "contact_source",
        "crane_color_scheme",
        "color_confidence",
        "color_evidence_note",
        "verified_url",
        "summary",
        "reasoning_note",
        "crm_priority",
        "crm_next_action",
        "crm_owner_notes",
    ]

    master_df = master_df.set_index("_row_id", drop=False)
    edited_df = edited_df.set_index("_row_id", drop=False)

    for row_id in edited_df.index:
        if row_id not in master_df.index:
            continue

        for column in editable_columns:
            if column in edited_df.columns and column in master_df.columns:
                master_df.at[row_id, column] = edited_df.at[row_id, column]

    return ensure_columns(master_df.reset_index(drop=True))


# ----------------------------
# Rendering helpers
# ----------------------------


COLOR_DEFINITIONS = [
    ("yellow", "#FACC15", ["yellow", "gelb", "gelbe", "gelber", "gelben", "gelbes", "gelbem", "gelb/schwarz", "gold", "golden"]),
    ("red", "#EF4444", ["red", "rot", "rote", "roter", "roten", "rotes", "rotem"]),
    ("blue", "#3B82F6", ["blue", "blau", "blaue", "blauer", "blauen", "blaues", "blauem"]),
    ("green", "#22C55E", ["green", "grün", "gruen", "grüne", "gruene", "grüner", "gruener", "grünen", "gruenen", "grünem", "gruenem"]),
    ("orange", "#F97316", ["orange", "orangen", "orangefarben"]),
    ("white", "#F8FAFC", ["white", "weiß", "weiss", "weiße", "weisse", "weißer", "weisser", "weißen", "weissen", "weißem", "weissem"]),
    ("black", "#111827", ["black", "schwarz", "schwarze", "schwarzer", "schwarzen", "schwarzes", "schwarzem"]),
    ("grey", "#9CA3AF", ["grey", "gray", "grau", "graue", "grauer", "grauen", "graues", "grauem"]),
    ("silver", "#CBD5E1", ["silver", "silber", "silbern", "silberfarben"]),
    ("purple", "#A855F7", ["purple", "violet", "violett", "lila"]),
    ("brown", "#92400E", ["brown", "braun", "braune", "brauner", "braunen"]),
]

COLOR_NAME_TO_HEX = {name: hex_value for name, hex_value, _ in COLOR_DEFINITIONS}
COLOR_ALIAS_TO_NAME = {
    alias.lower(): name
    for name, _, aliases in COLOR_DEFINITIONS
    for alias in aliases
}

STATUS_BADGE_CLASS = {
    "Active": "status-active",
    "Acquired": "status-acquired",
    "Defunct": "status-defunct",
    "Merged": "status-merged",
    "Rebranded": "status-rebranded",
    "Unclear": "status-unclear",
    "Not Relevant": "status-muted",
}


OTHER_COLOR_WORDS = {"or", "and", "und", "oder", "with", "mit", "main", "body", "boom", "jib", "ausleger"}


def _safe_cell(value: object, max_len: int = 220) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return html.escape(text)


def _plain_text(value: object, max_len: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_len and len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return text


def _company_anchor(company_name: str, url: str) -> str:
    safe_name = _safe_cell(company_name, max_len=90)
    safe_url = html.escape(normalize_url(url), quote=True)
    if not safe_url:
        return safe_name
    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_name}</a>'


def _status_badge(status: object) -> str:
    text = _plain_text(status) or "Unclear"
    css_class = STATUS_BADGE_CLASS.get(text, "status-unclear")
    return f'<span class="status-badge {css_class}">{html.escape(text)}</span>'


def _confidence_pill(value: object, label: str = "") -> str:
    try:
        confidence = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        confidence = 0.0

    pct = f"{confidence:.0%}"
    if confidence >= 0.7:
        css_class = "conf-high"
    elif confidence >= 0.35:
        css_class = "conf-mid"
    else:
        css_class = "conf-low"

    text = f"{label} {pct}".strip()
    return f'<span class="confidence-pill {css_class}">{html.escape(text)}</span>'


def _color_alias_regex() -> str:
    aliases = sorted(COLOR_ALIAS_TO_NAME.keys(), key=len, reverse=True)
    return r"\b(" + "|".join(re.escape(alias) for alias in aliases) + r")\b"


def _find_color_names(text: str) -> list[str]:
    if not text:
        return []

    found: list[str] = []
    for match in re.finditer(_color_alias_regex(), text.lower(), flags=re.I):
        color = COLOR_ALIAS_TO_NAME.get(match.group(1).lower())
        if color and color not in found:
            found.append(color)
    return found


def _clean_color_part(part: str) -> str:
    part = re.sub(r"\s+", " ", part or "").strip(" .,:;|-–—")
    part = re.sub(r"^(typical|fleet|brand|livery|colors?|colou?rs?)\s+", "", part, flags=re.I)
    part = part.strip(" .,:;|-–—")

    replacements = {
        "ausleger": "boom",
        "oberwagen": "upper",
        "unterwagen": "chassis",
        "gegengewicht": "counterweight",
        "kabine": "cab",
        "hauptfarbe": "main",
        "akzente": "accents",
        "körper": "body",
    }

    lowered = part.lower()
    for source, target in replacements.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)

    lowered = re.sub(r"\s*/\s*", "/", lowered)
    lowered = re.sub(r"\s*&\s*", "/", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()

    if not lowered or len(lowered) > 42:
        return "crane"
    return lowered


def _split_possible_color_pairs(text: str) -> list[tuple[str, str]]:
    """Return (part, color-text) pairs from semi-structured strings."""
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = normalized.replace("—", "-").replace("–", "-")

    # Examples handled:
    # boom/main - yellow; accents/counterweight - black or grey
    # Ausleger: gelb, main body = red
    # typical Liebherr livery: boom/main - yellow
    segments = re.split(
        r";|\n|,(?=\s*[A-Za-zÄÖÜäöüß0-9 /&+_.()]{2,42}\s*(?:-|:|=))",
        normalized,
    )

    pair_regex = re.compile(
        r"(?P<part>[A-Za-zÄÖÜäöüß0-9 /&+_.()]{2,60}?)"
        r"\s*(?:-|:|=)\s*"
        r"(?P<colors>[A-Za-zÄÖÜäöüß, /+&]+)$",
        flags=re.I,
    )

    pairs: list[tuple[str, str]] = []
    for segment in segments:
        segment = segment.strip(" .,")
        if not segment:
            continue

        # Remove non-color preambles such as "typical Liebherr LTM livery:".
        if ":" in segment and re.search(r":[^:]*[-=]", segment):
            segment = segment.rsplit(":", 1)[-1].strip()

        match = pair_regex.search(segment)
        if not match:
            continue

        raw_part = match.group("part")
        raw_colors = match.group("colors")
        if not _find_color_names(raw_colors):
            continue

        pairs.append((_clean_color_part(raw_part), raw_colors))

    return pairs


def parse_color_items(color_scheme: object, evidence_note: object = "") -> list[dict[str, str]]:
    """Convert a free-text crane color field into displayable swatches.

    The enrichment pipeline intentionally keeps color text conservative, for example:
    "typical Liebherr LTM livery: boom/main - yellow; accents/counterweight - black or grey".
    This parser turns that into small, safe HTML color chips without needing another LLM call.
    """
    scheme = _plain_text(color_scheme)
    note = _plain_text(evidence_note)
    combined = f"{scheme}; {note}".strip(" ;")

    if not combined or combined.lower() in {"unknown", "none", "nan", "null", "n/a"}:
        return []

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # First prefer structured component-color pairs.
    for part, color_text in _split_possible_color_pairs(combined):
        for color_name in _find_color_names(color_text):
            key = (part, color_name)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "part": part,
                    "color": color_name,
                    "hex": COLOR_NAME_TO_HEX[color_name],
                    "source": f"{part}: {color_text}",
                }
            )

    if items:
        return items[:8]

    # Fallback: show any colors found in the text as fleet/brand color chips.
    for color_name in _find_color_names(combined):
        key = ("fleet", color_name)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "part": "fleet",
                "color": color_name,
                "hex": COLOR_NAME_TO_HEX[color_name],
                "source": combined,
            }
        )

    return items[:8]


def render_color_chips_html(
    color_scheme: object,
    color_confidence: object = 0.0,
    color_evidence_note: object = "",
    *,
    compact: bool = True,
) -> str:
    items = parse_color_items(color_scheme, color_evidence_note)
    raw_scheme = _plain_text(color_scheme, max_len=360)
    raw_note = _plain_text(color_evidence_note, max_len=360)

    if not items:
        return (
            '<div class="color-cell color-empty">'
            '<span class="unknown-dot"></span>'
            '<span>Unknown</span>'
            f'{_confidence_pill(color_confidence, "")}'
            '</div>'
        )

    chips: list[str] = []
    for item in items:
        part = html.escape(item["part"])
        color = html.escape(item["color"])
        hex_value = html.escape(item["hex"], quote=True)
        title = html.escape(f"{item['part']} - {item['color']} | {raw_scheme}", quote=True)

        # White and very light colors need a visible border.
        extra_style = "border:1px solid rgba(15,23,42,0.35);" if item["color"] in {"white", "silver", "yellow"} else ""

        label = part if compact else f"{part} · {color}"
        chips.append(
            '<span class="color-chip" title="{title}">'
            '<span class="color-cube" style="background:{hex_value};{extra_style}"></span>'
            '<span class="color-chip-label">{label}</span>'
            '</span>'.format(
                title=title,
                hex_value=hex_value,
                extra_style=extra_style,
                label=html.escape(label),
            )
        )

    note_html = ""
    if raw_note and not compact:
        note_html = f'<div class="color-note">{html.escape(raw_note)}</div>'

    return (
        '<div class="color-cell">'
        '<div class="color-chip-row">'
        + "".join(chips)
        + '</div>'
        + f'<div class="color-meta">{_confidence_pill(color_confidence, "color")} '
        + f'<span class="color-raw" title="{html.escape(raw_scheme, quote=True)}">source</span></div>'
        + note_html
        + '</div>'
    )


def _dashboard_css() -> str:
    return """
    <style>
      .crm-table-wrap { overflow-x: auto; width: 100%; border-radius: 16px; border: 1px solid rgba(148,163,184,0.22); }
      table.crm-table { border-collapse: separate; border-spacing: 0; width: 100%; font-size: 0.88rem; }
      table.crm-table th {
        text-align: left;
        padding: 0.68rem 0.62rem;
        border-bottom: 1px solid rgba(148,163,184,0.26);
        position: sticky;
        top: 0;
        background: rgba(15,23,42,0.92);
        color: white;
        z-index: 1;
        white-space: nowrap;
      }
      table.crm-table td { vertical-align: top; padding: 0.68rem 0.62rem; border-bottom: 1px solid rgba(148,163,184,0.18); }
      table.crm-table tr:hover td { background: rgba(148,163,184,0.08); }
      table.crm-table a { text-decoration: none; font-weight: 750; }
      table.crm-table small { opacity: 0.72; }
      .status-badge, .confidence-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.16rem 0.48rem;
        font-size: 0.74rem;
        font-weight: 750;
        white-space: nowrap;
      }
      .status-active { background: rgba(34,197,94,0.16); color: #16A34A; border: 1px solid rgba(34,197,94,0.25); }
      .status-acquired, .status-merged, .status-rebranded { background: rgba(59,130,246,0.16); color: #2563EB; border: 1px solid rgba(59,130,246,0.24); }
      .status-defunct { background: rgba(239,68,68,0.14); color: #DC2626; border: 1px solid rgba(239,68,68,0.22); }
      .status-unclear { background: rgba(245,158,11,0.16); color: #D97706; border: 1px solid rgba(245,158,11,0.24); }
      .status-muted { background: rgba(100,116,139,0.16); color: #64748B; border: 1px solid rgba(100,116,139,0.22); }
      .conf-high { background: rgba(34,197,94,0.14); color: #16A34A; }
      .conf-mid { background: rgba(245,158,11,0.14); color: #D97706; }
      .conf-low { background: rgba(100,116,139,0.14); color: #64748B; }
      .color-cell { min-width: 160px; }
      .color-chip-row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.36rem; margin-bottom: 0.28rem; }
      .color-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        border: 1px solid rgba(148,163,184,0.24);
        background: rgba(148,163,184,0.08);
        border-radius: 999px;
        padding: 0.18rem 0.46rem 0.18rem 0.22rem;
        line-height: 1;
        box-shadow: 0 1px 2px rgba(15,23,42,0.08);
      }
      .color-cube {
        width: 1.05rem;
        height: 1.05rem;
        border-radius: 0.28rem;
        display: inline-block;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.24), 0 1px 2px rgba(15,23,42,0.18);
      }
      .color-chip-label { font-size: 0.72rem; font-weight: 700; color: inherit; opacity: 0.86; }
      .color-meta { display: flex; align-items: center; gap: 0.35rem; flex-wrap: wrap; }
      .color-raw { font-size: 0.72rem; opacity: 0.55; cursor: help; border-bottom: 1px dotted rgba(148,163,184,0.7); }
      .color-note { margin-top: 0.45rem; font-size: 0.82rem; opacity: 0.75; }
      .color-empty { display: flex; align-items: center; flex-wrap: wrap; gap: 0.4rem; color: #64748B; }
      .unknown-dot { width: 0.72rem; height: 0.72rem; border-radius: 0.22rem; display:inline-block; background: repeating-linear-gradient(45deg, #CBD5E1, #CBD5E1 3px, #F8FAFC 3px, #F8FAFC 6px); border: 1px solid rgba(100,116,139,0.35); }
      .detail-card {
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 16px;
        padding: 0.85rem;
        background: rgba(148,163,184,0.06);
      }
    </style>
    """


def render_clickable_crm_table(df: pd.DataFrame, max_rows: int = 200) -> None:
    st.subheader("CRM table")
    st.caption("Company names are clickable when a website URL is available. Crane colors are shown as swatches; hover over them to see the raw source text.")

    if df.empty:
        st.info("No rows match the current filters.")
        return

    rows = []
    for _, row in df.head(max_rows).iterrows():
        confidence = _confidence_pill(row.get("status_confidence", 0.0))
        contact_conf = _confidence_pill(row.get("contact_confidence", 0.0), "contact")
        color_html = render_color_chips_html(
            row.get("crane_color_scheme", ""),
            row.get("color_confidence", 0.0),
            row.get("color_evidence_note", ""),
            compact=True,
        )
        rows.append(
            "<tr>"
            f"<td>{_company_anchor(row.get('company_name', ''), row.get('company_website_url', ''))}</td>"
            f"<td>{_status_badge(row.get('ai_status', ''))}</td>"
            f"<td>{confidence}</td>"
            f"<td>{_safe_cell(row.get('market_role', ''))}</td>"
            f"<td>{_safe_cell(row.get('crane_capacity_range', ''), 120)}</td>"
            f"<td>{_safe_cell(row.get('responsible_sales_contacts', ''), 260)}<br>{contact_conf}</td>"
            f"<td>{color_html}</td>"
            f"<td>{_safe_cell(row.get('summary', ''), 320)}</td>"
            "</tr>"
        )

    table_html = f"""
    {_dashboard_css()}
    <div class="crm-table-wrap">
      <table class="crm-table">
        <thead>
          <tr>
            <th>Company</th>
            <th>Status</th>
            <th>Conf.</th>
            <th>Role</th>
            <th>Crane capacity</th>
            <th>Responsible contacts</th>
            <th>Crane colors</th>
            <th>Summary</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    if len(df) > max_rows:
        st.info(f"Showing first {max_rows} of {len(df)} matching rows. Increase the table row limit in the sidebar if needed.")

def render_metrics(df: pd.DataFrame) -> None:
    total = len(df)
    active = int((df["ai_status"] == "Active").sum()) if not df.empty else 0
    acquired = int((df["ai_status"] == "Acquired").sum()) if not df.empty else 0
    unclear = int((df["ai_status"] == "Unclear").sum()) if not df.empty else 0
    with_capacity = int(
        (
            df["crane_capacity_range"].fillna("").astype(str).str.lower().str.strip().ne("unknown")
            & df["crane_capacity_range"].fillna("").astype(str).str.strip().ne("")
        ).sum()
    ) if not df.empty else 0
    with_contacts = int((df["responsible_sales_contacts"].fillna("").astype(str).str.strip() != "").sum()) if not df.empty else 0
    avg_confidence = float(df["status_confidence"].mean()) if not df.empty else 0.0

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Companies", total)
    col2.metric("Active", active)
    col3.metric("Acquired", acquired)
    col4.metric("Unclear", unclear)
    col5.metric("Capacity found", with_capacity)
    col6.metric("Contacts found", with_contacts, help=f"Avg status confidence: {avg_confidence:.0%}")


def render_detail_panel(df: pd.DataFrame) -> None:
    st.subheader("Company detail")

    if df.empty:
        st.info("No company selected.")
        return

    company_names = df["company_name"].fillna("").astype(str).tolist()
    selected_company = st.selectbox(
        "Select company",
        company_names,
        index=0,
    )

    row = df[df["company_name"] == selected_company].iloc[0]

    left, right = st.columns([1, 1])

    with left:
        website_url = normalize_url(row.get("company_website_url", ""))
        if website_url:
            st.markdown(f"### [{row.get('company_name', '')}]({website_url})")
        else:
            st.markdown(f"### {row.get('company_name', '')}")

        st.write(f"**Status:** {row.get('ai_status', '')}")
        st.write(f"**Confidence:** {float(row.get('status_confidence', 0.0)):.0%}")
        st.write(f"**Market role:** {row.get('market_role', '')}")
        st.write(f"**Country:** {row.get('country', '')}")
        st.write(f"**Last checked:** {row.get('last_checked', '')}")

        verified_url = normalize_url(row.get("verified_url", ""))
        if verified_url:
            st.markdown(f"**Verified URL:** [{verified_url}]({verified_url})")

    with right:
        st.write("**AI summary**")
        st.write(row.get("summary", ""))

        st.write("**Reasoning note**")
        st.write(row.get("reasoning_note", ""))

    cap_col, contact_col, color_col = st.columns(3)
    with cap_col:
        st.write("**Crane capacity**")
        st.write(row.get("crane_capacity_range", ""))
        st.caption(row.get("crane_capacity_details", ""))

    with contact_col:
        st.write("**Responsible contacts**")
        st.write(row.get("responsible_sales_contacts", ""))
        st.caption(
            f"Source: {row.get('contact_source', '')} · confidence: {float(row.get('contact_confidence', 0.0)):.0%}"
        )

    with color_col:
        st.write("**Crane colors**")
        st.markdown(
            _dashboard_css()
            + '<div class="detail-card">'
            + render_color_chips_html(
                row.get("crane_color_scheme", ""),
                row.get("color_confidence", 0.0),
                row.get("color_evidence_note", ""),
                compact=False,
            )
            + '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Raw color text", expanded=False):
            st.write(row.get("crane_color_scheme", ""))
            st.caption(
                f"confidence: {float(row.get('color_confidence', 0.0)):.0%} · {row.get('color_evidence_note', '')}"
            )

    with st.expander("Legacy CRM data", expanded=False):
        st.write("**Emails:**", row.get("emails", ""))
        st.write("**Contacts:**", row.get("contacts", ""))
        st.write("**Phones:**", row.get("phones", ""))
        st.write("**Existing workbook web:**", row.get("existing_web", ""))
        st.write("**Original notes:**", row.get("original_notes", ""))
        st.write("**Legacy info:**", row.get("legacy_info", ""))
        st.write("**Source Excel rows:**", row.get("source_excel_rows", ""))

    evidence_urls = split_evidence_urls(str(row.get("evidence_urls", "") or ""))
    if evidence_urls:
        with st.expander("Evidence URLs", expanded=False):
            for idx, url in enumerate(evidence_urls, start=1):
                st.markdown(f"{idx}. [{url}]({url})")


def render_editor(filtered_df: pd.DataFrame, master_df: pd.DataFrame, data_path: str) -> None:
    st.subheader("Edit CRM data")
    st.caption("Use this editor for corrections; click **Save edits to CSV** afterward.")

    display_columns = [column for column in CORE_COLUMNS if column in filtered_df.columns]
    extra_columns = [column for column in filtered_df.columns if column not in display_columns]
    editor_df = filtered_df[display_columns + extra_columns].copy()

    edited_df = st.data_editor(
        editor_df,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "_row_id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
            "company_name": st.column_config.TextColumn("Company", width="medium"),
            "company_website_url": st.column_config.LinkColumn("Company website", width="medium"),
            "country": st.column_config.TextColumn("Country", disabled=True, width="small"),
            "emails": st.column_config.TextColumn("Emails", width="medium"),
            "contacts": st.column_config.TextColumn("Legacy contacts", width="medium"),
            "phones": st.column_config.TextColumn("Phones", width="medium"),
            "ai_status": st.column_config.SelectboxColumn("Status", options=STATUS_OPTIONS, required=True, width="medium"),
            "status_confidence": st.column_config.NumberColumn("Confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
            "market_role": st.column_config.SelectboxColumn("Market role", options=MARKET_ROLE_OPTIONS, required=True, width="medium"),
            "crane_capacity_range": st.column_config.TextColumn("Capacity range", width="medium"),
            "crane_capacity_details": st.column_config.TextColumn("Capacity details", width="large"),
            "responsible_sales_contacts": st.column_config.TextColumn("Responsible sales contacts", width="large"),
            "contact_confidence": st.column_config.NumberColumn("Contact conf.", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
            "contact_source": st.column_config.SelectboxColumn("Contact source", options=CONTACT_SOURCE_OPTIONS, width="medium"),
            "crane_color_scheme": st.column_config.TextColumn("Crane colors", width="medium"),
            "color_confidence": st.column_config.NumberColumn("Color conf.", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", width="small"),
            "color_evidence_note": st.column_config.TextColumn("Color evidence", width="large"),
            "verified_url": st.column_config.LinkColumn("Verified URL", width="medium"),
            "summary": st.column_config.TextColumn("Summary", width="large"),
            "evidence_urls": st.column_config.TextColumn("Evidence URLs", disabled=True, width="large"),
            "reasoning_note": st.column_config.TextColumn("Reasoning note", width="large"),
            "crm_priority": st.column_config.TextColumn("CRM priority", width="small"),
            "crm_next_action": st.column_config.TextColumn("Next action", width="medium"),
            "crm_owner_notes": st.column_config.TextColumn("Owner notes", width="large"),
            "source_row_count": st.column_config.NumberColumn("Source rows", disabled=True, width="small"),
            "source_excel_rows": st.column_config.TextColumn("Excel rows", disabled=True, width="medium"),
            "last_checked": st.column_config.TextColumn("Last checked", disabled=True, width="medium"),
            "llm_model": st.column_config.TextColumn("LLM model", disabled=True, width="medium"),
        },
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Save edits to CSV", type="primary"):
            updated_df = merge_edits_back(master_df, edited_df)
            st.session_state.crm_df = updated_df
            save_csv(updated_df, data_path)
            st.success(f"Saved edits to `{data_path}`.")

    with col2:
        export_df = st.session_state.crm_df.copy()
        if "_row_id" in export_df.columns:
            export_df = export_df.drop(columns=["_row_id"])

        st.download_button(
            label="Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="enriched_companies_export.csv",
            mime="text/csv",
        )

    with col3:
        st.caption("Edits are kept in this session until you click **Save edits to CSV**.")


# ----------------------------
# Main app
# ----------------------------


def main() -> None:
    st.title("🏗️ German Mobile Crane CRM")
    st.caption("Local dashboard for AI-enriched mobile crane company data.")

    with st.sidebar:
        st.header("Data source")

        data_path = st.text_input("Enriched CSV path", value=str(DEFAULT_DATA_PATH))

        if st.button("Reload CSV"):
            st.cache_data.clear()
            st.session_state.pop("crm_df", None)
            st.rerun()

        if "crm_df" not in st.session_state:
            st.session_state.crm_df = load_csv(data_path)

        df = ensure_columns(st.session_state.crm_df)

        st.divider()
        st.header("Filters")

        available_statuses = [status for status in STATUS_OPTIONS if status in set(df["ai_status"])] or STATUS_OPTIONS
        selected_statuses = st.multiselect("Status", options=STATUS_OPTIONS, default=available_statuses)

        available_roles = [role for role in MARKET_ROLE_OPTIONS if role in set(df["market_role"])] or MARKET_ROLE_OPTIONS
        selected_roles = st.multiselect("Market role", options=MARKET_ROLE_OPTIONS, default=available_roles)

        available_contact_sources = [src for src in CONTACT_SOURCE_OPTIONS if src in set(df["contact_source"])] or CONTACT_SOURCE_OPTIONS
        selected_contact_sources = st.multiselect("Contact source", options=CONTACT_SOURCE_OPTIONS, default=available_contact_sources)

        min_confidence = st.slider("Minimum status confidence", 0.0, 1.0, 0.0, 0.05, format="%.2f")
        min_contact_confidence = st.slider("Minimum contact confidence", 0.0, 1.0, 0.0, 0.05, format="%.2f")

        only_with_website = st.checkbox("Only companies with website link", value=False)
        only_with_capacity = st.checkbox("Only companies with capacity found", value=False)
        only_with_contacts = st.checkbox("Only companies with responsible contacts", value=False)

        search_text = st.text_input("Search", placeholder="Company, contact, capacity, color, summary, URL...")

        table_row_limit = st.number_input("Clickable table row limit", min_value=25, max_value=2000, value=200, step=25)

    if df.empty:
        st.warning(f"No data found. Run the enrichment pipeline first or check that this file exists: `{data_path}`")
        st.stop()

    filtered_df = filter_dataframe(
        df=df,
        selected_statuses=selected_statuses,
        selected_roles=selected_roles,
        selected_contact_sources=selected_contact_sources,
        min_confidence=min_confidence,
        min_contact_confidence=min_contact_confidence,
        search_text=search_text,
        only_with_website=only_with_website,
        only_with_capacity=only_with_capacity,
        only_with_contacts=only_with_contacts,
    )

    render_metrics(filtered_df)

    with st.expander("Status breakdown", expanded=False):
        if not filtered_df.empty:
            chart_df = filtered_df["ai_status"].fillna("Unclear").value_counts().reset_index()
            chart_df.columns = ["status", "count"]
            st.bar_chart(chart_df, x="status", y="count")
        else:
            st.info("No rows match the current filters.")

    table_tab, detail_tab, edit_tab = st.tabs(["Clickable CRM table", "Company detail", "Edit / export"])

    with table_tab:
        render_clickable_crm_table(filtered_df, max_rows=int(table_row_limit))

    with detail_tab:
        render_detail_panel(filtered_df)

    with edit_tab:
        render_editor(filtered_df, df, data_path)


if __name__ == "__main__":
    main()
