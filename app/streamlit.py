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


def _safe_cell(value: object, max_len: int = 220) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[:max_len].rsplit(" ", 1)[0] + "..."
    return html.escape(text)


def _company_anchor(company_name: str, url: str) -> str:
    safe_name = _safe_cell(company_name, max_len=90)
    safe_url = html.escape(normalize_url(url), quote=True)
    if not safe_url:
        return safe_name
    return f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{safe_name}</a>'


def render_clickable_crm_table(df: pd.DataFrame, max_rows: int = 200) -> None:
    st.subheader("CRM table")
    st.caption("Company names are clickable when a website URL is available.")

    if df.empty:
        st.info("No rows match the current filters.")
        return

    rows = []
    for _, row in df.head(max_rows).iterrows():
        confidence = f"{float(row.get('status_confidence', 0.0)):.0%}"
        contact_conf = f"{float(row.get('contact_confidence', 0.0)):.0%}"
        color_conf = f"{float(row.get('color_confidence', 0.0)):.0%}"
        rows.append(
            "<tr>"
            f"<td>{_company_anchor(row.get('company_name', ''), row.get('company_website_url', ''))}</td>"
            f"<td>{_safe_cell(row.get('ai_status', ''))}</td>"
            f"<td>{html.escape(confidence)}</td>"
            f"<td>{_safe_cell(row.get('market_role', ''))}</td>"
            f"<td>{_safe_cell(row.get('crane_capacity_range', ''), 120)}</td>"
            f"<td>{_safe_cell(row.get('responsible_sales_contacts', ''), 260)}<br><small>contact confidence: {html.escape(contact_conf)}</small></td>"
            f"<td>{_safe_cell(row.get('crane_color_scheme', ''), 160)}<br><small>color confidence: {html.escape(color_conf)}</small></td>"
            f"<td>{_safe_cell(row.get('summary', ''), 320)}</td>"
            "</tr>"
        )

    table_html = f"""
    <style>
      .crm-table-wrap {{ overflow-x: auto; width: 100%; }}
      table.crm-table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
      table.crm-table th {{ text-align: left; padding: 0.55rem; border-bottom: 1px solid #ddd; position: sticky; top: 0; background: var(--background-color, black); }}
      table.crm-table td {{ vertical-align: top; padding: 0.55rem; border-bottom: 1px solid rgba(128,128,128,0.25); }}
      table.crm-table a {{ text-decoration: none; font-weight: 650; }}
      table.crm-table small {{ opacity: 0.72; }}
    </style>
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
