from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

try:
    from groq import Groq
except ImportError:  # pragma: no cover - allows dashboard/tests without Groq installed
    Groq = None  # type: ignore[assignment]
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings
from src.schemas import CompanyEnrichment, ScrapedPage, SearchResult
from src.utils import truncate_text


SYSTEM_PROMPT = """
You are a precise CRM and business-data enrichment analyst for the European mobile crane industry.

You will receive one company record from a legacy Spanish/European crane contact workbook,
plus web search results and scraped page excerpts.

Your task:
1. Classify the company as Active, Acquired, Defunct, Merged, Rebranded, Unclear, or Not Relevant.
2. Extract the mobile crane capacity range/classes the company works with when evidence supports it.
3. Extract likely responsible contacts for crane sales/purchase enquiries.
4. Extract a crane/equipment color scheme when evidence supports it.

Evidence rules:
- Base your answer only on the supplied evidence.
- A URL is relevant evidence only if it directly mentions the target company, is the company's own website/domain,
  or is a reputable directory/profile specifically for that company.
- Do NOT use generic crane-market pages, manufacturer pages, Wikipedia pages about other companies, or rental portals
  as verified_url for a small target company unless that page explicitly mentions the target company.
- Prefer official websites, Impressum/Kontakt pages, parent-company pages, reputable directories, and recent evidence.
- Legacy workbook notes are useful CRM context but do not prove current activity by themselves.
- If the company appears to trade, rent, service, buy, or sell mobile cranes, crawler cranes, truck cranes,
  lifting platforms, or heavy lifting machinery, it is relevant to this CRM.
- If the company exists but appears unrelated to cranes/heavy lifting, use Not Relevant.
- If evidence is weak, old, contradictory, or mostly unrelated, use Unclear and set confidence <= 0.45.
- If there is a working official website or company-specific directory/profile confirming crane/heavy machinery activity,
  Active can be 0.70-0.95 depending on evidence quality.
- If no relevant evidence URL exists, verified_url must be an empty string.

Capacity extraction rules:
- crane_capacity_range should be concise, for example "40-700 t", "up to 1,600 t", "used AT cranes; exact capacity unknown", or "Unknown".
- Do not invent capacities from manufacturer brands alone. Use a capacity only if evidence explicitly states it or strongly lists fleet/model capacities.
- crane_capacity_details can mention models/classes such as Liebherr LTM, Grove GMK, Tadano ATF, crawler cranes, all-terrain cranes.

Responsible contact extraction rules:
- responsible_sales_contacts should include names, roles, emails, and phones likely useful for selling/buying crane equipment.
- Prefer explicit sales/contact/about/impressum evidence from the website.
- Legacy workbook contacts/emails/phones may be included, but mark contact_source as "legacy_workbook" or "both" and keep contact_confidence modest unless the website confirms responsibility.
- Do not claim a person is sales manager unless the evidence says so. Use language like "CRM contact" or "general contact" when role is uncertain.

Color extraction rules:
- crane_color_scheme should describe crane/equipment colors only when supported by text or image color hints.
- Image color hints are weak and may include sky, background, logo, or building colors. Use them cautiously.
- If only weak image hints exist, provide a tentative scheme and low color_confidence <= 0.45.
- If no support exists, use crane_color_scheme="Unknown", color_confidence=0, and explain in color_evidence_note.

Return valid JSON only.
"""


CRANE_TERMS = [
    "mobilkran", "autokran", "kranverleih", "kran", "crane", "mobile crane",
    "crawler crane", "liebherr", "grove", "demag", "tadano", "faun", "sennebogen",
    "lifting", "heavy transport", "schwertransport", "arbeitsbühne", "hebebühne",
]

CAPACITY_PATTERN = re.compile(
    r"(?:\b\d{1,4}(?:[.,]\d+)?\s*(?:-|–|—|bis|to)\s*\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)"
    r"|(?:\b(?:up to|bis zu|max\.?|maximum|tragkraft|lifting capacity|capacity)\s*\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)"
    r"|(?:\b\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)",
    flags=re.I,
)

COLOR_WORDS = [
    "blue", "red", "yellow", "orange", "white", "black", "green", "gray", "grey",
    "blau", "rot", "gelb", "orange", "weiß", "weiss", "schwarz", "grün", "gruen", "grau",
]


# ----------------------------
# Evidence diagnostics
# ----------------------------


def _company_tokens(company_name: str) -> list[str]:
    stopwords = {
        "gmbh", "mbh", "co", "kg", "ag", "ltd", "limited", "gruppe", "group",
        "und", "and", "the", "germany", "alemania", "company", "firma",
    }
    tokens = re.split(r"[^a-z0-9äöüß]+", str(company_name).lower())
    return [token for token in tokens if len(token) >= 3 and token not in stopwords]


def _domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if re.match(r"https?://", url, flags=re.I) else "https://" + url)
    return parsed.netloc.lower().removeprefix("www.")


def _allowed_urls(search_results: list[SearchResult], scraped_pages: list[ScrapedPage]) -> set[str]:
    urls = {result.url for result in search_results if result.url}
    urls.update(page.url for page in scraped_pages if page.url)
    return urls


def _relevant_evidence_counts(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
) -> dict[str, int]:
    company_name = str(company_record.get("company_name", ""))
    tokens = _company_tokens(company_name)
    existing_web = str(company_record.get("existing_web", "") or "")
    existing_domain = _domain(existing_web)

    direct_domain_hits = 0
    text_hits = 0
    crane_hits = 0

    for page in scraped_pages:
        page_domain = _domain(page.url)
        haystack = f"{page.title} {page.url} {page.text}".lower()
        if existing_domain and (page_domain == existing_domain or page_domain.endswith("." + existing_domain)):
            direct_domain_hits += 1
        if any(token in haystack for token in tokens):
            text_hits += 1
        if any(term in haystack for term in CRANE_TERMS):
            crane_hits += 1

    for result in search_results:
        if result.source_type in {"legacy_url", "email_domain"}:
            direct_domain_hits += 1
        haystack = f"{result.title} {result.url} {result.snippet}".lower()
        if any(token in haystack for token in tokens):
            text_hits += 1
        if any(term in haystack for term in CRANE_TERMS):
            crane_hits += 1

    return {
        "direct_domain_hits": direct_domain_hits,
        "company_text_hits": text_hits,
        "crane_context_hits": crane_hits,
    }


def _capacity_hints(search_results: list[SearchResult], scraped_pages: list[ScrapedPage]) -> list[str]:
    hints: list[str] = []
    sources: list[tuple[str, str]] = []

    for result in search_results:
        sources.append((result.url, f"{result.title}. {result.snippet}"))
    for page in scraped_pages:
        sources.append((page.url, f"{page.title}. {page.text}"))

    for url, text in sources:
        for match in CAPACITY_PATTERN.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            if len(value) < 2:
                continue
            hint = f"{value} — {url}"
            if hint not in hints:
                hints.append(hint)
            if len(hints) >= 12:
                return hints
    return hints


def _color_hints(scraped_pages: list[ScrapedPage]) -> list[str]:
    hints: list[str] = []
    for page in scraped_pages:
        for hint in page.image_color_hints:
            line = f"{hint} — page={page.url}"
            if line not in hints:
                hints.append(line)
            if len(hints) >= 8:
                return hints
    return hints


def _has_color_support(enrichment: CompanyEnrichment, scraped_pages: list[ScrapedPage]) -> bool:
    if _color_hints(scraped_pages):
        return True
    text = " ".join(f"{page.title} {page.text}" for page in scraped_pages).lower()
    return any(word in text for word in COLOR_WORDS)


# ----------------------------
# Prompt construction
# ----------------------------


def _build_user_prompt(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
) -> str:
    search_payload = [
        {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "source_type": result.source_type,
            "relevance_score": result.relevance_score,
        }
        for result in search_results
    ]

    page_payload = [
        {
            "title": page.title,
            "url": page.url,
            "text_excerpt": truncate_text(page.text, 3500),
            "image_color_hints": page.image_color_hints[:5],
        }
        for page in scraped_pages
    ]

    compact_record = {
        "company_name": company_record.get("company_name", ""),
        "country": company_record.get("country", ""),
        "emails": company_record.get("emails", ""),
        "contacts": company_record.get("contacts", ""),
        "phones": company_record.get("phones", ""),
        "top_tags": company_record.get("top_tags", ""),
        "legacy_info": company_record.get("legacy_info", ""),
        "existing_web": company_record.get("existing_web", ""),
        "original_notes": truncate_text(str(company_record.get("original_notes", "") or ""), 2500),
        "source_sheets": company_record.get("source_sheets", ""),
        "source_row_count": company_record.get("source_row_count", ""),
    }

    evidence_diagnostics = _relevant_evidence_counts(company_record, search_results, scraped_pages)
    capacity_hints = _capacity_hints(search_results, scraped_pages)
    color_hints = _color_hints(scraped_pages)

    schema_hint = {
        "ai_status": "Active | Acquired | Defunct | Merged | Rebranded | Unclear | Not Relevant",
        "status_confidence": "number between 0 and 1",
        "market_role": (
            "Manufacturer | Dealer | Rental Company | Service Provider | "
            "Parts Supplier | Parent Company | Unknown"
        ),
        "verified_url": "single best relevant evidence URL, or empty string if none",
        "summary": "2-4 sentence CRM-friendly summary",
        "evidence_urls": ["list of relevant evidence URLs only"],
        "reasoning_note": "brief explanation, no hidden chain-of-thought",
        "company_website_url": "best website to open from CRM table; prefer official/verified URL",
        "crane_capacity_range": "concise capacity range/classes, or Unknown",
        "crane_capacity_details": "short evidence-based details about capacity, fleet, models, or crane classes",
        "responsible_sales_contacts": "names/roles/emails/phones likely useful for crane buying/selling enquiries",
        "contact_confidence": "number between 0 and 1",
        "contact_source": "website | legacy_workbook | both | none",
        "crane_color_scheme": "e.g. boom - blue; main body - red, or Unknown",
        "color_confidence": "number between 0 and 1",
        "color_evidence_note": "brief explanation of color evidence quality",
    }

    return f"""
Legacy company record:
{json.dumps(compact_record, ensure_ascii=False, indent=2)}

Evidence diagnostics computed before the LLM:
{json.dumps(evidence_diagnostics, ensure_ascii=False, indent=2)}

Capacity hints found by regex before the LLM:
{json.dumps(capacity_hints, ensure_ascii=False, indent=2)}

Image/color hints found before the LLM:
{json.dumps(color_hints, ensure_ascii=False, indent=2)}

Search results and direct candidates:
{json.dumps(search_payload, ensure_ascii=False, indent=2)}

Scraped evidence pages:
{json.dumps(page_payload, ensure_ascii=False, indent=2)}

Return JSON using this exact shape:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""


# ----------------------------
# Fallbacks / post-processing
# ----------------------------


def _legacy_contact_string(company_record: dict[str, Any]) -> str:
    pieces: list[str] = []
    contacts = str(company_record.get("contacts", "") or "").strip()
    emails = str(company_record.get("emails", "") or "").strip()
    phones = str(company_record.get("phones", "") or "").strip()
    if contacts:
        pieces.append(f"CRM contact(s): {contacts}")
    if emails:
        pieces.append(f"Email(s): {emails}")
    if phones:
        pieces.append(f"Phone(s): {phones}")
    return "; ".join(pieces)


def _direct_or_verified_url(search_results: list[SearchResult], preferred: str = "") -> str:
    if preferred:
        return preferred
    for result in search_results:
        if result.source_type in {"legacy_url", "email_domain"} and result.url:
            return result.url
    for result in search_results:
        if result.url:
            return result.url
    return ""


def _fallback_enrichment(
    search_results: list[SearchResult],
    message: str,
    company_record: dict[str, Any] | None = None,
) -> CompanyEnrichment:
    company_record = company_record or {}
    evidence_urls = [
        result.url for result in search_results
        if result.source_type in {"legacy_url", "email_domain"} and result.url
    ][:3]
    legacy_contacts = _legacy_contact_string(company_record)
    return CompanyEnrichment(
        ai_status="Unclear",
        status_confidence=0.0,
        market_role="Unknown",
        verified_url=evidence_urls[0] if evidence_urls else "",
        summary="The automated enrichment could not produce a reliable classification.",
        evidence_urls=evidence_urls,
        reasoning_note=message,
        company_website_url=evidence_urls[0] if evidence_urls else "",
        crane_capacity_range="Unknown",
        crane_capacity_details="",
        responsible_sales_contacts=legacy_contacts,
        contact_confidence=0.35 if legacy_contacts else 0.0,
        contact_source="legacy_workbook" if legacy_contacts else "none",
        crane_color_scheme="Unknown",
        color_confidence=0.0,
        color_evidence_note="No reliable color evidence was available.",
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
)
def _call_groq_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = completion.choices[0].message.content
    if not content:
        raise ValueError("Groq returned an empty response.")

    return json.loads(content)


def _sanitize_raw_response(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    string_keys = [
        "verified_url",
        "summary",
        "reasoning_note",
        "company_website_url",
        "crane_capacity_range",
        "crane_capacity_details",
        "responsible_sales_contacts",
        "contact_source",
        "crane_color_scheme",
        "color_evidence_note",
    ]
    for key in string_keys:
        if raw.get(key) is None:
            raw[key] = ""
    if raw.get("evidence_urls") is None:
        raw["evidence_urls"] = []
    return raw


def _postprocess_enrichment(
    enrichment: CompanyEnrichment,
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
) -> CompanyEnrichment:
    allowed = _allowed_urls(search_results, scraped_pages)
    relevant_counts = _relevant_evidence_counts(company_record, search_results, scraped_pages)

    # Remove hallucinated or unrelated evidence URLs.
    enrichment.evidence_urls = [url for url in enrichment.evidence_urls if url in allowed]

    if enrichment.verified_url and enrichment.verified_url not in allowed:
        enrichment.verified_url = ""

    if not enrichment.verified_url and enrichment.evidence_urls:
        enrichment.verified_url = enrichment.evidence_urls[0]

    # The website link in the CRM can be broader than verified evidence, but must still come from candidates.
    if enrichment.company_website_url and enrichment.company_website_url not in allowed:
        enrichment.company_website_url = ""
    if not enrichment.company_website_url:
        enrichment.company_website_url = _direct_or_verified_url(search_results, enrichment.verified_url)

    # If there are no direct/company hits, do not allow a high-confidence status.
    if relevant_counts["direct_domain_hits"] == 0 and relevant_counts["company_text_hits"] == 0:
        enrichment.ai_status = "Unclear"
        enrichment.market_role = "Unknown"
        enrichment.verified_url = ""
        enrichment.evidence_urls = []
        enrichment.status_confidence = min(float(enrichment.status_confidence), 0.35)
        if "No relevant company-specific evidence" not in enrichment.reasoning_note:
            enrichment.reasoning_note = (
                "No relevant company-specific evidence was found. " + enrichment.reasoning_note
            ).strip()

    # If the model says Unclear but has an official/company-domain page with crane context,
    # let confidence be moderate rather than zero, but do not override status here.
    if enrichment.ai_status == "Unclear" and relevant_counts["direct_domain_hits"] > 0:
        enrichment.status_confidence = max(float(enrichment.status_confidence), 0.35)

    if not enrichment.crane_capacity_range.strip():
        enrichment.crane_capacity_range = "Unknown"

    # Preserve useful legacy contacts even if the LLM found no explicit sales contact online.
    legacy_contacts = _legacy_contact_string(company_record)
    if not enrichment.responsible_sales_contacts.strip() and legacy_contacts:
        enrichment.responsible_sales_contacts = legacy_contacts
        enrichment.contact_confidence = max(float(enrichment.contact_confidence), 0.35)
        enrichment.contact_source = "legacy_workbook"
    elif enrichment.responsible_sales_contacts and not enrichment.contact_source:
        enrichment.contact_source = "website" if enrichment.contact_confidence >= 0.55 else "legacy_workbook"

    # Color extraction is inherently noisy. Clamp confidence unless supported by text/image hints.
    if not enrichment.crane_color_scheme.strip():
        enrichment.crane_color_scheme = "Unknown"
    if enrichment.crane_color_scheme.lower() != "unknown" and not _has_color_support(enrichment, scraped_pages):
        enrichment.color_confidence = min(float(enrichment.color_confidence), 0.35)
        if not enrichment.color_evidence_note:
            enrichment.color_evidence_note = "Color scheme was not strongly supported by scraped text or image hints."
    if enrichment.crane_color_scheme.lower() == "unknown":
        enrichment.color_confidence = min(float(enrichment.color_confidence), 0.05)
        if not enrichment.color_evidence_note:
            enrichment.color_evidence_note = "No reliable color evidence was available."

    return enrichment


# ----------------------------
# Public API
# ----------------------------


def enrich_company_with_llm(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> CompanyEnrichment:
    if not search_results and not scraped_pages:
        return _fallback_enrichment(
            search_results=[],
            message="No web evidence was found.",
            company_record=company_record,
        )

    if Groq is None:
        return _fallback_enrichment(
            search_results=search_results,
            message="The groq package is not installed. Run pip install groq or pip install -r requirements.txt.",
            company_record=company_record,
        )

    client = Groq(api_key=settings.groq_api_key)
    user_prompt = _build_user_prompt(
        company_record=company_record,
        search_results=search_results,
        scraped_pages=scraped_pages,
    )

    try:
        raw = _call_groq_json(
            client=client,
            model=settings.groq_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        raw = _sanitize_raw_response(raw)
        enrichment = CompanyEnrichment.model_validate(raw)

        if not enrichment.evidence_urls:
            enrichment.evidence_urls = [page.url for page in scraped_pages] or [
                result.url for result in search_results
                if result.source_type in {"legacy_url", "email_domain"}
            ][:3]

        enrichment = _postprocess_enrichment(
            enrichment=enrichment,
            company_record=company_record,
            search_results=search_results,
            scraped_pages=scraped_pages,
        )

        return enrichment

    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return _fallback_enrichment(
            search_results=search_results,
            message=f"LLM response could not be validated: {exc}",
            company_record=company_record,
        )
