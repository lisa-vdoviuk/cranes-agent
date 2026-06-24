from __future__ import annotations

"""
llm_ollama.py — Text-LLM enrichment via local Ollama.

Replaces the old llm_groq.py. All config keys are now provider-agnostic
(llm_model, llm_base_url, etc.) to match the rewritten config.py.

Key fixes applied:
  - settings.groq_model AttributeError eliminated (was crashing every save).
  - Color fields removed from the LLM prompt (handled by vision model in
    color_inference.py). The LLM no longer wastes tokens on colour reasoning.
  - enrichment_path recorded as a machine-readable field on every return branch.
  - Heuristic-bypass threshold tightened: requires BOTH strong_page_hits > 1
    AND direct_domain_hits > 0 (was > 0 for strong_page_hits, causing 85 %
    Active inflation).
  - Page cache uses JSON (scraper.py change); prompt uses readable indented JSON.
  - CRANE_TERMS and constants moved to constants.py.
"""

import json
import re
from typing import Any
from urllib.parse import urlparse

try:
    from ollama import Client as OllamaClient
except ImportError:  # pragma: no cover
    OllamaClient = None  # type: ignore[assignment]

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.color_inference import infer_crane_color_scheme
from src.config import Settings
from src.constants import CRANE_TERMS
from src.schemas import CompanyEnrichment, ScrapedPage, SearchResult
from src.site_verifier import classify_url_category, resolve_official_site
from src.utils import read_json, safe_hash, truncate_text, write_json


# ---------------------------------------------------------------------------
# System prompt (colour fields intentionally absent — handled by vision model)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a precise CRM and business-data enrichment analyst for the European mobile crane industry.
Return valid JSON only.

Task:
- Classify the target company: Active, Acquired, Defunct, Merged, Rebranded, Unclear, or Not Relevant.
- Extract market role, best relevant URL, concise summary, capacity range/details, and likely sales/purchase contacts.

Evidence rules:
- Use only the supplied evidence — do not invent facts.
- A URL is relevant only if it directly mentions the target company, is the company's own domain,
  or is a company-specific profile.
- Do not use generic crane-market pages, manufacturer pages, or pages about other companies as verified_url.
- If evidence is weak or unrelated, use Unclear and confidence <= 0.45.
- If there is company-specific crane/heavy-lifting evidence, Active confidence may reach 0.70-0.95.
- If no relevant evidence URL exists, verified_url must be an empty string.
- Do not invent crane capacities or job roles.
- Legacy workbook contacts are valid CRM contacts unless website evidence proves a different sales role.

Parked / expired / for-sale domain rules (CRITICAL):
- If the only web evidence for a URL describes the domain as being for sale, parked, expired,
  or redirects to a domain registrar or parking service (e.g. GoDaddy, Sedo, IONOS, Strato,
  united-domains, 1&1, Namecheap, dan.com, etc.), you MUST treat that URL as invalid.
- German equivalents to watch for: "Domain kaufen", "Domain zu verkaufen", "Domain erwerben",
  "Domain abgelaufen", "Hier entsteht", "Website im Aufbau", "Demnächst verfügbar",
  "Jetzt Domain registrieren", "Domain ist verfügbar", "Domain noch nicht registriert".
- Do NOT assign Active status, do NOT set company_website_url or verified_url, and do NOT
  use such a page as evidence of company activity.
- If all available URLs are parked or invalid, treat the company as having no verifiable
  web presence; use Unclear or Defunct as appropriate and set confidence <= 0.40.\
"""

CAPACITY_PATTERN = re.compile(
    r"(?:\b\d{1,4}(?:[.,]\d+)?\s*(?:-|–|—|bis|to)\s*\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)"
    r"|(?:\b(?:up to|bis zu|max\.?|maximum|tragkraft|lifting capacity|capacity)\s*\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)"
    r"|(?:\b\d{1,4}(?:[.,]\d+)?\s*(?:t|to\.?|tons?|tonnes?|tonnen)\b)",
    flags=re.I,
)


# ---------------------------------------------------------------------------
# Evidence diagnostics
# ---------------------------------------------------------------------------

def _company_tokens(company_name: str) -> list[str]:
    stopwords = {
        "gmbh", "mbh", "co", "kg", "ag", "ltd", "limited", "gruppe", "group",
        "und", "and", "the", "germany", "alemania", "company", "firma",
    }
    tokens = re.split(r"[^a-z0-9äöüß]+", str(company_name).lower())
    return [t for t in tokens if len(t) >= 3 and t not in stopwords]


def _domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if re.match(r"https?://", url, flags=re.I) else "https://" + url)
    return parsed.netloc.lower().removeprefix("www.")


def _allowed_urls(search_results: list[SearchResult], scraped_pages: list[ScrapedPage]) -> set[str]:
    urls = {r.url for r in search_results if r.url}
    urls.update(p.url for p in scraped_pages if p.url)
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
    company_text_hits = 0
    crane_context_hits = 0
    strong_page_hits = 0

    for page in scraped_pages:
        page_domain = _domain(page.url)
        haystack = f"{page.title} {page.url} {page.text}".lower()
        is_direct = bool(
            existing_domain and (
                page_domain == existing_domain
                or page_domain.endswith("." + existing_domain)
            )
        )
        has_company = any(tok in haystack for tok in tokens)
        has_crane = any(term in haystack for term in CRANE_TERMS)
        if is_direct:
            direct_domain_hits += 1
        if has_company:
            company_text_hits += 1
        if has_crane:
            crane_context_hits += 1
        if has_crane and (is_direct or has_company):
            strong_page_hits += 1

    for result in search_results:
        if result.source_type in {"legacy_url", "email_domain"}:
            direct_domain_hits += 1
        haystack = f"{result.title} {result.url} {result.snippet}".lower()
        if any(tok in haystack for tok in tokens):
            company_text_hits += 1
        if any(term in haystack for term in CRANE_TERMS):
            crane_context_hits += 1

    return {
        "direct_domain_hits": direct_domain_hits,
        "company_text_hits": company_text_hits,
        "crane_context_hits": crane_context_hits,
        "strong_page_hits": strong_page_hits,
    }


def _capacity_hints(
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    limit: int = 6,
) -> list[str]:
    hints: list[str] = []
    sources: list[tuple[str, str]] = []
    for r in search_results:
        sources.append((r.url, f"{r.title}. {r.snippet}"))
    for p in scraped_pages:
        sources.append((p.url, f"{p.title}. {p.text}"))
    for url, text in sources:
        for match in CAPACITY_PATTERN.finditer(text):
            value = re.sub(r"\s+", " ", match.group(0)).strip()
            hint = f"{value} — {url}"
            if hint not in hints:
                hints.append(hint)
            if len(hints) >= limit:
                return hints
    return hints


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
    for r in search_results:
        if r.source_type in {"legacy_url", "email_domain"} and r.url:
            return r.url
    for r in search_results:
        if r.url:
            return r.url
    return ""


def _market_role_from_text(text: str) -> str:
    lowered = text.lower()
    if any(t in lowered for t in ["manufacturer", "hersteller", "manufactures", "produces", "entwickelt und fertigt"]):
        return "Manufacturer"
    if any(t in lowered for t in ["kranverleih", "vermietung", "mieten", "rental", "rent", "hire"]):
        return "Rental Company"
    if any(t in lowered for t in ["used crane", "gebrauchtkran", "verkauf", "sales", "dealer", "handel", "trading"]):
        return "Dealer"
    if any(t in lowered for t in ["service", "maintenance", "wartung", "reparatur"]):
        return "Service Provider"
    return "Unknown"


def _combined_evidence_text(
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    max_chars: int = 6000,
) -> str:
    pieces = [f"{r.title}. {r.snippet}" for r in search_results[:8]]
    pieces.extend(f"{p.title}. {p.text}" for p in scraped_pages[:4])
    return truncate_text(" ".join(pieces), max_chars)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _compact_record(company_record: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "company_name": company_record.get("company_name", ""),
        "country": company_record.get("country", ""),
        "emails": truncate_text(str(company_record.get("emails", "") or ""), 300),
        "contacts": truncate_text(str(company_record.get("contacts", "") or ""), 300),
        "phones": truncate_text(str(company_record.get("phones", "") or ""), 240),
        "top_tags": truncate_text(str(company_record.get("top_tags", "") or ""), 180),
        "legacy_info": truncate_text(str(company_record.get("legacy_info", "") or ""), 350),
        "existing_web": company_record.get("existing_web", ""),
        "original_notes": truncate_text(
            str(company_record.get("original_notes", "") or ""),
            settings.llm_record_notes_chars,
        ),
    }


def _build_user_prompt(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> str:
    selected_results = sorted(
        search_results, key=lambda r: r.relevance_score, reverse=True
    )[: settings.llm_max_search_items]
    selected_pages = scraped_pages[: settings.llm_max_pages]

    search_payload = [
        {
            "title": r.title,
            "url": r.url,
            "snippet": truncate_text(r.snippet, 500),
            "source_type": r.source_type,
            "relevance_score": r.relevance_score,
        }
        for r in selected_results
    ]

    page_payload = [
        {
            "title": p.title,
            "url": p.url,
            "text_excerpt": truncate_text(p.text, settings.llm_page_excerpt_chars),
        }
        for p in selected_pages
    ]

    evidence_diagnostics = _relevant_evidence_counts(
        company_record, selected_results, selected_pages
    )
    capacity_hints = _capacity_hints(selected_results, selected_pages, limit=5)

    # Colour fields are intentionally absent — they are handled by the vision model.
    schema_hint = {
        "ai_status": "Active | Acquired | Defunct | Merged | Rebranded | Unclear | Not Relevant",
        "status_confidence": "number 0..1",
        "market_role": "Manufacturer | Dealer | Rental Company | Service Provider | Parts Supplier | Parent Company | Unknown",
        "verified_url": "single best relevant evidence URL, or empty string",
        "summary": "1-2 short CRM-friendly sentences",
        "evidence_urls": ["relevant evidence URLs only"],
        "reasoning_note": "brief explanation, no hidden chain-of-thought",
        "company_website_url": "best website to open from CRM table",
        "crane_capacity_range": "concise capacity/classes, or Unknown",
        "crane_capacity_details": "short evidence-based details",
        "responsible_sales_contacts": "names/roles/emails/phones useful for crane buying/selling enquiries",
        "contact_confidence": "number 0..1",
        "contact_source": "website | legacy_workbook | both | none",
    }

    return (
        "Legacy company record:\n"
        + json.dumps(_compact_record(company_record, settings), ensure_ascii=False, indent=2)
        + "\n\nEvidence diagnostics:\n"
        + json.dumps(evidence_diagnostics, ensure_ascii=False, indent=2)
        + "\n\nCapacity hints:\n"
        + json.dumps(capacity_hints, ensure_ascii=False, indent=2)
        + "\n\nSearch/direct candidates:\n"
        + json.dumps(search_payload, ensure_ascii=False, indent=2)
        + "\n\nScraped page excerpts:\n"
        + json.dumps(page_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON using this exact shape:\n"
        + json.dumps(schema_hint, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# LLM cache
# ---------------------------------------------------------------------------

def _llm_cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
    return f"v6-ollama::{model}::{safe_hash(system_prompt + user_prompt)}"


def _load_llm_cache(settings: Settings) -> dict[str, Any]:
    if not settings.llm_cache_enabled:
        return {}
    return read_json(settings.llm_cache_path, default={})


def _save_llm_cache(settings: Settings, cache: dict[str, Any]) -> None:
    if settings.llm_cache_enabled:
        write_json(settings.llm_cache_path, cache)


# ---------------------------------------------------------------------------
# Fallback / heuristic enrichment
# ---------------------------------------------------------------------------

def _fallback_enrichment(
    search_results: list[SearchResult],
    message: str,
    company_record: dict[str, Any] | None = None,
) -> CompanyEnrichment:
    company_record = company_record or {}
    evidence_urls = [
        r.url for r in search_results
        if r.source_type in {"legacy_url", "email_domain"} and r.url
    ][:3]
    legacy_contacts = _legacy_contact_string(company_record)
    return CompanyEnrichment(
        ai_status="Unclear",
        status_confidence=0.0,
        market_role="Unknown",
        verified_url="",
        summary="The automated enrichment could not produce a reliable classification.",
        evidence_urls=evidence_urls,
        reasoning_note=message,
        company_website_url="",
        official_website_confidence=0.0,
        site_status="no_official_site",
        site_rejection_reason=message,
        profile_urls=[],
        rejected_urls=[],
        official_site_debug="fallback",        crane_capacity_range="Unknown",
        crane_capacity_details="",
        responsible_sales_contacts=legacy_contacts,
        contact_confidence=0.35 if legacy_contacts else 0.0,
        contact_source="legacy_workbook" if legacy_contacts else "none",
        crane_color_scheme="Unknown",
        color_confidence=0.0,
        color_evidence_note="Color analysis not available (fallback path).",
        enrichment_path="fallback",
    )


def _heuristic_enrichment(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    reason: str,
) -> CompanyEnrichment:
    counts = _relevant_evidence_counts(company_record, search_results, scraped_pages)
    combined = _combined_evidence_text(search_results, scraped_pages)
    capacity = _capacity_hints(search_results, scraped_pages, limit=3)
    legacy_contacts = _legacy_contact_string(company_record)
    evidence_urls = [p.url for p in scraped_pages[:3] if p.url]
    if not evidence_urls:
        evidence_urls = [
            r.url for r in search_results
            if r.source_type in {"legacy_url", "email_domain"} and r.url
        ][:3]

    # Tightened threshold vs. old code: require > 1 strong_page_hits to reach Active
    # without LLM confirmation. The old "> 0" was causing ~85 % Active inflation.
    if counts["strong_page_hits"] > 1:
        status = "Active"
        confidence = 0.72
        summary = (
            "Multiple company-specific pages with crane/heavy-lifting evidence found. "
            "LLM skipped (strong deterministic signal)."
        )
    else:
        status = "Unclear"
        confidence = 0.25 if counts["direct_domain_hits"] else 0.0
        summary = (
            "Company-specific evidence was insufficient for reliable automated classification. "
            "LLM skipped."
        )

    return CompanyEnrichment(
        ai_status=status,
        status_confidence=confidence,
        market_role=_market_role_from_text(combined),
        verified_url=evidence_urls[0] if evidence_urls and status == "Active" else "",
        summary=summary,
        evidence_urls=evidence_urls if status == "Active" else [],
        reasoning_note=reason,
        company_website_url=_direct_or_verified_url(
            search_results, evidence_urls[0] if evidence_urls else ""
        ),
        official_website_confidence=0.0,
        site_status="",
        site_rejection_reason="",
        profile_urls=[],
        rejected_urls=[],
        official_site_debug="",
        crane_capacity_range=capacity[0].split(" — ", 1)[0] if capacity else "Unknown",
        crane_capacity_details=" | ".join(capacity),
        responsible_sales_contacts=legacy_contacts,
        contact_confidence=0.35 if legacy_contacts else 0.0,
        contact_source="legacy_workbook" if legacy_contacts else "none",
        crane_color_scheme="Unknown",
        color_confidence=0.0,
        color_evidence_note="Color will be analysed by vision model post-classification.",
        enrichment_path="heuristic",
    )


def _should_skip_llm_low_evidence(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> bool:
    if not settings.llm_skip_low_evidence:
        return False
    counts = _relevant_evidence_counts(company_record, search_results, scraped_pages)
    if counts["direct_domain_hits"] == 0 and counts["company_text_hits"] == 0:
        return True
    if not scraped_pages and counts["crane_context_hits"] == 0:
        return True
    return False


def _should_use_confident_heuristic(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> bool:
    if not settings.llm_skip_confident_heuristics:
        return False
    counts = _relevant_evidence_counts(company_record, search_results, scraped_pages)
    # Tightened: require BOTH > 1 strong page hits AND a direct domain hit.
    return counts["strong_page_hits"] > 1 and counts["direct_domain_hits"] > 0


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=3, max=20),
)
def _call_ollama_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    completion = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        stream=False,
        options={"num_predict": max_tokens},
    )
    content = completion.get("message", {}).get("content", "")
    if not content:
        raise ValueError("Ollama returned an empty response.")
    return json.loads(content)


def _sanitize_raw_response(raw: dict[str, Any]) -> dict[str, Any]:
    raw = dict(raw or {})
    string_keys = [
        "verified_url", "summary", "reasoning_note", "company_website_url",
        "crane_capacity_range", "crane_capacity_details",
        "responsible_sales_contacts", "contact_source",
    ]
    for key in string_keys:
        if raw.get(key) is None:
            raw[key] = ""
    if raw.get("evidence_urls") is None:
        raw["evidence_urls"] = []
    # Colour fields are set later by the vision pipeline — default them here.
    raw.setdefault("crane_color_scheme", "Unknown")
    raw.setdefault("color_confidence", 0.0)
    raw.setdefault("color_evidence_note", "")
    raw.setdefault("official_website_confidence", 0.0)
    raw.setdefault("site_status", "")
    raw.setdefault("site_rejection_reason", "")
    raw.setdefault("profile_urls", [])
    raw.setdefault("rejected_urls", [])
    raw.setdefault("official_site_debug", "")
    raw.setdefault("enrichment_path", "llm")
    return raw


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _apply_vision_color(
    enrichment: CompanyEnrichment,
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> CompanyEnrichment:
    """Run the vision + text color pipeline and attach results to enrichment."""
    if enrichment.ai_status == "Not Relevant":
        enrichment.crane_color_scheme = "Unknown"
        enrichment.color_confidence = 0.0
        enrichment.color_evidence_note = (
            "Company classified as Not Relevant to crane/heavy-lifting activity."
        )
        return enrichment

    # Only run vision color inference for Active/credible companies.
    if enrichment.ai_status == "Unclear" and enrichment.status_confidence < 0.5:
        enrichment.crane_color_scheme = "Unknown"
        enrichment.color_confidence = 0.0
        enrichment.color_evidence_note = (
            "Color analysis skipped: status confidence too low for reliable inference."
        )
        return enrichment

    inferred = infer_crane_color_scheme(
        company_record=company_record,
        search_results=search_results,
        scraped_pages=scraped_pages,
        settings=settings,
    )

    enrichment.crane_color_scheme = inferred.scheme
    enrichment.color_confidence = inferred.confidence
    enrichment.color_evidence_note = inferred.note
    return enrichment


def _postprocess_enrichment(
    enrichment: CompanyEnrichment,
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> CompanyEnrichment:
    allowed = _allowed_urls(search_results, scraped_pages)
    relevant_counts = _relevant_evidence_counts(company_record, search_results, scraped_pages)

    enrichment.evidence_urls = [u for u in enrichment.evidence_urls if u in allowed]

    if enrichment.verified_url and enrichment.verified_url not in allowed:
        enrichment.verified_url = ""
    if not enrichment.verified_url and enrichment.evidence_urls:
        enrichment.verified_url = enrichment.evidence_urls[0]

    if enrichment.company_website_url and enrichment.company_website_url not in allowed:
        enrichment.company_website_url = ""
    if not enrichment.company_website_url:
        enrichment.company_website_url = _direct_or_verified_url(
            search_results, enrichment.verified_url
        )

    site_resolution = resolve_official_site(
        company_record=company_record,
        search_results=search_results,
        scraped_pages=scraped_pages,
        min_official_score=float(getattr(settings, "site_min_official_score", 60)),
        official_site_required=getattr(settings, "official_site_required", True),
        allow_profile_as_verified_url=getattr(settings, "allow_profile_as_verified_url", False),
        max_profile_evidence_urls=getattr(settings, "max_profile_evidence_urls", 2),
        parked_urls={
            p.url for p in scraped_pages
            if getattr(p, "domain_health", "ok").startswith("parked:")
        },
    )
    enrichment.company_website_url = site_resolution.best_url or ""
    enrichment.official_website_confidence = site_resolution.confidence
    enrichment.site_status = site_resolution.status
    enrichment.site_rejection_reason = site_resolution.rejection_reason
    enrichment.profile_urls = site_resolution.profile_urls
    enrichment.rejected_urls = site_resolution.rejected_urls
    enrichment.official_site_debug = site_resolution.debug

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

    if enrichment.ai_status == "Unclear" and relevant_counts["direct_domain_hits"] > 0:
        enrichment.status_confidence = max(float(enrichment.status_confidence), 0.25)

    if not enrichment.crane_capacity_range.strip():
        enrichment.crane_capacity_range = "Unknown"

    legacy_contacts = _legacy_contact_string(company_record)
    if not enrichment.responsible_sales_contacts.strip() and legacy_contacts:
        enrichment.responsible_sales_contacts = legacy_contacts
        enrichment.contact_confidence = max(float(enrichment.contact_confidence), 0.35)
        enrichment.contact_source = "legacy_workbook"
    elif enrichment.responsible_sales_contacts and not enrichment.contact_source:
        enrichment.contact_source = (
            "website" if enrichment.contact_confidence >= 0.55 else "legacy_workbook"
        )

    return _apply_vision_color(enrichment, company_record, search_results, scraped_pages, settings)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_company_with_llm(
    company_record: dict[str, Any],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    settings: Settings,
) -> CompanyEnrichment:
    """
    Enrich a single company record using the local Ollama text LLM.
    Color analysis is performed by the vision model (LLaVA) in _postprocess_enrichment.
    """
    if not search_results and not scraped_pages:
        return _postprocess_enrichment(
            _fallback_enrichment([], "No web evidence was found.", company_record),
            company_record, [], [], settings,
        )

    if not settings.llm_enabled:
        return _postprocess_enrichment(
            _heuristic_enrichment(
                company_record, search_results, scraped_pages,
                "LLM disabled by LLM_ENABLED=false.",
            ),
            company_record, search_results, scraped_pages, settings,
        )

    if _should_skip_llm_low_evidence(company_record, search_results, scraped_pages, settings):
        return _postprocess_enrichment(
            _heuristic_enrichment(
                company_record, search_results, scraped_pages,
                "LLM skipped: insufficient company-specific evidence.",
            ),
            company_record, search_results, scraped_pages, settings,
        )

    if _should_use_confident_heuristic(company_record, search_results, scraped_pages, settings):
        return _postprocess_enrichment(
            _heuristic_enrichment(
                company_record, search_results, scraped_pages,
                "LLM skipped: strong deterministic company+domain+crane evidence (>1 strong page hits).",
            ),
            company_record, search_results, scraped_pages, settings,
        )

    if OllamaClient is None:
        return _postprocess_enrichment(
            _fallback_enrichment(
                search_results,
                "ollama package not installed. Run: pip install ollama",
                company_record,
            ),
            company_record, search_results, scraped_pages, settings,
        )

    user_prompt = _build_user_prompt(company_record, search_results, scraped_pages, settings)
    cache_key = _llm_cache_key(settings.llm_model, SYSTEM_PROMPT, user_prompt)
    cache = _load_llm_cache(settings)

    try:
        if settings.llm_cache_enabled and cache_key in cache:
            raw = cache[cache_key]
        else:
            client = OllamaClient(host=settings.llm_base_url)
            raw = _call_ollama_json(
                client=client,
                model=settings.llm_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=settings.llm_max_tokens,
            )
            cache[cache_key] = raw
            _save_llm_cache(settings, cache)

        raw = _sanitize_raw_response(raw)
        raw["enrichment_path"] = "llm"
        enrichment = CompanyEnrichment.model_validate(raw)

        if not enrichment.evidence_urls:
            enrichment.evidence_urls = [p.url for p in scraped_pages] or [
                r.url for r in search_results
                if r.source_type in {"legacy_url", "email_domain"}
            ][:3]

        return _postprocess_enrichment(
            enrichment=enrichment,
            company_record=company_record,
            search_results=search_results,
            scraped_pages=scraped_pages,
            settings=settings,
        )

    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        return _postprocess_enrichment(
            _fallback_enrichment(
                search_results=search_results,
                message=f"LLM response could not be validated: {exc}",
                company_record=company_record,
            ),
            company_record, search_results, scraped_pages, settings,
        )
