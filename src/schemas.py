from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


CompanyStatus = Literal[
    "Active",
    "Acquired",
    "Defunct",
    "Merged",
    "Rebranded",
    "Unclear",
    "Not Relevant",
]

MarketRole = Literal[
    "Manufacturer",
    "Dealer",
    "Rental Company",
    "Service Provider",
    "Parts Supplier",
    "Parent Company",
    "Unknown",
]

EnrichmentPath = Literal["llm", "heuristic", "fallback"]

# Formalised site-status values produced by resolve_official_site() and the
# parked-domain detector.  "dead_or_parked" is now actually emitted (previously
# referenced in docstrings but never produced by the code).
SiteStatus = Literal[
    "official_site_found",
    "dead_or_parked",
    "profile_only",
    "weak_candidates",
    "no_official_site",
]


class SearchResult(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""
    source_type: str = "search"  # search | legacy_url | email_domain | fallback
    relevance_score: float = 0.0

    @field_validator("title", "url", "snippet", "source_type", mode="before")
    @classmethod
    def _none_to_empty_string(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)


class ScrapedPage(BaseModel):
    url: str
    title: str = ""
    text: str = ""
    # Crane image URLs found on this page (downloaded and analysed by the vision model).
    crane_image_urls: list[str] = Field(default_factory=list)
    # Health verdict from parked_domain_detector.  Values: "ok" or
    # "parked:<method>:<evidence>" — stored in page cache for audit traceability.
    domain_health: str = "ok"

    @field_validator("url", "title", "text", mode="before")
    @classmethod
    def _none_to_empty_string(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("crane_image_urls", mode="before")
    @classmethod
    def _normalize_crane_image_urls(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [p.strip() for p in value.replace("\n", "|").split("|") if p.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []


class CompanyEnrichment(BaseModel):
    ai_status: CompanyStatus = Field(
        description="Current apparent status of the company in relation to the mobile crane market."
    )
    status_confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score from 0 to 1 based only on supplied evidence.",
    )
    market_role: MarketRole
    verified_url: str = Field(
        default="",
        description="Best official or reliable source URL supporting the assessment. Empty when no relevant evidence exists.",
    )
    summary: str = Field(description="Brief CRM-friendly summary of what was found.")
    evidence_urls: list[str] = Field(
        default_factory=list,
        description="URLs used as evidence.",
    )
    reasoning_note: str = Field(
        description="Short explanation of the decision. Do not include hidden chain-of-thought."
    )

    company_website_url: str = Field(
        default="",
        description="Best website URL to open from the CRM table; prefer verified_url, official workbook URL, or official email domain website.",
    )

    official_website_confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence that company_website_url is the official company-owned website, not a profile or marketplace.",
    )
    site_status: str = Field(
        default="",
        description="Official site resolution status: official_site_found, profile_only, dead_or_parked, weak_candidates, or no_official_site.",
    )
    site_rejection_reason: str = Field(
        default="",
        description="Short reason why no official website was accepted, or why candidates were rejected.",
    )
    profile_urls: list[str] = Field(
        default_factory=list,
        description="Company profile/platform/marketplace URLs found but not accepted as official websites.",
    )
    rejected_urls: list[str] = Field(
        default_factory=list,
        description="Rejected URL candidates with reasons, for audit/debugging.",
    )
    official_site_debug: str = Field(
        default="",
        description="Compact debug trace of official-site candidate scoring.",
    )

    crane_capacity_range: str = Field(
        default="Unknown",
        description="Condensed capacity range of mobile cranes the company works with, e.g. '40-700 t'. Use 'Unknown' when not evidenced.",
    )
    crane_capacity_details: str = Field(
        default="",
        description="Short details about crane models, capacity classes, or fleet capacity evidence.",
    )
    responsible_sales_contacts: str = Field(
        default="",
        description="Names, roles, emails, and phones of likely responsible contacts for crane sales/purchase enquiries.",
    )
    contact_confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence that the extracted contacts are relevant for crane buying/selling enquiries.",
    )
    contact_source: str = Field(
        default="none",
        description="Source of contact evidence: website, legacy_workbook, both, or none.",
    )
    crane_color_scheme: str = Field(
        default="Unknown",
        description="Physical crane/equipment color scheme determined by vision model analysis of actual crane images.",
    )
    color_confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence that the color scheme reflects the actual physical equipment paint, not website UI.",
    )
    color_evidence_note: str = Field(
        default="",
        description="Brief note explaining how color was determined (vision model, text, or unavailable).",
    )

    # Observability: which pipeline path produced this record.
    enrichment_path: EnrichmentPath = Field(
        default="fallback",
        description="Which pipeline path produced this enrichment: llm, heuristic, or fallback.",
    )

    @field_validator(
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
        mode="before",
    )
    @classmethod
    def _none_to_empty_string(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("evidence_urls", mode="before")
    @classmethod
    def _normalize_evidence_urls(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [p.strip() for p in value.replace("\n", "|").split("|") if p.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    @field_validator("status_confidence", "contact_confidence", "color_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> float:
        if value is None or value == "":
            return 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    @field_validator("enrichment_path", mode="before")
    @classmethod
    def _coerce_enrichment_path(cls, value: object) -> str:
        if value in {"llm", "heuristic", "fallback"}:
            return str(value)
        return "fallback"
