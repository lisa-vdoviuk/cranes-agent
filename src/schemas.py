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
    # Optional visual hints extracted from website images. These are noisy by nature,
    # so the LLM must treat them as weak evidence and only use them when plausible.
    image_color_hints: list[str] = Field(default_factory=list)

    @field_validator("url", "title", "text", mode="before")
    @classmethod
    def _none_to_empty_string(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("image_color_hints", mode="before")
    @classmethod
    def _normalize_image_color_hints(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.replace("\n", "|").split("|") if part.strip()]
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

    # New CRM enrichment fields requested in v3.
    company_website_url: str = Field(
        default="",
        description="Best website URL to open from the CRM table; prefer verified_url, official workbook URL, or official email domain website.",
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
        description="Known or likely crane/equipment color scheme, e.g. 'boom - blue; main body - red'. Use Unknown if not supported.",
    )
    color_confidence: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Confidence that the color scheme is based on relevant evidence, not just a website logo or generic photo.",
    )
    color_evidence_note: str = Field(
        default="",
        description="Brief note explaining whether color evidence came from text, images, or was unavailable.",
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
            return [part.strip() for part in value.replace("\n", "|").split("|") if part.strip()]
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
