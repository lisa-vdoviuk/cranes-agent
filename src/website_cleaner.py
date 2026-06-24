from __future__ import annotations

"""Website cleaning and live official-site validation.

This module is intentionally deterministic.  It does not ask the LLM whether a
website is valid; it verifies URLs with live HTTP redirects, parked-domain
fingerprints, platform/domain categories, and company-identity signals.

Policy used by the CRM dataset:
- company_website_url may contain only a live official/company-owned website.
- directory/social/marketplace/profile pages are rejected as CRM website links.
- parked, expired, registrar, default hosting, and unpaid-domain pages are
  rejected even when they return HTTP 200.
- if no official site is validated, the clean dataset marks website activity as
  Inactive and clears company_website_url.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import Settings, ensure_cache_dirs, get_settings
from src.constants import CRANE_TERMS
from src.parked_domain_detector import DomainHealthResult, check_domain_health
from src.site_verifier import classify_url_category, normalize_url

LOGGER = logging.getLogger("crane_enrichment.website_cleaner")

FREE_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "hotmail.com", "hotmail.de", "outlook.com",
    "live.com", "yahoo.com", "yahoo.de", "aol.com", "gmx.de", "web.de",
    "t-online.de", "icloud.com", "msn.com", "proton.me", "protonmail.com",
}

LEGAL_STOPWORDS = {
    "gmbh", "mbh", "co", "kg", "ag", "ltd", "limited", "llc", "sarl", "sas",
    "gruppe", "group", "holding", "und", "and", "the", "de", "germany",
    "german", "alemania", "company", "firma", "nutzfahrzeug", "baumaschinen",
    "maschinen", "service", "services", "technik", "technologies", "solutions",
}

LEGAL_CONTACT_SIGNALS = (
    "impressum", "kontakt", "contact", "datenschutz", "privacy", "legal notice",
    "geschäftsführer", "geschaeftsfuehrer", "ust-id", "ustid", "handelsregister",
    "registergericht", "adresse", "anschrift", "phone", "telefon", "e-mail", "email",
)

# Additional platform/listing hosts that must never become company_website_url.
# Keep this list conservative; it is only for CRM website cleaning, not evidence.
BLOCKED_PLATFORM_DOMAINS = {
    "facebook.com", "instagram.com", "linkedin.com", "xing.com", "youtube.com",
    "youtu.be", "tiktok.com", "pinterest.com", "northdata.de", "northdata.com",
    "kompass.com", "kompass.de", "europages.com", "werliefertwas.de",
    "wlw.de", "11880.com", "yellowpages.com", "yellowpages.de", "firmenwissen.de",
    "companyhouse.de", "implisense.com", "creditsafe.com", "dnb.com",
    "opencorporates.com", "crunchbase.com", "yelp.com", "google.com",
    "maps.google.com", "business.site", "machineryzone.com", "mascus.com",
    "machinio.com", "truckscout24.com", "truck1.eu", "autoline.info",
    "cranetrader.com", "cranetrader.de", "cranemarket.com", "ebay.com",
    "amazon.com", "alibaba.com",
}

# Default host / unpaid domain / registrar landing fingerprints that are not
# always caught by classic "domain for sale" phrases.  We inspect title and the
# first part of body text, so broad phrases here are acceptable when combined
# with weak company identity.
PLACEHOLDER_PHRASES = (
    # English registrar / expired / default pages
    "domain payment", "pay for this domain", "renew your domain", "domain renewal",
    "pending renewal", "redemption period", "clienthold", "serverhold",
    "this domain has recently been registered", "this domain has been registered",
    "this domain was recently registered", "domain has been registered",
    "this domain is reserved", "domain reserved", "reserved domain",
    "future home of", "web server's default page", "web server default page",
    "apache2 ubuntu default page", "apache2 debian default page", "nginx default page",
    "plesk default page", "parallels plesk panel", "site not published",
    "website not published", "this website is not yet configured",
    "this site is currently unavailable", "default website page", "default site page",
    "placeholder page", "there is no website configured at this address",
    "no website is currently present", "this domain points to", "hosted by godaddy",
    "parking page", "temporary landing page",
    # German / European hosting placeholders
    "diese domain wurde registriert", "domain wurde registriert",
    "diese domain ist reserviert", "domain ist reserviert", "domain reserviert",
    "diese internetpräsenz wurde noch nicht veröffentlicht",
    "diese internetpraesenz wurde noch nicht veroeffentlicht",
    "hier entsteht eine neue internetpräsenz", "hier entsteht eine neue internetpraesenz",
    "hier entsteht eine neue website", "hier entsteht in kürze",
    "webseite wurde noch nicht veröffentlicht", "webseite wurde noch nicht veroeffentlicht",
    "webspace wurde erfolgreich eingerichtet", "hosting wurde eingerichtet",
    "standardseite", "standard-seite", "standard website", "confixx",
    "plesk obsidian", "diese seite wurde deaktiviert", "account wurde gesperrt",
    "domain parken", "domain geparkt", "geparkte domain", "domaingrabber",
    "kunden-login", "kundenmenü", "kundenmenue", "hosting paket", "hosting-paket",
)

TWO_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "com.au", "com.br", "com.tr", "com.pl", "co.at",
    "com.es", "com.mx", "com.ar", "co.nz", "co.za",
}


@dataclass(frozen=True)
class WebsiteValidation:
    original_url: str
    accepted_url: str = ""
    final_url: str = ""
    final_domain: str = ""
    category: str = "unknown"
    status: str = "rejected"  # accepted | rejected | fetch_failed
    reason: str = ""
    identity_score: float = 0.0
    domain_health: str = "ok"
    title: str = ""
    checked_variants: list[str] = field(default_factory=list)

    @property
    def is_accepted(self) -> bool:
        return bool(self.accepted_url and self.status == "accepted")


@dataclass(frozen=True)
class RowWebsiteCleaningResult:
    accepted_url: str = ""
    website_activity_status: str = "Inactive"  # Active | Inactive
    clean_website_status: str = "inactive_no_official_site"
    clean_website_confidence: float = 0.0
    clean_website_reason: str = ""
    clean_website_checked_at: str = ""
    clean_website_original_url: str = ""
    clean_website_final_url: str = ""
    clean_website_final_domain: str = ""
    clean_website_category: str = ""
    clean_website_identity_score: float = 0.0
    clean_website_health: str = ""
    clean_website_candidates_checked: str = ""
    clean_website_rejected_candidates: str = ""


# ---------------------------------------------------------------------------
# URL and text helpers
# ---------------------------------------------------------------------------

def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (compatible; CraneCRMWebsiteCleaner/1.0; +https://example.local)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de,en;q=0.9,es;q=0.7",
    }


def _clean_optional(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "n/a"}:
        return ""
    return text




def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _as_mutable_object_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose columns accept mixed audit values.

    Pandas can load CSVs as Arrow string columns (`string[pyarrow]`).  During
    cleaning we intentionally write numeric audit fields such as 0.0/0.55 into
    columns that may have been read as strings.  Arrow string columns reject
    that with: "Invalid value '0.0' for dtype 'str'".  Object dtype keeps the
    cleaning stage robust and `to_csv` serializes the values normally.
    """
    return df.copy().astype("object")


def _domain(url: str) -> str:
    url = normalize_url(url)
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _registered_domain(host: str) -> str:
    host = (host or "").lower().strip(".").removeprefix("www.")
    if not host or "." not in host:
        return host
    parts = host.split(".")
    suffix2 = ".".join(parts[-2:])
    suffix3 = ".".join(parts[-3:]) if len(parts) >= 3 else suffix2
    if suffix2 in TWO_LEVEL_SUFFIXES and len(parts) >= 3:
        return suffix3
    return suffix2


def _same_registered_domain(a: str, b: str) -> bool:
    da = _registered_domain(_domain(a) or a)
    db = _registered_domain(_domain(b) or b)
    return bool(da and db and da == db)


def _root_url(url: str, *, scheme: str | None = None, www: bool | None = None) -> str:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    if not parsed.netloc:
        return ""
    host = parsed.netloc.lower()
    if www is True and not host.startswith("www."):
        host = "www." + host
    elif www is False and host.startswith("www."):
        host = host[4:]
    return urlunparse((scheme or parsed.scheme or "https", host, "/", "", "", ""))


def _url_variants(url: str) -> list[str]:
    normalized = normalize_url(url)
    if not normalized:
        return []
    parsed = urlparse(normalized)
    variants = [normalized]
    # Root variants catch cases where an old deep link is dead but the site root works.
    for scheme in ("https", "http"):
        for www in (None, True, False):
            root = _root_url(normalized, scheme=scheme, www=www)
            if root:
                variants.append(root)
    # Keep original http/https path variants as well.
    if parsed.scheme == "https":
        variants.append(urlunparse(("http", parsed.netloc, parsed.path or "/", "", parsed.query, "")))
    elif parsed.scheme == "http":
        variants.append(urlunparse(("https", parsed.netloc, parsed.path or "/", "", parsed.query, "")))

    output: list[str] = []
    for item in variants:
        if item and item not in output:
            output.append(item)
    return output[:8]


def _split_urls(value: object) -> list[str]:
    text = _clean_optional(value)
    if not text:
        return []
    output: list[str] = []
    for part in re.split(r"\s*\|\s*|\s+", text.replace("\n", "|")):
        url = normalize_url(part.strip().strip(";,.)]}>\"'"))
        if url and _domain(url) and url not in output:
            output.append(url)
    return output


def _email_domains(value: object) -> list[str]:
    text = _clean_optional(value)
    if not text:
        return []
    domains: list[str] = []
    for match in re.finditer(r"[A-Z0-9._%+\-]+@([A-Z0-9.\-]+\.[A-Z]{2,})", text, flags=re.I):
        domain = match.group(1).lower().strip(".").removeprefix("www.")
        if domain not in FREE_EMAIL_DOMAINS and domain not in domains:
            domains.append(domain)
    return domains


def _company_tokens(company_name: str) -> list[str]:
    raw = re.split(r"[^a-z0-9äöüß]+", str(company_name).lower())
    output: list[str] = []
    for token in raw:
        if len(token) < 3 or token in LEGAL_STOPWORDS:
            continue
        if token not in output:
            output.append(token)
    return output


def _normalize_company_name(company_name: str) -> str:
    tokens = _company_tokens(company_name)
    return " ".join(tokens)


def _extract_title_and_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html or "", "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = re.sub(r"\s+", " ", soup.title.string).strip()
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip()
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title, text


def _phrase_hit(text: str, phrases: Iterable[str]) -> str:
    lowered = (text or "").lower()
    for phrase in phrases:
        if phrase in lowered:
            return phrase
    return ""


def _placeholder_health(html: str, title: str, text: str) -> DomainHealthResult:
    haystack = f"{title}\n{text[:1200]}"
    hit = _phrase_hit(haystack, PLACEHOLDER_PHRASES)
    if hit:
        return DomainHealthResult(True, "placeholder_phrase", hit)
    # Very small pages with hosting control-panel markers are almost never real company sites.
    html_lower = (html or "").lower()[:6000]
    marker_hit = _phrase_hit(
        html_lower,
        (
            "wp-admin/install.php", "plesk-site-preview", "cpanel", "webmail login",
            "powered by plesk", "parallels", "defaultwebpage.cgi", "cgi-sys/defaultwebpage.cgi",
        ),
    )
    if marker_hit and len(text) < 1200:
        return DomainHealthResult(True, "hosting_default_marker", marker_hit)
    return DomainHealthResult(False, "none", "")


def _blocked_platform_domain(host: str) -> str:
    host = (host or "").lower().removeprefix("www.")
    if not host:
        return ""
    for blocked in BLOCKED_PLATFORM_DOMAINS:
        if host == blocked or host.endswith("." + blocked):
            return blocked
    return ""


def _identity_score(
    *,
    company_name: str,
    original_url: str,
    final_url: str,
    title: str,
    text: str,
    email_domains: list[str],
) -> float:
    tokens = _company_tokens(company_name)
    normalized_name = _normalize_company_name(company_name)
    haystack = f"{title} {final_url} {text[:5000]}".lower()
    final_host = _domain(final_url)
    registered = _registered_domain(final_host)
    score = 0.0

    if normalized_name and normalized_name in re.sub(r"\s+", " ", haystack):
        score += 35.0

    matched_tokens = [token for token in tokens if token in haystack]
    score += min(35.0, 8.0 * len(matched_tokens))

    # Distinctive company token in the registered domain is a strong ownership signal.
    for token in tokens:
        ascii_token = token.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
        if token in registered or ascii_token in registered:
            score += 25.0
            break

    for email_domain in email_domains:
        if _registered_domain(email_domain) == registered:
            score += 30.0
            break

    if _same_registered_domain(original_url, final_url):
        score += 10.0

    if any(signal in haystack for signal in LEGAL_CONTACT_SIGNALS):
        score += 12.0

    if any(term in haystack for term in CRANE_TERMS):
        score += 10.0

    # Penalize generic pages that do not identify the company.
    if len(text) < 250 and not matched_tokens:
        score -= 25.0
    if _phrase_hit(haystack[:1200], ("privacy policy", "cookie policy", "404", "not found")) and not matched_tokens:
        score -= 15.0

    return max(0.0, min(100.0, score))


def _fetch_html(url: str, timeout: int) -> tuple[str, str, str]:
    response = requests.get(
        url,
        headers=_request_headers(),
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content-type: {content_type}")
    return response.text, str(response.url), content_type


# ---------------------------------------------------------------------------
# Candidate collection and validation
# ---------------------------------------------------------------------------

def website_candidate_urls(row: pd.Series | dict, max_candidates: int = 8) -> list[str]:
    """Collect URLs that should be checked for official-website validity."""
    get = row.get if isinstance(row, dict) else row.get
    candidates: list[str] = []

    # Prioritize currently accepted fields, then historical input/evidence.
    for col in (
        "company_website_url", "verified_url", "existing_web", "website", "web",
        "url", "evidence_urls", "profile_urls",
    ):
        for url in _split_urls(get(col, "")):
            if url not in candidates:
                candidates.append(url)

    # Email domains are good official-site candidates, but must still pass live validation.
    for domain in _email_domains(get("emails", "")):
        for url in (f"https://{domain}", f"https://www.{domain}"):
            if url not in candidates:
                candidates.append(url)

    return candidates[:max_candidates]


def validate_official_website_url(
    url: str,
    *,
    company_name: str,
    email_domains: list[str],
    settings: Settings,
    min_identity_score: float | None = None,
) -> WebsiteValidation:
    """Live-validate one URL as a real official company website."""
    original_url = normalize_url(url)
    category = classify_url_category(original_url)
    original_host = _domain(original_url)
    min_identity_score = float(
        min_identity_score
        if min_identity_score is not None
        else getattr(settings, "clean_min_identity_score", 50)
    )

    if not original_url or not original_host:
        return WebsiteValidation(original_url=original_url, category="unknown", reason="blank_or_invalid_url")

    blocked = _blocked_platform_domain(original_host)
    if blocked or category in {"social", "marketplace", "directory", "profile"}:
        return WebsiteValidation(
            original_url=original_url,
            category=category,
            final_domain=original_host,
            reason=f"platform_or_profile_url:{blocked or category}",
        )

    checked: list[str] = []
    errors: list[str] = []
    best_rejection: WebsiteValidation | None = None

    for variant in _url_variants(original_url):
        checked.append(variant)
        try:
            html, final_url, _content_type = _fetch_html(variant, int(settings.request_timeout_seconds))
        except Exception as exc:
            errors.append(f"{variant}: {type(exc).__name__}: {exc}")
            continue

        final_host = _domain(final_url)
        final_blocked = _blocked_platform_domain(final_host)
        final_category = classify_url_category(final_url)
        title, text = _extract_title_and_text(html)
        parking = check_domain_health(final_url, html, text)
        placeholder = _placeholder_health(html, title, text)
        health = parking if parking.is_parked else placeholder
        health_tag = "ok" if not health.is_parked else f"{health.detection_method}:{health.evidence}"

        identity = _identity_score(
            company_name=company_name,
            original_url=original_url,
            final_url=final_url,
            title=title,
            text=text,
            email_domains=email_domains,
        )

        common = dict(
            original_url=original_url,
            final_url=final_url,
            final_domain=final_host,
            category=final_category,
            identity_score=identity,
            domain_health=health_tag,
            title=title,
            checked_variants=checked.copy(),
        )

        if final_blocked or final_category in {"social", "marketplace", "directory", "profile"}:
            best_rejection = WebsiteValidation(
                **common,
                status="rejected",
                reason=f"redirected_to_platform_or_profile:{final_blocked or final_category}",
            )
            continue

        if health.is_parked:
            return WebsiteValidation(
                **common,
                status="rejected",
                reason=f"parked_or_placeholder:{health.detection_method}:{health.evidence}",
            )

        # External redirects can be valid rebrands, but only with strong identity.
        if not _same_registered_domain(original_url, final_url) and identity < max(min_identity_score, 65.0):
            best_rejection = WebsiteValidation(
                **common,
                status="rejected",
                reason="external_redirect_without_strong_company_identity",
            )
            continue

        if identity < min_identity_score:
            best_rejection = WebsiteValidation(
                **common,
                status="rejected",
                reason=f"weak_company_identity:{identity:.0f}<{min_identity_score:.0f}",
            )
            continue

        return WebsiteValidation(
            **common,
            accepted_url=final_url,
            status="accepted",
            reason="live_official_site_validated",
        )

    if best_rejection is not None:
        return best_rejection

    return WebsiteValidation(
        original_url=original_url,
        final_domain=original_host,
        category=category,
        status="fetch_failed",
        reason="fetch_failed:" + " | ".join(errors[:3]),
        checked_variants=checked,
    )


def clean_website_for_row(
    row: pd.Series | dict,
    *,
    settings: Settings,
    max_candidates: int | None = None,
    min_identity_score: float | None = None,
) -> RowWebsiteCleaningResult:
    get = row.get if isinstance(row, dict) else row.get
    company_name = _clean_optional(get("company_name", ""))
    email_domains = _email_domains(get("emails", ""))
    candidates = website_candidate_urls(
        row,
        max_candidates=int(max_candidates or getattr(settings, "clean_max_website_candidates", 8)),
    )
    checked_at = datetime.now(timezone.utc).isoformat()

    if not candidates:
        return RowWebsiteCleaningResult(
            clean_website_checked_at=checked_at,
            clean_website_status="inactive_no_candidates",
            clean_website_reason="No candidate website URLs were available.",
        )

    validations: list[WebsiteValidation] = []
    for candidate in candidates:
        validation = validate_official_website_url(
            candidate,
            company_name=company_name,
            email_domains=email_domains,
            settings=settings,
            min_identity_score=min_identity_score,
        )
        validations.append(validation)
        LOGGER.debug(
            "Website clean candidate company=%s url=%s status=%s reason=%s final=%s identity=%.0f health=%s",
            company_name,
            candidate,
            validation.status,
            validation.reason,
            validation.final_url,
            validation.identity_score,
            validation.domain_health,
        )
        if validation.is_accepted:
            return RowWebsiteCleaningResult(
                accepted_url=validation.accepted_url,
                website_activity_status="Active",
                clean_website_status="active_official_site",
                clean_website_confidence=validation.identity_score / 100.0,
                clean_website_reason=validation.reason,
                clean_website_checked_at=checked_at,
                clean_website_original_url=validation.original_url,
                clean_website_final_url=validation.final_url,
                clean_website_final_domain=validation.final_domain,
                clean_website_category=validation.category,
                clean_website_identity_score=validation.identity_score,
                clean_website_health=validation.domain_health,
                clean_website_candidates_checked=" | ".join(candidates),
                clean_website_rejected_candidates=" | ".join(
                    f"{v.original_url} => {v.reason}" for v in validations if not v.is_accepted
                ),
            )

    # No accepted URL.  Pick the strongest rejection to expose in summary fields.
    def rank(v: WebsiteValidation) -> tuple[int, float]:
        if "parked" in v.reason or "placeholder" in v.reason:
            return (4, v.identity_score)
        if v.status == "fetch_failed":
            return (3, v.identity_score)
        if "platform" in v.reason or "profile" in v.reason:
            return (2, v.identity_score)
        return (1, v.identity_score)

    best = sorted(validations, key=rank, reverse=True)[0]
    if "parked" in best.reason or "placeholder" in best.reason:
        status = "inactive_dead_or_parked"
    elif best.status == "fetch_failed":
        status = "inactive_fetch_failed"
    elif "platform" in best.reason or "profile" in best.reason:
        status = "inactive_profile_or_platform_only"
    else:
        status = "inactive_weak_or_unverified"

    return RowWebsiteCleaningResult(
        website_activity_status="Inactive",
        clean_website_status=status,
        clean_website_confidence=0.0,
        clean_website_reason=best.reason or "No official active website could be validated.",
        clean_website_checked_at=checked_at,
        clean_website_original_url=best.original_url,
        clean_website_final_url=best.final_url,
        clean_website_final_domain=best.final_domain,
        clean_website_category=best.category,
        clean_website_identity_score=best.identity_score,
        clean_website_health=best.domain_health,
        clean_website_candidates_checked=" | ".join(candidates),
        clean_website_rejected_candidates=" | ".join(
            f"{v.original_url} => {v.reason}" for v in validations
        ),
    )


# ---------------------------------------------------------------------------
# DataFrame / CSV API
# ---------------------------------------------------------------------------

def clean_enriched_dataframe(
    df: pd.DataFrame,
    *,
    settings: Settings | None = None,
    mark_inactive_without_official_site: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return a clean copy of an enriched dataframe with website audit columns."""
    settings = settings or get_settings()
    ensure_cache_dirs(settings)
    log = logger or LOGGER
    output = _as_mutable_object_dataframe(df)

    # Preserve pre-clean values for audit/reversibility.
    for col in ("company_website_url", "verified_url", "ai_status", "status_confidence", "site_status"):
        if col in output.columns and f"pre_clean_{col}" not in output.columns:
            output[f"pre_clean_{col}"] = output[col]

    required_cols = [
        "company_website_url", "site_status", "site_rejection_reason",
        "official_website_confidence", "verified_url", "ai_status", "status_confidence",
        "summary",
    ]
    for col in required_cols:
        if col not in output.columns:
            output[col] = "" if col not in {"status_confidence", "official_website_confidence"} else 0.0

    for idx, row in output.iterrows():
        company_name = _clean_optional(row.get("company_name", "")) or f"row-{idx + 1}"
        log.info("[%s/%s] Cleaning website: %s", idx + 1, len(output), company_name)
        result = clean_website_for_row(row, settings=settings)

        result_dict = result.__dict__
        for key, value in result_dict.items():
            output.at[idx, key] = value

        # Clean primary CRM URL fields.
        output.at[idx, "company_website_url"] = result.accepted_url
        output.at[idx, "official_website_confidence"] = result.clean_website_confidence

        if result.accepted_url:
            output.at[idx, "site_status"] = "official_site_found"
            output.at[idx, "site_rejection_reason"] = ""
            # verified_url is allowed to equal the official site.  We do not keep profiles here.
            output.at[idx, "verified_url"] = result.accepted_url
            if mark_inactive_without_official_site and _clean_optional(output.at[idx, "ai_status"]) == "Inactive":
                output.at[idx, "ai_status"] = "Active"
                output.at[idx, "status_confidence"] = max(_safe_float(output.at[idx, "status_confidence"]), 0.55)
        else:
            output.at[idx, "site_status"] = "dead_or_parked" if "parked" in result.clean_website_status else "no_official_site"
            output.at[idx, "site_rejection_reason"] = result.clean_website_reason
            output.at[idx, "verified_url"] = ""
            if mark_inactive_without_official_site:
                output.at[idx, "ai_status"] = "Inactive"
                output.at[idx, "status_confidence"] = min(_safe_float(output.at[idx, "status_confidence"]), 0.20)
                summary = _clean_optional(output.at[idx, "summary"])
                prefix = "No active official website was validated."
                if summary and prefix.lower() not in summary.lower():
                    output.at[idx, "summary"] = f"{prefix} {summary}"
                elif not summary:
                    output.at[idx, "summary"] = prefix

    return output


def default_clean_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.name == "enriched_pipeline.csv":
        return path.with_name("enriched_pipeline_clean.csv")
    if path.name == "enriched_companies.csv":
        return path.with_name("enriched_companies_clean.csv")
    return path.with_name(path.stem + "_clean" + path.suffix)


def clean_enriched_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    settings: Settings | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    input_path = Path(input_csv)
    output_path = Path(output_csv) if output_csv else default_clean_output_path(input_path)
    df = pd.read_csv(input_path, dtype=str)
    clean_df = clean_enriched_dataframe(df, settings=settings, logger=logger)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    return clean_df
