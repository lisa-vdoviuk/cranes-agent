from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse

from src.schemas import ScrapedPage, SearchResult

SOCIAL_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "xing.com",
    "pinterest.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
}

MARKETPLACE_DOMAINS = {
    "ebay.com",
    "amazon.com",
    "alibaba.com",
    "machineryzone.com",
    "mascus.com",
    "machinio.com",
    "truckscout24.com",
    "truck1.eu",
    "autoline.info",
}

DIRECTORY_DOMAINS = {
    "yellowpages.com",
    "pages.de",
    "kompass.com",
    "europages.com",
    "werliefertwas.de",
    "yellowpages.de",
    "yelp.com",
    "hotfrog.com",
    "local.ch",
    "11880.com",
    "gulesider.no",
}

PROFILE_HINTS = {
    "/profile/",
    "/company/",
    "/unternehmen/",
    "/firma/",
    "/about-us/",
    "/about/",
    "/impressum/",
    "/kontakt/",
    "/kontaktaufnahme/",
    "/pages/",
    "/p/",
    "/people/",
    "/team/",
    "/person/",
}

COMPANY_STOPWORDS = {
    "gmbh", "mbh", "co", "kg", "ag", "ltd", "limited",
    "gruppe", "group", "und", "and", "the", "de", "germany",
    "german", "alemania", "berlin", "company", "firma",
}


@dataclass(frozen=True)
class OfficialSiteResolution:
    best_url: str = ""
    confidence: float = 0.0
    status: str = "no_official_site"
    rejection_reason: str = ""
    profile_urls: list[str] = None
    rejected_urls: list[str] = None
    debug: str = ""

    def __post_init__(self):
        object.__setattr__(self, "profile_urls", self.profile_urls or [])
        object.__setattr__(self, "rejected_urls", self.rejected_urls or [])


def normalize_url(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "n/a"}:
        return ""
    if not re.match(r"https?://", text, flags=re.I):
        text = "https://" + text
    return text


def _domain(url: str) -> str:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    domain = parsed.netloc.lower().removeprefix("www.")
    return domain


def _path(url: str) -> str:
    normalized = normalize_url(url)
    return urlparse(normalized).path.lower()


def classify_url_category(url: str) -> str:
    domain = _domain(url)
    path = _path(url)
    if not domain:
        return "unknown"

    if any(social in domain for social in SOCIAL_DOMAINS):
        return "social"
    if any(market in domain for market in MARKETPLACE_DOMAINS):
        return "marketplace"
    if any(directory in domain for directory in DIRECTORY_DOMAINS):
        return "directory"
    if any(hint in path for hint in PROFILE_HINTS):
        return "profile"
    if path and path != "/" and len(path.split("/")) <= 2 and any(
        hint in path for hint in {"kontakt", "impressum", "about", "company"}
    ):
        return "profile"
    if domain.endswith("wikipedia.org"):
        return "directory"
    return "official_site"


def filter_results_for_scraping(search_results: list[SearchResult]) -> list[SearchResult]:
    filtered: list[SearchResult] = []
    for result in search_results:
        category = classify_url_category(result.url)
        if category in {"profile", "marketplace", "directory", "social"}:
            continue
        filtered.append(result)

    if not filtered:
        return search_results
    return filtered


def _company_tokens(company_name: str) -> list[str]:
    raw = re.split(r"[^a-z0-9äöüß]+", company_name.lower())
    tokens: list[str] = []
    for token in raw:
        if len(token) < 3:
            continue
        if token in COMPANY_STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _score_text(text: str, tokens: list[str]) -> float:
    if not text or not tokens:
        return 0.0
    lowered = text.lower()
    return float(sum(1 for token in tokens if token in lowered)) * 5.0


def _score_candidate(
    url: str,
    category: str,
    company_name: str,
    title: str = "",
    snippet: str = "",
    page_text: str = "",
    existing_web: str = "",
    verified_url: str = "",
) -> float:
    score = 0.0
    if category == "official_site":
        score += 50.0
    elif category == "directory":
        score += 10.0
    elif category == "profile":
        score += 5.0
    elif category == "marketplace":
        score += 2.0
    elif category == "social":
        score += 0.0
    else:
        score += 20.0

    if url.rstrip("/") in {normalize_url(existing_web).rstrip("/"), normalize_url(verified_url).rstrip("/")}:
        score += 20.0

    tokens = _company_tokens(company_name)
    score += _score_text(title, tokens)
    score += _score_text(snippet, tokens)
    score += _score_text(page_text, tokens)

    if any(path_hint in _path(url) for path_hint in {"kontakt", "kontaktaufnahme", "impressum", "about", "unternehmen"}):
        score += 5.0

    return min(max(score, 0.0), 100.0)


def resolve_official_site(
    company_record: dict[str, object],
    search_results: list[SearchResult],
    scraped_pages: list[ScrapedPage],
    min_official_score: float = 60.0,
    official_site_required: bool = True,
    allow_profile_as_verified_url: bool = False,
    max_profile_evidence_urls: int = 2,
    parked_urls: set[str] | None = None,
) -> OfficialSiteResolution:
    """Resolve the best official website URL for a company.

    Parameters
    ----------
    parked_urls:
        Set of URLs already identified as parked / expired / for-sale by the
        parked_domain_detector.  Any URL in this set is immediately rejected
        with status ``dead_or_parked`` rather than being scored as an official
        site.  Scraper-level filtering should mean this set is rarely non-empty,
        but it acts as a defence-in-depth safety net (e.g. for URLs coming from
        the workbook ``existing_web`` column that were never scraped).
    """
    company_name = str(company_record.get("company_name", "") or "").strip()
    existing_web = str(company_record.get("existing_web", "") or "").strip()
    verified_url = str(company_record.get("verified_url", "") or "").strip()
    parked_urls = parked_urls or set()

    candidates: dict[str, dict[str, object]] = {}

    def add_candidate(url: str, title: str = "", snippet: str = "", page_text: str = "", source_type: str = ""):
        normalized = normalize_url(url)
        if not normalized or normalized in candidates:
            return
        category = classify_url_category(normalized)
        score = _score_candidate(
            normalized,
            category,
            company_name,
            title=title,
            snippet=snippet,
            page_text=page_text,
            existing_web=existing_web,
            verified_url=verified_url,
        )
        candidates[normalized] = {
            "category": category,
            "score": score,
            "title": title,
            "snippet": snippet,
            "page_text": page_text,
            "source_type": source_type,
        }

    if existing_web:
        add_candidate(existing_web, title="Existing workbook URL", snippet="", page_text="", source_type="legacy_url")
    if verified_url:
        add_candidate(verified_url, title="Verified URL", snippet="", page_text="", source_type="verified_url")

    for result in search_results:
        add_candidate(result.url, title=result.title, snippet=result.snippet, source_type=result.source_type)
    for page in scraped_pages:
        add_candidate(page.url, title=page.title, snippet="", page_text=page.text, source_type="scraped_page")

    if not candidates:
        return OfficialSiteResolution(
            status="no_official_site",
            rejection_reason="No candidate URLs were available.",
            debug="no candidates",
        )

    sorted_candidates = sorted(candidates.items(), key=lambda item: item[1]["score"], reverse=True)
    best_url, best_info = sorted_candidates[0]
    best_category = best_info["category"]
    best_score = float(best_info["score"])

    profile_urls: list[str] = [url for url, info in candidates.items() if info["category"] == "profile"]
    rejected_urls: list[str] = [f"{url} ({info['category']})" for url, info in candidates.items() if url != best_url]

    # ── Defence-in-depth: reject any best_url flagged as parked ──────────
    if best_url in parked_urls:
        debug = (
            f"best={best_url} score={best_score:.1f} category={best_category} "
            f"status=dead_or_parked (parked_urls hit); candidates={len(candidates)}"
        )
        return OfficialSiteResolution(
            best_url="",
            confidence=0.0,
            status="dead_or_parked",
            rejection_reason="Domain appears to be parked, expired, or for sale.",
            profile_urls=profile_urls[:max_profile_evidence_urls],
            rejected_urls=rejected_urls,
            debug=debug,
        )

    if best_category == "official_site" and best_score >= min_official_score:
        status = "official_site_found"
        rejection_reason = ""
    elif best_category == "official_site" and best_score < min_official_score:
        status = "weak_candidates"
        rejection_reason = "Official-looking site score below threshold."
    elif allow_profile_as_verified_url and best_category in {"profile", "marketplace", "directory"}:
        status = "profile_only"
        rejection_reason = "Profile or platform URL accepted as verified site because profile links are allowed."
    elif best_category == "profile":
        status = "profile_only"
        rejection_reason = "Only profile/platform pages were found."
    elif best_category == "directory":
        status = "weak_candidates"
        rejection_reason = "Only directory/listing pages were found."
    elif best_category == "marketplace":
        status = "weak_candidates"
        rejection_reason = "Only marketplace pages were found."
    else:
        status = "no_official_site"
        rejection_reason = "No official company-owned website could be verified."

    best_url_output = best_url if status == "official_site_found" else ""
    if not official_site_required and not best_url_output and best_category in {"profile", "marketplace", "directory"}:
        best_url_output = best_url

    debug = (
        f"best={best_url} score={best_score:.1f} category={best_category} status={status}; "
        f"profiles={len(profile_urls)} candidates={len(candidates)}"
    )

    return OfficialSiteResolution(
        best_url=best_url_output,
        confidence=best_score / 100.0,
        status=status,
        rejection_reason=rejection_reason,
        profile_urls=profile_urls[:max_profile_evidence_urls],
        rejected_urls=rejected_urls,
        debug=debug,
    )
