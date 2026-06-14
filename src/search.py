from __future__ import annotations

import re
import time
from typing import Any
from urllib.parse import urlparse

from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings
from src.schemas import SearchResult
from src.utils import read_json, write_json


FREE_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com", "hotmail.com", "hotmail.de", "outlook.com",
    "live.com", "yahoo.com", "yahoo.de", "aol.com", "gmx.de", "web.de",
    "t-online.de", "icloud.com", "msn.com", "proton.me", "protonmail.com",
}

LEGAL_STOPWORDS = {
    "gmbh", "mbh", "co", "kg", "ag", "ltd", "limited", "gruppe", "group",
    "und", "and", "the", "de", "germany", "german", "alemania", "berlin",
    "company", "firma", "nutzfahrzeug", "baumaschinen", "bohr", "rammtechnik",
}

GENERIC_CRANE_DOMAINS = {
    "wikipedia.org", "truckcenter24.de", "rental-portal.com", "cranetrader.de",
    "cranetrader.com", "cranemarket.com", "ensun.io", "pinterest.com",
}


# ----------------------------
# Cleaning / extraction helpers
# ----------------------------


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null", "n/a"}


def _first_url(value: object) -> str:
    if _is_blank(value):
        return ""
    text = str(value).strip()
    candidates = re.split(r"\s*\|\s*|\s+", text)
    for candidate in candidates:
        candidate = candidate.strip().strip(";,)")
        if not candidate:
            continue
        if not re.match(r"https?://", candidate, flags=re.I):
            candidate = "https://" + candidate
        parsed = urlparse(candidate)
        if parsed.netloc and "." in parsed.netloc:
            return candidate
    return ""


def _domain_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url if re.match(r"https?://", url, flags=re.I) else "https://" + url)
    domain = parsed.netloc.lower().removeprefix("www.")
    return domain


def _domains_from_emails(value: object) -> list[str]:
    if _is_blank(value):
        return []

    domains: list[str] = []
    for match in re.finditer(r"[A-Z0-9._%+\-]+@([A-Z0-9.\-]+\.[A-Z]{2,})", str(value), flags=re.I):
        domain = match.group(1).lower().strip(".")
        domain = domain.removeprefix("www.")
        if domain not in FREE_EMAIL_DOMAINS and domain not in domains:
            domains.append(domain)
    return domains


def _company_tokens(company_name: str) -> list[str]:
    raw = re.split(r"[^a-z0-9äöüß]+", company_name.lower())
    tokens = []
    for token in raw:
        if len(token) < 3:
            continue
        if token in LEGAL_STOPWORDS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens


def _result_haystack(result: SearchResult | dict[str, Any]) -> str:
    if isinstance(result, SearchResult):
        return f"{result.title} {result.url} {result.snippet}".lower()
    return f"{result.get('title', '')} {result.get('href', '') or result.get('url', '')} {result.get('body', '') or result.get('snippet', '')}".lower()


def _score_result(
    company_name: str,
    result: SearchResult,
    official_domains: list[str],
) -> float:
    haystack = _result_haystack(result)
    tokens = _company_tokens(company_name)
    score = 0.0

    # Exact name or distinctive token matches are strong signals.
    normalized_name = re.sub(r"\s+", " ", company_name.lower()).strip()
    if normalized_name and normalized_name in haystack:
        score += 5.0

    for token in tokens:
        if token in haystack:
            score += 1.5

    result_domain = _domain_from_url(result.url)
    if result_domain in official_domains:
        score += 6.0
    elif any(result_domain.endswith("." + domain) for domain in official_domains):
        score += 5.0

    if result.source_type in {"legacy_url", "email_domain"}:
        score += 4.0

    # Generic crane-market pages should not be treated as company evidence unless
    # they also clearly mention the company.
    if any(generic in result_domain for generic in GENERIC_CRANE_DOMAINS):
        score -= 3.0

    return round(max(score, 0.0), 2)


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    output: list[SearchResult] = []
    for result in results:
        key = result.url.strip().rstrip("/").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(result)
    return output


# ----------------------------
# Query building
# ----------------------------


def _build_queries(
    company_name: str,
    country: str = "",
    existing_web: str = "",
    emails: str = "",
    contacts: str = "",
) -> tuple[list[str], list[str], list[SearchResult]]:
    direct_results: list[SearchResult] = []
    official_domains: list[str] = []

    direct_url = _first_url(existing_web)
    if direct_url:
        domain = _domain_from_url(direct_url)
        if domain:
            official_domains.append(domain)
        direct_results.append(
            SearchResult(
                title="Legacy workbook URL",
                url=direct_url,
                snippet="URL stored in the legacy Excel workbook.",
                source_type="legacy_url",
                relevance_score=10.0,
            )
        )

    for domain in _domains_from_emails(emails):
        if domain not in official_domains:
            official_domains.append(domain)
        direct_results.append(
            SearchResult(
                title="Company email domain candidate",
                url=f"https://{domain}",
                snippet=f"Candidate website inferred from company email domain {domain}.",
                source_type="email_domain",
                relevance_score=8.0,
            )
        )
        direct_results.append(
            SearchResult(
                title="Company email domain candidate",
                url=f"https://www.{domain}",
                snippet=f"Candidate website inferred from company email domain {domain}.",
                source_type="email_domain",
                relevance_score=8.0,
            )
        )

    clean_country = "Germany" if str(country).strip().lower() == "alemania" else str(country).strip()

    queries = [
        f'"{company_name}"',
        f'"{company_name}" {clean_country} official website Impressum Kontakt',
        f'"{company_name}" Kran Mobilkran Autokran Kranverleih',
        f'"{company_name}" crane mobile crane crane rental used crane',
        # v3: targeted queries for capacity/fleet and responsible contacts.
        f'"{company_name}" Fuhrpark Mobilkran Tragkraft Tonnen',
        f'"{company_name}" fleet mobile crane capacity tonnes',
        f'"{company_name}" Ansprechpartner Verkauf Kran Kontakt',
        f'"{company_name}" used cranes sales contact buyer seller',
    ]

    if contacts and not _is_blank(contacts):
        # Use only the first contact phrase. This helps for small/old companies.
        first_contact = str(contacts).split("|")[0].strip()
        if first_contact:
            queries.append(f'"{company_name}" "{first_contact}"')

    for domain in official_domains:
        queries.append(f'site:{domain} Kran OR Mobilkran OR crane OR Impressum OR Kontakt')
        queries.append(f'"{domain}" "{company_name}"')

    # Remove accidental duplicates while preserving order.
    deduped_queries: list[str] = []
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        if query and query not in deduped_queries:
            deduped_queries.append(query)

    return deduped_queries, official_domains, _dedupe_results(direct_results)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def _ddgs_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    with DDGS() as ddgs:
        return list(
            ddgs.text(
                query,
                region="de-de",
                safesearch="moderate",
                max_results=max_results,
                backend="auto",
            )
        )


# ----------------------------
# Public API
# ----------------------------


def search_company_web(
    company_name: str,
    settings: Settings,
    country: str = "",
    existing_web: str = "",
    legacy_info: str = "",
    emails: str = "",
    contacts: str = "",
    use_cache: bool = True,
) -> list[SearchResult]:
    """
    Search strategy v2:
    1. Always include direct URLs from the workbook and company email domains.
    2. Run several narrow exact-name searches instead of one broad generic crane query.
    3. Score and filter results so generic Liebherr/Demag/Tadano pages are not sent as evidence
       for unrelated small companies.
    """
    cache = read_json(settings.search_cache_path, default={})

    queries, official_domains, direct_results = _build_queries(
        company_name=company_name,
        country=country,
        existing_web=existing_web,
        emails=emails,
        contacts=contacts,
    )

    all_results: list[SearchResult] = []
    all_results.extend(direct_results)

    for query in queries:
        cache_key = f"v2::{query}"
        if use_cache and cache_key in cache:
            raw_results = cache[cache_key]
        else:
            try:
                raw_results = _ddgs_text_search(query, settings.max_search_results)
                cache[cache_key] = raw_results
                write_json(settings.search_cache_path, cache)
                time.sleep(0.5)
            except Exception as exc:
                print(f"[WARN] Search failed for query {query!r}: {exc}")
                continue

        for item in raw_results:
            url = item.get("href") or item.get("url") or ""
            if not url:
                continue
            result = SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("body") or item.get("snippet") or "",
                source_type="search",
            )
            result.relevance_score = _score_result(company_name, result, official_domains)
            all_results.append(result)

    # Score direct results too, then dedupe.
    rescored: list[SearchResult] = []
    for result in _dedupe_results(all_results):
        result.relevance_score = max(
            result.relevance_score,
            _score_result(company_name, result, official_domains),
        )
        rescored.append(result)

    # Keep strong direct candidates and results that actually mention the target.
    filtered = [
        result
        for result in rescored
        if result.source_type in {"legacy_url", "email_domain"} or result.relevance_score >= 2.5
    ]

    # Sort by quality. Direct workbook URLs and exact company matches float to the top.
    filtered.sort(key=lambda item: item.relevance_score, reverse=True)

    # Do not flood the scraper/LLM. Usually 5-8 good URLs are better than 20 noisy ones.
    return filtered[: max(settings.max_search_results, 8)]
