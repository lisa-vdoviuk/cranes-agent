from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings
from src.constants import IMAGE_RELEVANCE_TERMS
from src.parked_domain_detector import DomainHealthResult, check_domain_health
from src.schemas import ScrapedPage, SearchResult
from src.site_verifier import classify_url_category, filter_results_for_scraping
from src.utils import safe_hash, truncate_text


BLOCKED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "xing.com",
    "youtube.com",
    "youtu.be",
}

# Image extensions we are willing to download for vision analysis.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

def _is_probably_html_url(url: str) -> bool:
    lowered = url.lower().split("?", 1)[0]
    non_html = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".mp3", ".webp", ".svg",
    )
    return not lowered.endswith(non_html)


def _domain_is_blocked(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def _normalize_url(url: str) -> str:
    url = str(url or "").strip()
    if not url:
        return ""
    if not re.match(r"https?://", url, flags=re.I):
        url = "https://" + url
    return url


def _root_url(url: str) -> str:
    parsed = urlparse(_normalize_url(url))
    if not parsed.netloc:
        return ""
    return urlunparse((parsed.scheme, parsed.netloc, "/", "", "", ""))


def _url_variants(url: str) -> list[str]:
    """Try the provided URL, then https/http and root fallbacks for old workbook URLs."""
    normalized = _normalize_url(url)
    if not normalized:
        return []

    parsed = urlparse(normalized)
    variants = [normalized]

    if parsed.scheme == "http":
        variants.append(urlunparse(("https", parsed.netloc, parsed.path or "/", "", parsed.query, "")))
    elif parsed.scheme == "https":
        variants.append(urlunparse(("http", parsed.netloc, parsed.path or "/", "", parsed.query, "")))

    root = _root_url(normalized)
    if root:
        variants.append(root)
        root_parsed = urlparse(root)
        alt_scheme = "https" if root_parsed.scheme == "http" else "http"
        variants.append(urlunparse((alt_scheme, root_parsed.netloc, "/", "", "", "")))

    output: list[str] = []
    for item in variants:
        if item and item not in output:
            output.append(item)
    return output


def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (compatible; CraneCRMResearchBot/4.0; +https://example.local)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "de,en;q=0.9,es;q=0.7",
    }


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
)
def _fetch_url(url: str, timeout: int) -> tuple[str, str]:
    response = requests.get(
        url,
        headers=_request_headers(),
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type}")
    return response.text, str(response.url)


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return re.sub(r"\s+", " ", soup.title.string).strip()
    h1 = soup.find("h1")
    if h1:
        return re.sub(r"\s+", " ", h1.get_text(" ", strip=True)).strip()
    return ""


def _extract_clean_text(html: str, url: str) -> str:
    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if extracted:
        return extracted

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# ---------------------------------------------------------------------------
# Crane image candidate extraction
# ---------------------------------------------------------------------------

def _srcset_candidates(srcset: str) -> list[str]:
    urls: list[str] = []
    for part in str(srcset or "").split(","):
        candidate = part.strip().split(" ", 1)[0].strip()
        if candidate:
            urls.append(candidate)
    return urls


def _background_image_urls(style: str) -> list[str]:
    return [
        m.group(1).strip("'\"")
        for m in re.finditer(r"url\(([^)]+)\)", str(style or ""), flags=re.I)
    ]


def _looks_like_crane_image(src: str, alt: str, title: str, context: str = "") -> bool:
    """Return True only when the image metadata strongly suggests an actual crane photograph."""
    haystack = f"{src} {alt} {title} {context}".lower()
    return any(term in haystack for term in IMAGE_RELEVANCE_TERMS)


def _is_image_url(url: str) -> bool:
    path = urlparse(url).path.lower().split("?", 1)[0]
    return any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS)


def find_crane_image_candidates(html: str, base_url: str, max_candidates: int = 8) -> list[str]:
    """
    Scan the HTML of a scraped page and return absolute URLs of images that
    are likely to show physical crane equipment.

    Strategy (in priority order):
    1. OpenGraph / Twitter card meta images — often the main hero or fleet photo.
    2. <img> tags whose src/alt/title/parent text contains crane-related terms.
    3. CSS background-image rules in elements with crane-related surrounding text.

    Only actual image-file URLs are returned (jpg/jpeg/png/webp/gif).
    SVG, data URIs, and non-image URLs are excluded.
    """
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    # 1. OpenGraph / Twitter card — highest priority, these are curated hero images.
    for meta in soup.find_all("meta"):
        prop = str(meta.get("property") or meta.get("name") or "").lower()
        if prop in {"og:image", "twitter:image", "twitter:image:src"}:
            src = str(meta.get("content") or "").strip()
            if src:
                abs_url = urljoin(base_url, src)
                if _is_image_url(abs_url):
                    candidates.append(abs_url)

    # 2. <img> tags with crane-relevant metadata.
    for img in soup.find_all("img"):
        alt = str(img.get("alt") or "").strip()
        title_attr = str(img.get("title") or "").strip()
        parent_text = img.parent.get_text(" ", strip=True)[:200] if img.parent else ""

        src_attrs = [
            str(img.get(name) or "").strip()
            for name in ("src", "data-src", "data-original", "data-lazy-src", "data-ll-src")
        ]
        src_attrs.extend(_srcset_candidates(str(img.get("srcset") or "")))
        src_attrs.extend(_srcset_candidates(str(img.get("data-srcset") or "")))

        for src in src_attrs:
            if not src:
                continue
            if not _looks_like_crane_image(src, alt, title_attr, parent_text):
                continue
            abs_url = urljoin(base_url, src)
            lowered = abs_url.lower()
            # Skip SVG icons, data URIs, and tiny icon/logo filenames.
            if lowered.startswith("data:") or "svg" in lowered:
                continue
            if any(skip in lowered for skip in ("logo", "icon", "sprite", "banner", "header", "footer")):
                continue
            if _is_image_url(abs_url):
                candidates.append(abs_url)

    # 3. CSS background images in elements with crane-related text.
    for tag in soup.find_all(style=True):
        style = str(tag.get("style") or "")
        context = tag.get_text(" ", strip=True)[:200]
        for src in _background_image_urls(style):
            if not _looks_like_crane_image(src, "", "", context):
                continue
            abs_url = urljoin(base_url, src)
            if _is_image_url(abs_url):
                candidates.append(abs_url)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    output: list[str] = []
    for url in candidates:
        key = url.strip().rstrip("/").lower()
        if key and key not in seen:
            seen.add(key)
            output.append(url)
        if len(output) >= max_candidates:
            break

    return output


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------

def _image_cache_path(url: str, settings: Settings) -> Path:
    return settings.image_cache_dir / f"{safe_hash(url)}.jpg"


def download_crane_image(url: str, settings: Settings) -> Path | None:
    """
    Download a crane image to the local image cache and return its path.
    Returns None on any error or if the file already exists (cache hit).
    """
    cache_path = _image_cache_path(url, settings)
    if cache_path.exists():
        return cache_path  # already cached

    settings.image_cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(
            url,
            headers={
                **_request_headers(),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
            timeout=settings.vision_image_timeout_seconds,
            stream=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if content_type and "image" not in content_type:
            return None

        content = response.content
        if len(content) > settings.vision_max_image_bytes:
            print(f"    [IMG] Skipping oversized image ({len(content)//1024} KB): {url}")
            return None
        if len(content) < 5_000:
            # Likely a 1×1 tracker pixel or stub.
            return None

        cache_path.write_bytes(content)
        return cache_path

    except Exception as exc:
        print(f"    [IMG] Could not download {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page cache — JSON format (replaces the old text + delimiter approach)
# ---------------------------------------------------------------------------

def _page_cache_path(url: str, settings: Settings) -> Path:
    return settings.page_cache_dir / f"{safe_hash(url)}.json"


def _load_page_cache(url: str, settings: Settings) -> dict | None:
    path = _page_cache_path(url, settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_page_cache(url: str, settings: Settings, data: dict) -> None:
    settings.page_cache_dir.mkdir(parents=True, exist_ok=True)
    path = _page_cache_path(url, settings)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        print(f"    [CACHE] Could not write page cache for {url}: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_search_results(
    search_results: list[SearchResult],
    settings: Settings,
    use_cache: bool = True,
) -> list[ScrapedPage]:
    """
    Scrape up to settings.max_pages_to_scrape pages from the ranked search results.
    For each page, extract text content AND collect URLs of crane images found on the
    page. The image URLs are stored in ScrapedPage.crane_image_urls so the vision
    model can analyse them later (in color_inference.py).

    Pages whose domain appears to be parked, expired, or for sale are silently
    skipped so they cannot pollute the official-site resolver or the LLM prompt.
    """
    search_results = filter_results_for_scraping(search_results)
    pages: list[ScrapedPage] = []
    seen_urls: set[str] = set()

    for result in search_results:
        if len(pages) >= settings.max_pages_to_scrape:
            break

        for url in _url_variants(result.url):
            if len(pages) >= settings.max_pages_to_scrape:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)

            if not _is_probably_html_url(url) or _domain_is_blocked(url):
                continue

            # ---- Try cache first ----
            if use_cache:
                cached = _load_page_cache(url, settings)
                if cached and len(cached.get("text", "").strip()) >= settings.min_page_text_chars:
                    # Respect previously detected parked status stored in cache.
                    cached_health = cached.get("domain_health", "ok")
                    if cached_health.startswith("parked:"):
                        print(f"  [PARKED] Skipping cached parked domain {url}: {cached_health}")
                        break
                    pages.append(
                        ScrapedPage(
                            url=url,
                            title=cached.get("title", result.title),
                            text=truncate_text(cached["text"], settings.max_page_text_chars),
                            crane_image_urls=cached.get("crane_image_urls", []),
                            domain_health=cached_health,
                        )
                    )
                    break

            # ---- Live fetch ----
            try:
                html, final_url = _fetch_url(url, settings.request_timeout_seconds)
            except Exception as exc:
                print(f"  [WARN] Could not fetch {url}: {exc}")
                continue

            title = _extract_title(html) or result.title
            text = truncate_text(
                _extract_clean_text(html, url=final_url),
                settings.max_page_text_chars,
            )

            # ---- Parked-domain check (runs before min-text filtering) ----
            # Parking / unpaid-domain pages often have short text but HTTP 200.
            # If we check only after min_page_text_chars, those pages are skipped
            # without being marked as invalid, and the resolver may later accept
            # the raw search/direct URL as an official website.
            health: DomainHealthResult = check_domain_health(final_url, html, text)
            if health.is_parked:
                health_tag = f"parked:{health.detection_method}:{health.evidence}"
                print(
                    f"  [PARKED] Skipping {url} → {final_url}: "
                    f"{health.detection_method} matched '{health.evidence}'"
                )
                parked_cache = {
                    "title": title,
                    "text": text,
                    "crane_image_urls": [],
                    "domain_health": health_tag,
                }
                # Cache both the requested URL and final redirected URL so future
                # runs know the original candidate is invalid too.
                _save_page_cache(url, settings, parked_cache)
                if final_url != url:
                    _save_page_cache(final_url, settings, parked_cache)
                break   # Don't try other URL variants for the same result

            if len(text) < settings.min_page_text_chars:
                continue

            crane_image_urls = find_crane_image_candidates(html, base_url=final_url)

            _save_page_cache(final_url, settings, {
                "title": title,
                "text": text,
                "crane_image_urls": crane_image_urls,
                "domain_health": "ok",
            })

            pages.append(
                ScrapedPage(
                    url=final_url,
                    title=title,
                    text=text,
                    crane_image_urls=crane_image_urls,
                    domain_health="ok",
                )
            )
            break

    return pages
