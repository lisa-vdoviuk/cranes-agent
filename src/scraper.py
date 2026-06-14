from __future__ import annotations

import math
import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings
from src.schemas import ScrapedPage, SearchResult
from src.utils import safe_hash, truncate_text

try:  # Pillow is optional but recommended for v3 color hints.
    from PIL import Image
except Exception:  # pragma: no cover - depends on local environment
    Image = None  # type: ignore[assignment]


BLOCKED_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "xing.com",
    "youtube.com",
    "youtu.be",
}

IMAGE_RELEVANCE_TERMS = [
    "kran",
    "crane",
    "mobilkran",
    "autokran",
    "mobile-crane",
    "mobile_crane",
    "crawler",
    "teleskop",
    "liebherr",
    "tadano",
    "demag",
    "grove",
    "faun",
    "fleet",
    "fuhrpark",
    "vermietung",
]

COLOR_PALETTE = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "red": (220, 20, 30),
    "orange": (245, 130, 30),
    "yellow": (245, 210, 30),
    "green": (40, 150, 70),
    "blue": (40, 90, 200),
    "navy": (20, 35, 100),
    "purple": (140, 60, 160),
    "brown": (120, 80, 40),
}


class _FetchedPage(tuple):
    __slots__ = ()

    @property
    def html(self) -> str:
        return self[0]

    @property
    def final_url(self) -> str:
        return self[1]


# ----------------------------
# URL / fetch helpers
# ----------------------------


def _is_probably_html_url(url: str) -> bool:
    lowered = url.lower().split("?", 1)[0]
    blocked_extensions = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".doc", ".docx",
        ".xls", ".xlsx", ".ppt", ".pptx", ".mp4", ".mp3", ".webp", ".svg",
    )
    return not lowered.endswith(blocked_extensions)


def _domain_is_blocked(url: str) -> bool:
    domain = urlparse(url).netloc.lower()
    return any(blocked in domain for blocked in BLOCKED_DOMAINS)


def _cache_path_for_url(url: str, settings: Settings) -> Path:
    return settings.page_cache_dir / f"{safe_hash(url)}.txt"


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
        variants.append(urlunparse(("https" if root_parsed.scheme == "http" else "http", root_parsed.netloc, "/", "", "", "")))

    output: list[str] = []
    for item in variants:
        if item and item not in output:
            output.append(item)
    return output


def _request_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (compatible; CraneCRMResearchBot/3.0; +https://example.local)"
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

    return response.text, response.url


# ----------------------------
# Text extraction
# ----------------------------


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


# ----------------------------
# Image color hints
# ----------------------------


def _looks_like_relevant_image(src: str, alt: str, title: str) -> bool:
    haystack = f"{src} {alt} {title}".lower()
    return any(term in haystack for term in IMAGE_RELEVANCE_TERMS)


def _image_candidates(html: str, base_url: str, max_candidates: int = 6) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[tuple[str, str]] = []

    # OpenGraph images are often the main hero/fleet image.
    for meta in soup.find_all("meta"):
        prop = str(meta.get("property") or meta.get("name") or "").lower()
        if prop in {"og:image", "twitter:image"}:
            src = str(meta.get("content") or "").strip()
            if src:
                candidates.append((urljoin(base_url, src), "open graph image"))

    for img in soup.find_all("img"):
        src = str(img.get("src") or img.get("data-src") or img.get("data-original") or "").strip()
        if not src:
            continue
        alt = str(img.get("alt") or "").strip()
        title = str(img.get("title") or "").strip()
        if not _looks_like_relevant_image(src, alt, title):
            continue
        label = " ".join(part for part in [alt, title] if part).strip() or "crane-related image candidate"
        candidates.append((urljoin(base_url, src), label))

    output: list[tuple[str, str]] = []
    seen: set[str] = set()
    for url, label in candidates:
        lowered = url.lower().split("?", 1)[0]
        if not lowered.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        if url in seen:
            continue
        seen.add(url)
        output.append((url, label))
        if len(output) >= max_candidates:
            break

    return output


def _closest_color_name(rgb: tuple[int, int, int]) -> str:
    best_name = "unknown"
    best_distance = math.inf
    for name, ref in COLOR_PALETTE.items():
        distance = sum((rgb[i] - ref[i]) ** 2 for i in range(3))
        if distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def _dominant_colors_from_image_bytes(content: bytes, max_colors: int = 4) -> list[str]:
    if Image is None:
        return []

    with Image.open(BytesIO(content)) as img:
        img = img.convert("RGB")
        img.thumbnail((120, 120))
        # Adaptive palette makes this fast and robust enough for CRM color hints.
        quantized = img.quantize(colors=8, method=Image.Quantize.MEDIANCUT)
        palette = quantized.getpalette() or []
        counts = quantized.getcolors(maxcolors=120 * 120) or []

    colors: list[tuple[int, tuple[int, int, int]]] = []
    for count, palette_index in counts:
        offset = palette_index * 3
        if offset + 2 >= len(palette):
            continue
        rgb = (palette[offset], palette[offset + 1], palette[offset + 2])
        # Skip nearly transparent-looking / ultra-light backgrounds only when there are enough alternatives.
        colors.append((count, rgb))

    colors.sort(key=lambda item: item[0], reverse=True)

    output: list[str] = []
    seen_names: set[str] = set()
    for _, rgb in colors:
        name = _closest_color_name(rgb)
        if name in seen_names and len(seen_names) >= 2:
            continue
        seen_names.add(name)
        output.append(f"{name} {_rgb_to_hex(rgb)}")
        if len(output) >= max_colors:
            break

    return output


def _download_image(url: str, timeout: int) -> bytes:
    response = requests.get(
        url,
        headers={**_request_headers(), "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"},
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "image" not in content_type:
        raise ValueError(f"Unsupported image content type: {content_type}")

    content = response.content
    if len(content) > 5_000_000:
        raise ValueError("Image too large for quick color analysis")
    return content


def _extract_image_color_hints(html: str, base_url: str, timeout: int, max_hints: int = 3) -> list[str]:
    if Image is None:
        return []

    hints: list[str] = []
    for image_url, label in _image_candidates(html, base_url):
        if len(hints) >= max_hints:
            break
        try:
            colors = _dominant_colors_from_image_bytes(_download_image(image_url, timeout))
        except Exception:
            continue
        if not colors:
            continue
        hints.append(
            f"Image hint: {label}; image_url={image_url}; dominant_colors={', '.join(colors)}"
        )
    return hints


# ----------------------------
# Public API
# ----------------------------


def _read_cached_text_and_hints(cache_path: Path) -> tuple[str, list[str]]:
    cached = cache_path.read_text(encoding="utf-8")
    if "\n\n--- IMAGE COLOR HINTS ---\n" not in cached:
        return cached, []
    text, hints_blob = cached.split("\n\n--- IMAGE COLOR HINTS ---\n", 1)
    hints = [line.strip("- ").strip() for line in hints_blob.splitlines() if line.strip()]
    return text, hints


def _write_cached_text_and_hints(cache_path: Path, text: str, hints: list[str]) -> None:
    payload = text
    if hints:
        payload += "\n\n--- IMAGE COLOR HINTS ---\n"
        payload += "\n".join(f"- {hint}" for hint in hints)
    cache_path.write_text(payload, encoding="utf-8")


def scrape_search_results(
    search_results: list[SearchResult],
    settings: Settings,
    use_cache: bool = True,
) -> list[ScrapedPage]:
    pages: list[ScrapedPage] = []
    seen_urls: set[str] = set()

    # Search results are already quality-sorted by search.py.
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

            cache_path = _cache_path_for_url(url, settings)

            try:
                if use_cache and cache_path.exists():
                    cached_text, cached_hints = _read_cached_text_and_hints(cache_path)
                    if len(cached_text.strip()) >= settings.min_page_text_chars:
                        pages.append(
                            ScrapedPage(
                                url=url,
                                title=result.title,
                                text=truncate_text(cached_text, settings.max_page_text_chars),
                                image_color_hints=cached_hints,
                            )
                        )
                        break
                    continue

                html, final_url = _fetch_url(url, settings.request_timeout_seconds)
                title = _extract_title(html) or result.title
                text = truncate_text(_extract_clean_text(html, url=final_url), settings.max_page_text_chars)

                if len(text) < settings.min_page_text_chars:
                    continue

                color_hints = _extract_image_color_hints(
                    html=html,
                    base_url=final_url,
                    timeout=settings.request_timeout_seconds,
                )

                _write_cached_text_and_hints(cache_path, text, color_hints)

                pages.append(
                    ScrapedPage(
                        url=final_url,
                        title=title,
                        text=text,
                        image_color_hints=color_hints,
                    )
                )
                break

            except Exception as exc:
                print(f"[WARN] Could not scrape {url}: {exc}")

    return pages
