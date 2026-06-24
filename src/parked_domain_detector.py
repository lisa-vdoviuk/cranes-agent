from __future__ import annotations

"""parked_domain_detector.py — Detect parked, expired, or for-sale domains.

A parked domain returns HTTP 200 with real content, so HTTP status codes alone
cannot catch it.  This module uses two complementary signals:

  1. Registrar-redirect detection  — the *final* URL (after all redirects) lands
     on a known domain-parking or domain-registrar host.
  2. Content fingerprinting         — the page title / first text characters
     contain well-known "domain for sale" phrases in English or German.

Both signals are cheap (no extra HTTP requests) because the scraper already
captures the final URL and the extracted text before calling this module.
"""

from dataclasses import dataclass
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# Parking / registrar hosts
# ---------------------------------------------------------------------------

#: Registrar or parking-service domains whose *netloc* we compare against
#: the final URL after redirects.  A match means the original domain is
#: parked / expired / for sale on that platform.
#:
#: German registrars are included (united-domains, IONOS, Strato, …) because
#: the majority of companies in this pipeline are German.
PARKING_REGISTRAR_DOMAINS: frozenset[str] = frozenset(
    {
        # ── Global parking / aftermarket ─────────────────────────────────
        "sedo.com",
        "sedo.de",               # German-language Sedo interface
        "afternic.com",
        "dan.com",
        "hugedomains.com",
        "squadhelp.com",
        "undeveloped.com",
        "brandbucket.com",
        "efty.com",
        "parkingcrew.net",
        "parklogic.com",
        "bodis.com",
        "above.com",
        "domainsponsor.com",
        "smartname.com",
        "domainnameshop.com",
        "uniregistry.com",
        "flippa.com",
        "domcop.com",
        "namedrive.com",
        "domain.com",
        "domainmarket.com",
        "buydomains.com",
        # ── GoDaddy (global + parking subsidiary) ────────────────────────
        "godaddy.com",
        "parked.godaddy.com",
        "domainpage.godaddy.com",
        # ── Namecheap ─────────────────────────────────────────────────────
        "namecheap.com",
        "parkingpage.namecheap.com",
        # ── Google / Squarespace registrar ───────────────────────────────
        "domains.google",
        "get.tech",
        # ── German / European registrars (unregistered landing pages) ────
        "united-domains.de",
        "united-domains.ag",
        "ionos.de",
        "ionos.com",
        "1und1.de",             # 1&1 / IONOS brand
        "strato.de",
        "strato.com",
        "hosteurope.de",
        "domainfactory.de",
        "df.eu",
        "inwx.de",
        "inwx.com",
        "nic.de",               # DENIC placeholder pages
        "key-systems.net",
        "regfisc.de",
        "checkdomain.de",
        "manitu.de",
        "all-inkl.com",
        "netcup.de",
        "domaingo.com",
        "webgo.de",
        "hetzner.de",           # Hetzner domain parking
        "domainprovider.de",
        "domainregistry.de",
        "domains.de",
        "domainshop.de",
        "domaindiscount24.com",
        "domrobot.com",
        "joker.com",
        "eurodns.com",
        "gandi.net",
        "rrpproxy.net",
        # ── Other European registrars ─────────────────────────────────────
        "nameisp.com",
        "123-reg.co.uk",
        "123reg.co.uk",
        "names.co.uk",
        "fasthosts.co.uk",
        "dynadot.com",
        "namesilo.com",
        "porkbun.com",
        "register.com",
        "networksolutions.com",
        "squarespace.domains",
        "one.com",
        "easyname.com",
        "ovh.com",
        "ovhcloud.com",
        "arsys.es",
        "siteground.com",
        "bluehost.com",
        "name.com",
        "hover.com",
        "sav.com",
        "epik.com",
        "nameshield.net",
        "regtons.com",
        "webnode.com",
        "wix.com",
    }
)


# ---------------------------------------------------------------------------
# Content fingerprints
# ---------------------------------------------------------------------------

#: Case-insensitive substrings that, when found in the page title or the
#: first ~600 characters of extracted text, strongly indicate a parked,
#: for-sale, or under-construction domain.
#:
#: English phrases come first; German phrases follow (§ DE).
PARKING_PHRASES: tuple[str, ...] = (
    # ── English ──────────────────────────────────────────────────────────
    "this domain is for sale",
    "domain is for sale",
    "domain for sale",
    "buy this domain",
    "purchase this domain",
    "make an offer for this domain",
    "make an offer on this domain",
    "this domain may be for sale",
    "domain may be for sale",
    "domain name for sale",
    "this web page is parked",
    "web page is parked",
    "parked domain",
    "parked page",
    "domain has expired",
    "this domain has expired",
    "domain expired",
    "domain registration expired",
    "renew this domain",
    "account has been suspended",
    "this account has been suspended",
    "website coming soon",
    "site coming soon",
    "coming soon",
    "under construction",
    "page under construction",
    "website under construction",
    "register this domain",
    "claim this domain",
    "this domain is available",
    "domain is available",
    "inquire about this domain",
    "interested in this domain",
    "hosted by",                    # "Hosted by GoDaddy" parking pages
    "this site is for sale",
    "site is for sale",
    # ── German (§ DE) ─────────────────────────────────────────────────────
    "domain kaufen",                # "buy domain"
    "diese domain kaufen",          # "buy this domain"
    "domain zu verkaufen",          # "domain for sale"
    "domain steht zum verkauf",     # "domain is for sale"
    "domain ist zu verkaufen",      # "domain is for sale"
    "zum verkauf stehende domain",  # "domain for sale"
    "domain erwerben",              # "acquire domain"
    "domain ersteigern",            # "bid on domain"
    "domain angebot machen",        # "make an offer for domain"
    "angebot für diese domain",     # "offer for this domain"
    "angebot abgeben",              # "submit an offer"
    "diese domain ist zum verkauf",
    "diese domain steht zum verkauf",
    "domain verfügbar",             # "domain available"
    "domain ist verfügbar",
    "diese domain ist verfügbar",
    "domain noch nicht registriert",  # "domain not yet registered"
    "noch nicht registriert",
    "domain nicht registriert",
    "domain abgelaufen",            # "domain expired"
    "domain ist abgelaufen",
    "domain-registrierung abgelaufen",
    "ablauf der domain",
    "domainregistrierung erneuern",  # "renew domain registration"
    "domain erneuern",              # "renew domain"
    "website im aufbau",            # "website under construction"
    "seite im aufbau",              # "page under construction"
    "website befindet sich im aufbau",
    "seite befindet sich im aufbau",
    "demnächst verfügbar",          # "coming soon"
    "in kürze verfügbar",           # "available shortly"
    "coming soon",                  # also appears on German-hosted pages
    "webseite folgt in kürze",      # "website coming soon"
    "unsere neue webseite",         # common on parked/placeholder pages
    "hier entsteht",                # "here will be" – classic German placeholder
    "hier entsteht eine neue webseite",
    "hier entsteht unsere",
    "diese seite ist noch",         # "this page is not yet …"
    "diese webseite befindet sich noch",
    "konto gesperrt",               # "account suspended"
    "account gesperrt",
    "webhosting-paket",             # registrar upsell pages
    "wählen sie ein paket",         # "choose a package"
    "jetzt domain registrieren",    # "register domain now" – registrar CTA
    "domain registrieren",          # "register domain"
    "günstiger webspace",           # cheap hosting upsell
    "webspace buchen",              # "book webspace"
    "jetzt bestellen",              # "order now" – registrar CTA on parked pages
    "hosting buchen",               # "book hosting"
    "diese domain gehört",          # "this domain belongs to" – registrar info
    "domain gehört zu",
    "parked bei",                   # "parked at" – German registrar phrasing
    "geparkt bei",                  # "parked at" (alternative)
    # ── Registrar payment / unpaid / default hosting pages ─────────────
    "domain payment",
    "pay for this domain",
    "pending renewal",
    "redemption period",
    "clienthold",
    "serverhold",
    "this domain has recently been registered",
    "this domain has been registered",
    "this domain was recently registered",
    "domain has been registered",
    "this domain is reserved",
    "domain reserved",
    "future home of",
    "web server's default page",
    "web server default page",
    "apache2 ubuntu default page",
    "apache2 debian default page",
    "nginx default page",
    "plesk default page",
    "parallels plesk panel",
    "site not published",
    "website not published",
    "this website is not yet configured",
    "this site is currently unavailable",
    "default website page",
    "default site page",
    "placeholder page",
    "there is no website configured at this address",
    "no website is currently present",
    "parking page",
    "temporary landing page",
    "diese domain wurde registriert",
    "domain wurde registriert",
    "diese domain ist reserviert",
    "domain ist reserviert",
    "domain reserviert",
    "diese internetpräsenz wurde noch nicht veröffentlicht",
    "diese internetpraesenz wurde noch nicht veroeffentlicht",
    "hier entsteht eine neue internetpräsenz",
    "hier entsteht eine neue internetpraesenz",
    "webseite wurde noch nicht veröffentlicht",
    "webseite wurde noch nicht veroeffentlicht",
    "webspace wurde erfolgreich eingerichtet",
    "hosting wurde eingerichtet",
    "standardseite",
    "standard-seite",
    "plesk obsidian",
    "diese seite wurde deaktiviert",
    "account wurde gesperrt",
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainHealthResult:
    """Outcome of a single parked-domain check."""

    is_parked: bool
    detection_method: str   # "registrar_redirect" | "content_phrase" | "title_phrase" | "none"
    evidence: str           # matched registrar domain or matched phrase


_HEALTHY = DomainHealthResult(is_parked=False, detection_method="none", evidence="")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _netloc(url: str) -> str:
    """Return the lowercase netloc (host) of *url*, stripping 'www.'."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return ""
    return host.removeprefix("www.")


def _is_parking_host(netloc: str) -> bool:
    """Return True if *netloc* exactly matches or is a subdomain of a known parking registrar."""
    if not netloc:
        return False
    if netloc in PARKING_REGISTRAR_DOMAINS:
        return True
    # Subdomain match: "parked.godaddy.com" matches "godaddy.com"
    for registrar in PARKING_REGISTRAR_DOMAINS:
        if netloc.endswith("." + registrar):
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_domain_health(
    final_url: str,
    html: str,
    extracted_text: str,
) -> DomainHealthResult:
    """Check whether a fetched page indicates a parked / expired / for-sale domain.

    Parameters
    ----------
    final_url:
        The *resolved* URL after HTTP redirects (``str(response.url)``).
        This is what catches registrar redirects even if the original URL
        looked legitimate.
    html:
        Raw HTML of the page (used to extract the ``<title>`` for phrase
        matching when ``extracted_text`` is sparse).
    extracted_text:
        Text already extracted by trafilatura / BeautifulSoup (available for
        free from the scraper).  Only the first 600 characters are inspected
        to keep the check fast and avoid false positives deep in page content.

    Returns
    -------
    DomainHealthResult
        ``is_parked=True`` with the detection method and matched evidence when
        a parking signal is found; ``is_parked=False`` otherwise.
    """

    # ── Signal 1: registrar redirect ──────────────────────────────────────
    host = _netloc(final_url)
    if _is_parking_host(host):
        return DomainHealthResult(
            is_parked=True,
            detection_method="registrar_redirect",
            evidence=host,
        )

    # ── Signal 2: title phrase matching ───────────────────────────────────
    # Pull title from raw HTML (cheap string search, no full parse needed).
    title_match = _extract_title_text(html)
    if title_match:
        hit = _phrase_match(title_match)
        if hit:
            return DomainHealthResult(
                is_parked=True,
                detection_method="title_phrase",
                evidence=hit,
            )

    # ── Signal 3: body-text phrase matching (first 600 chars) ─────────────
    body_sample = (extracted_text or "")[:600]
    hit = _phrase_match(body_sample)
    if hit:
        return DomainHealthResult(
            is_parked=True,
            detection_method="content_phrase",
            evidence=hit,
        )

    return _HEALTHY


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_title_text(html: str) -> str:
    """Extract raw text between <title>…</title> without a full HTML parse."""
    if not html:
        return ""
    lower = html.lower()
    start = lower.find("<title")
    if start == -1:
        return ""
    tag_end = lower.find(">", start)
    if tag_end == -1:
        return ""
    end = lower.find("</title>", tag_end)
    if end == -1:
        return ""
    return html[tag_end + 1 : end].strip()


def _phrase_match(text: str) -> str:
    """Return the first parking phrase found in *text* (case-insensitive), or ''."""
    lowered = text.lower()
    for phrase in PARKING_PHRASES:
        if phrase in lowered:
            return phrase
    return ""
