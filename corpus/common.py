"""
Shared utilities for the corpus harvest + build pipeline.
Every stage script (harvest/0X_*.py and root 0X_*.py) imports from here so
crawling etiquette (UA rotation, robots.txt, rate limiting) is consistent
and defined in exactly one place.
"""
from __future__ import annotations

import random
import re
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_DIR = Path(__file__).resolve().parent
HARVEST_DIR = CORPUS_DIR / "harvest"
RAW_DIR = CORPUS_DIR / "raw"
CLEAN_DIR = CORPUS_DIR / "clean"
LOGS_DIR = CORPUS_DIR / "logs"
for _d in (RAW_DIR, CLEAN_DIR, LOGS_DIR, HARVEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load corpus/.env (CORE_API_KEY, CONTACT_EMAIL, SEMANTIC_SCHOLAR_API_KEY,
# etc.) once here so every script that does `import common` picks these up
# automatically -- no per-script dotenv wiring needed. Every script imports
# common before reading any of these env vars, so this always runs first.
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=CORPUS_DIR / ".env")
except ImportError:
    print("[common] python-dotenv not installed -- .env will not be auto-loaded "
          "(env vars set directly in the shell still work). `pip install python-dotenv` to fix.")

# ---------------------------------------------------------------------------
# Target taxonomy (shared across every stage)
# ---------------------------------------------------------------------------
CROPS = [
    "maize", "cassava", "rice", "cowpea", "yam", "tomato",
    "sorghum", "groundnut", "pepper", "soybean",
]

ZONES = [
    "Sudan Savanna", "Northern Guinea Savanna", "Southern Guinea Savanna",
    "Derived Savanna", "Forest Zone", "Mangrove/Coastal",
]

# Sub-Saharan Africa coverage. Nigeria stays first and remains the primary
# market -- ZONES above is Nigeria-specific agro-ecological detail that
# doesn't apply to the other nine, which is why the zone matrix in
# 03_gap_analysis.py is kept as its own separate, Nigeria-only section
# rather than folded into the crop x country matrix.
COUNTRIES = [
    "Nigeria", "Kenya", "Ghana", "Tanzania", "Uganda", "Ethiopia",
    "Rwanda", "Senegal", "Mali", "Burkina Faso",
]

SOURCES_FIELDS = [
    "source_id", "url", "title", "publisher", "year", "licence",
    "crops_covered", "has_tables", "file_path", "retrieval_date",
]

SQL_TABLES = [
    "crop_calendar", "spacing", "fertilizer_rate",
    "agrochemical", "variety", "pest",
]

SQL_TABLE_KEYWORDS = {
    "crop_calendar": ["planting date", "planting window", "first rains", "sowing date", "cropping calendar"],
    "spacing": ["spacing", "plants per hectare", "row spacing", "plant population"],
    "fertilizer_rate": ["fertiliser", "fertilizer", "npk", "basal dressing", "top dress", "urea"],
    "agrochemical": ["pesticide", "insecticide", "fungicide", "herbicide", "agrochemical", "pre-harvest interval"],
    "variety": ["variety", "cultivar", "released variety", "maturity days"],
    "pest": ["pest", "symptom", "disease", "cultural control", "chemical control", "infestation"],
}

# ---------------------------------------------------------------------------
# Networking etiquette
# ---------------------------------------------------------------------------
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
]

REQUEST_TIMEOUT = 10
REQUEST_DELAY = 2.0

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _robots_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"


def random_ua() -> str:
    return random.choice(USER_AGENTS)


def robots_allowed(url: str, user_agent: str = "*") -> bool:
    """Best-effort robots.txt check. On any failure to fetch/parse robots.txt,
    fail OPEN (allow) since absence of a robots.txt does not imply disallow.

    Fetches robots.txt with `requests` under an explicit timeout rather than
    calling RobotFileParser.read() (which does its own urllib.urlopen() with
    NO timeout at all) -- that call can hang indefinitely on hosts where a
    plain `requests`/curl GET succeeds in under a second, which would freeze
    the entire pipeline on the very first request to such a host."""
    parsed = urlparse(url)
    key = parsed.netloc
    rp = _robots_cache.get(key)
    if rp is None:
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = requests.get(
                _robots_url(url), timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": random_ua()},
            )
            if resp.status_code >= 400:
                rp = None  # no robots.txt (or unreadable) -> fail open
            else:
                rp.parse(resp.text.splitlines())
        except Exception:
            rp = None
        _robots_cache[key] = rp
    if rp is None:
        return True
    try:
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True


def polite_get(
    url: str,
    session: requests.Session | None = None,
    timeout: int = REQUEST_TIMEOUT,
    delay: float = REQUEST_DELAY,
    allow_redirects: bool = True,
    extra_headers: dict | None = None,
):
    """GET a URL respecting robots.txt, rotating UA, and rate limiting.

    Returns (response_or_None, status_note). status_note is one of:
    "ok", "robots_disallowed", "timeout", "connection_error", "http_error:<code>", "blocked:<code>"
    Never raises.
    """
    if not robots_allowed(url):
        return None, "robots_disallowed"

    headers = {"User-Agent": random_ua(), "Accept-Language": "en-US,en;q=0.9"}
    if extra_headers:
        headers.update(extra_headers)

    sess = session or requests
    try:
        time.sleep(delay)
        resp = sess.get(url, headers=headers, timeout=timeout, allow_redirects=allow_redirects)
        if resp.status_code in (403, 429, 503):
            return resp, f"blocked:{resp.status_code}"
        if resp.status_code >= 400:
            return resp, f"http_error:{resp.status_code}"
        return resp, "ok"
    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.ConnectionError:
        return None, "connection_error"
    except requests.exceptions.RequestException as exc:
        return None, f"request_error:{exc.__class__.__name__}"


def new_session() -> requests.Session:
    return requests.Session()


# ---------------------------------------------------------------------------
# Small text helpers used by several stages
# ---------------------------------------------------------------------------

def find_crops(text: str) -> list[str]:
    # Word-boundary matching matters: "rice" is a plain substring of
    # "price", which shows up constantly in extension text discussing
    # market prices, and a naive substring check would misattribute those
    # passages to rice.
    text_l = (text or "").lower()
    return [c for c in CROPS if re.search(rf"\b{re.escape(c)}\b", text_l)]


def find_zones(text: str) -> list[str]:
    text_l = (text or "").lower()
    return [z for z in ZONES if re.search(rf"\b{re.escape(z.lower())}\b", text_l)]


def find_country(text: str) -> str | None:
    """First COUNTRIES match in text, word-boundary matched (mirrors
    find_crops/find_zones). Used where a country needs to be inferred from
    free text rather than already being known from crawl context."""
    text_l = (text or "").lower()
    for country in COUNTRIES:
        if re.search(rf"\b{re.escape(country.lower())}\b", text_l):
            return country
    return None


def guess_sql_tables(text: str) -> list[str]:
    text_l = (text or "").lower()
    hits = []
    for table, keywords in SQL_TABLE_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            hits.append(table)
    return hits


def safe_filename(s: str, maxlen: int = 120) -> str:
    keep = "".join(c if c.isalnum() or c in " _-." else "_" for c in s)
    keep = "_".join(keep.split())
    return keep[:maxlen] or "untitled"


def progress_line(stage: str, i: int, total: int, label: str, ok: bool, reason: str = "") -> None:
    """One line per item, printed immediately after it finishes -- the
    caller must call this for every single item (no batching/sampling) so a
    long unattended run always shows visible, per-item liveness."""
    if ok:
        status = f"OK - {reason}" if reason else "OK"
    else:
        status = f"ERROR: {reason}"
    print(f"[{stage}] {i}/{total}: {label} ({status})")


def print_elapsed_checkpoint(stage: str, i: int, total: int, start_time: float, every: int = 50) -> None:
    if i % every == 0:
        print(f"[{stage}] --- {i}/{total} processed, elapsed {time.time() - start_time:.1f}s ---")
