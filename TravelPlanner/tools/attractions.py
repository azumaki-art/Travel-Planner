"""Attraction search backed by Trip.com's travel guide.

Trip.com renders its travel-guide pages on the server, so a plain HTTP request
returns the full attraction list. That is used here instead of scraping a
JavaScript-rendered page with Selenium, which makes the tool much faster and
removes the Chrome/ChromeDriver requirement.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from tools.search_backend import search_text

TRIP_HOST = "https://www.trip.com"
LOCALE_QUERY = "locale=en-XX&curr=USD"

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# /travel-guide/destination/tokyo-294/  or  /travel-guide/attraction/tokyo-294/...
# The trailing slash is optional: search engines return both spellings.
_DESTINATION_RE = re.compile(r"/travel-guide/(?:destination|attraction)/([a-z0-9\-]+)-(\d+)(?=/|\?|\"|'|\s|$)")
_DETAIL_HREF_RE = re.compile(r"/travel-guide/attraction/([a-z0-9\-]+)/([a-z0-9\-]+)-(\d+)(?=/|\?|$)")
_DIRECT_ID_RE = re.compile(r"^([a-z0-9\-]+?)[\s\-_]*(\d+)$")

# city name -> resolved trip.com destination, so repeated lookups cost nothing
_DESTINATION_CACHE: Dict[str, Dict] = {}


def _clean(text: str) -> str:
    """Collapse whitespace and drop trip.com's private-use icon glyphs."""
    text = re.sub(r"[\ue000-\uf8ff]", " ", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _text_of(node) -> str:
    return _clean(node.get_text(" ", strip=True)) if node else ""


def _get_page(url: str):
    """Fetch a page and return both the parsed soup and the raw HTML."""
    response = requests.get(url, headers=_HEADERS, timeout=30)
    if response.status_code != 200:
        raise ValueError(
            f"Trip.com returned HTTP {response.status_code} for {url}. "
            f"The page may have moved, or the request was rate limited."
        )
    html = response.text
    return BeautifulSoup(html, "html.parser"), html


def _get_soup(url: str) -> BeautifulSoup:
    return _get_page(url)[0]


# --------------------------------------------------------------------------- #
# destination resolution
# --------------------------------------------------------------------------- #
def _resolve_destination(destination: str) -> Dict:
    """Map a destination name such as "Tokyo" onto its Trip.com guide page.

    Trip.com URLs are keyed by an internal numeric district id (Tokyo is 294),
    which is not derivable from the name. The id is looked up with a web search
    restricted to Trip.com's travel guide, and the result is cached.
    """
    raw = str(destination or "").strip()
    if not raw:
        raise ValueError("destination must not be empty.")

    cache_key = raw.lower()
    if cache_key in _DESTINATION_CACHE:
        return _DESTINATION_CACHE[cache_key]

    # Allow the caller to pass a slug-id pair ("tokyo-294") or a full guide URL.
    direct = _DESTINATION_RE.search(raw) or _DIRECT_ID_RE.match(raw.lower())
    if direct:
        slug, district_id = direct.group(1), direct.group(2)
        resolved = {
            "name": raw,
            "slug": slug,
            "district_id": district_id,
            "url": f"{TRIP_HOST}/travel-guide/destination/{slug}-{district_id}/?{LOCALE_QUERY}",
        }
        _DESTINATION_CACHE[cache_key] = resolved
        return resolved

    try:
        results = search_text(
            f"trip.com travel guide destination {raw}",
            max_results=10,
        )
    except Exception as exc:  # search backend unavailable
        raise ValueError(
            f"Could not look up \"{raw}\" on Trip.com because the web search "
            f"backend failed ({exc}). Retry, or pass the Trip.com destination id "
            f"directly in the form \"tokyo-294\"."
        ) from exc

    best = None
    for item in results or []:
        blob = " ".join(str(v) for v in item.values())
        for slug, district_id in _DESTINATION_RE.findall(blob):
            # Prefer the entry whose slug overlaps the requested name.
            if best is None:
                best = (slug, district_id)
            if raw.lower().replace(" ", "-") in slug:
                best = (slug, district_id)
                break
        if best and raw.lower().replace(" ", "-") in best[0]:
            break

    if not best:
        raise ValueError(
            f"Could not find \"{raw}\" in the Trip.com travel guide. Check the "
            f"spelling, or pass the Trip.com destination id directly in the form "
            f"\"tokyo-294\"."
        )

    slug, district_id = best
    resolved = {
        "name": raw,
        "slug": slug,
        "district_id": district_id,
        "url": f"{TRIP_HOST}/travel-guide/destination/{slug}-{district_id}/?{LOCALE_QUERY}",
    }
    _DESTINATION_CACHE[cache_key] = resolved
    return resolved


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _listing_url(dest: Dict) -> str:
    return (
        f"{TRIP_HOST}/travel-guide/attraction/{dest['slug']}-{dest['district_id']}"
        f"/tourist-attractions/?{LOCALE_QUERY}"
    )


def _parse_listing_card(link) -> Optional[Dict]:
    """Turn one attraction card of the list page into a dictionary."""
    href = link.get("href", "")
    match = _DETAIL_HREF_RE.search(href)
    if not match:
        return None

    city_slug, name_slug, poi_id = match.groups()

    name = _text_of(link.select_one("h2"))
    if not name:
        img = link.select_one("img")
        name = _clean(img.get("alt", "")) if img else ""
    if not name:
        name = name_slug.replace("-", " ").title()

    # "Free entry" / "From US$ 16.99" is rendered in one of these blocks; the
    # currency symbol sometimes lives in a separate element, hence the fallbacks.
    price = _text_of(link.select_one('[class*="Price_realPrice"]'))
    if not price:
        price = _text_of(link.select_one('[class*="Price_priceDesc"]'))
    if price:
        price = _normalize_price(price)
    else:
        price = _extract_price(_text_of(link))

    detail_url = href if href.startswith("http") else TRIP_HOST + href
    detail_url = detail_url.split("?")[0] + f"/?{LOCALE_QUERY}"

    return {
        "name": name,
        "poi_id": poi_id,
        "city_slug": city_slug,
        "award": _text_of(link.select_one('[class*="XRankView_popularText"]')),
        "tagline": _text_of(link.select_one('[class*="HotSpotView_hotSpot"]')),
        "district": _text_of(link.select_one('[class*="District_districtText"]')),
        "price": price,
        "url": detail_url,
    }


def _jsonld_nodes(soup: BeautifulSoup) -> List[Dict]:
    nodes = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except Exception:
            continue
        graph = data.get("@graph", [data]) if isinstance(data, dict) else data
        for node in graph if isinstance(graph, list) else [graph]:
            if isinstance(node, dict):
                nodes.append(node)
    return nodes


def _format_address(address) -> str:
    """Build a readable address, without repeating components already present."""
    if isinstance(address, str):
        return _clean(address)
    if not isinstance(address, dict):
        return ""
    parts: List[str] = []
    for value in (
        address.get("streetAddress"),
        address.get("addressLocality"),
        address.get("addressRegion"),
        address.get("postalCode"),
        address.get("addressCountry"),
    ):
        value = _clean(value)
        if not value:
            continue
        # e.g. a streetAddress that already ends with the country name
        if any(value.lower() in existing.lower() for existing in parts):
            continue
        parts.append(value)
    return ", ".join(parts)


def _extract_price(text: str) -> str:
    """Ticket price, covering "Free entry" and the various "From ..." spellings."""
    if "Free entry" in text:
        return "Free entry"
    found = re.search(r"From\s+(?:US)?\s*\$?\s*([\d][\d.,]*)", text)
    if found:
        return f"From US$ {found.group(1)}"
    found = re.search(r"US\$\s*([\d][\d.,]*)", text)
    if found:
        return f"From US$ {found.group(1)}"
    return "not available"


def _normalize_price(price: str) -> str:
    """Trip.com sometimes strips the currency symbol, e.g. "From 64"."""
    price = _clean(price)
    if not price:
        return ""
    if "free" in price.lower():
        return "Free entry"
    found = re.search(r"([\d][\d.,]*)", price)
    if found:
        return f"From US$ {found.group(1)}"
    return price


_JSON_STR = r'"((?:[^"\\]|\\.)*)"'


def _json_unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value


def _extract_embedded_poi(html: str) -> Dict:
    """Recover the POI record that Trip.com embeds in the page's script data.

    This is the most reliable source on the page: unlike the JSON-LD, it is present
    in both the full and the reduced page variants Trip.com serves. The record holds
    the rating, review count, suggested visit duration, address, district and
    coordinates, and sits immediately before the "rating"/"reviewCount" pair, which
    is used as the anchor.
    """
    anchor = re.search(r'"rating":\s*([\d.]+)\s*,\s*"reviewCount":\s*(\d+)', html)
    if not anchor:
        return {}

    # Scope the search to the record around the anchor rather than the whole page.
    window = html[max(0, anchor.start() - 4000): anchor.end() + 200]

    def last(pattern: str) -> str:
        matches = re.findall(pattern, window)
        return _json_unescape(matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1]) if matches else ""

    poi = {
        "rating": anchor.group(1),
        "review_count": anchor.group(2),
    }

    address = last(r'"address":\s*' + _JSON_STR)
    if address:
        poi["address"] = _clean(address)

    district = last(r'"districtName":\s*' + _JSON_STR)
    if district:
        poi["district"] = _clean(district)

    duration = last(r'"suggestedDuration":\s*' + _JSON_STR)
    if duration:
        poi["recommended_duration"] = _clean(duration)

    highlight = last(r'"recommendReason":\s*' + _JSON_STR)
    if highlight:
        poi["highlight"] = _clean(highlight)

    lat = last(r'"gglat":\s*(-?[\d.]+)')
    lng = last(r'"gglon":\s*(-?[\d.]+)')
    if lat and lng:
        poi["coordinate"] = f"{lng},{lat}"

    score = last(r'"hotScore":\s*' + _JSON_STR)
    if score:
        poi["trip_score"] = _clean(score)

    tag_block = re.findall(r'"tags":\s*\[(.*?)\]', window)
    if tag_block:
        tags = re.findall(r'"tagName":\s*' + _JSON_STR, tag_block[-1])
        tags = [_clean(_json_unescape(t)) for t in tags]
        tags = [t for t in tags if t]
        if tags:
            poi["category"] = ", ".join(tags)

    return poi


def _parse_attraction_detail(url: str) -> Dict:
    """Read one attraction page and pull out the planning-relevant facts.

    Values are taken from the embedded POI record and the page's JSON-LD where
    available (both stable), and from the labelled page text otherwise.
    """
    soup, html = _get_page(url)
    nodes = _jsonld_nodes(soup)

    for script in soup.find_all("script"):
        script.decompose()
    body = _clean(soup.get_text(" ", strip=True))

    detail = {"url": url}
    detail.update(_extract_embedded_poi(html))

    for node in nodes:
        node_type = node.get("@type")
        if not isinstance(node_type, str):
            continue
        # Deliberately no `name` here: the list page already provides it, and the
        # JSON-LD graph also contains an Organization node named "Trip.com".
        if node.get("address") and not detail.get("address"):
            detail["address"] = _format_address(node["address"])
        if node.get("telephone") and not detail.get("telephone"):
            detail["telephone"] = _clean(node["telephone"])
        if isinstance(node.get("geo"), dict) and not detail.get("coordinate"):
            geo = node["geo"]
            if geo.get("longitude") is not None and geo.get("latitude") is not None:
                detail["coordinate"] = f"{geo['longitude']},{geo['latitude']}"
        rating = node.get("aggregateRating")
        if isinstance(rating, dict) and rating.get("ratingValue") is not None:
            detail.setdefault("rating", str(rating["ratingValue"]))
            if rating.get("reviewCount") is not None:
                detail.setdefault("review_count", str(rating["reviewCount"]))
        hours = node.get("openingHoursSpecification")
        if hours and not detail.get("opening_hours"):
            detail["opening_hours"] = _format_opening_hours(hours)

    # --- labelled text fallbacks (these labels are stable in the page body) ---
    if not detail.get("opening_hours"):
        found = re.search(
            r"Open:\s*([0-9]{1,2}[:：][0-9]{2}\s*[–\-~—]\s*[0-9]{1,2}[:：][0-9]{2})", body
        )
        if found:
            detail["opening_hours"] = found.group(1)

    if not detail.get("recommended_duration"):
        found = re.search(
            r"Recommended sightseeing time:\s*([0-9]+\s*(?:[–\-~—]\s*[0-9]+\s*)?(?:hours?|minutes?|days?))",
            body,
        )
        if found:
            detail["recommended_duration"] = _clean(found.group(1))

    if not detail.get("address"):
        found = re.search(r"Address:\s*(.+?)\s*(?:Map|Phone|Highlights|FAQ|$)", body)
        if found:
            detail["address"] = _clean(found.group(1))

    if not detail.get("rating"):
        found = re.search(r"([0-9]\.[0-9])\s*/\s*5", body)
        if found:
            detail["rating"] = found.group(1)
    if not detail.get("review_count"):
        found = re.search(r"([\d,]+)\s+reviews", body)
        if found:
            detail["review_count"] = found.group(1)

    # Trip.com renders ticket prices on the destination list page, not on the
    # server-rendered detail page, so only override the price when one is found.
    price = _extract_price(body)
    if price != "not available":
        detail["price"] = price
        detail["free_entry"] = price == "Free entry"

    detail["open_now"] = bool(re.search(r"\bOpen\b(?!:)", body))

    # Category tags such as "Historic landmark" appear right before the status.
    if not detail.get("category"):
        found = re.search(
            r"of Best Things to Do in [^ ]+(?:\s+[A-Z][a-z]+)*\s+(.{0,80}?)\s+(?:Open|Closed)\b", body
        )
        if found:
            detail["category"] = _clean(found.group(1))

    found = re.search(r"Highlights:\s*(.+?)(?:Loading\.\.\.|FAQ|Recommended|$)", body)
    if found:
        summary = _clean(found.group(1))
        summary = re.sub(r"^Must-See Features and Attractions\s*", "", summary)
        summary = re.sub(r"Some information may have been translated by Google Translate\s*", "", summary)
        if summary:
            detail["summary"] = summary[:1200]

    return detail


def _format_opening_hours(hours) -> str:
    if isinstance(hours, str):
        return _clean(hours)
    entries = hours if isinstance(hours, list) else [hours]
    parts = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        opens, closes = entry.get("opens"), entry.get("closes")
        if opens and closes:
            parts.append(f"{opens}-{closes}")
        elif entry.get("description"):
            parts.append(_clean(entry["description"]))
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# tool
# --------------------------------------------------------------------------- #
@tool
def get_attractions_information(
    destination: Annotated[str, "destination name, must be a specific city or town, e.g. \"Tokyo\". A Trip.com slug-id such as \"tokyo-294\" is also accepted."],
    max_attractions: Annotated[int, "maximum number of attractions to return, between 1 and 10."] = 10,
    include_details: Annotated[bool, "when true, also open each attraction page to collect opening hours, recommended visit duration and address. Slower but much more useful for planning."] = True,
) -> dict:
    """Attractions search tool, backed by the Trip.com travel guide. Returns an
overview of the destination plus a list of its top attractions. Each attraction
includes its name, a short description, its Trip.com page, rating and review count,
ticket price, the area it sits in, and - when include_details is true - its opening
hours, the recommended sightseeing time and its address."""
    dest = _resolve_destination(destination)
    limit = max(1, min(int(max_attractions), 10))

    listing_soup = _get_soup(_listing_url(dest))

    attractions: List[Dict] = []
    seen_ids = set()
    for link in listing_soup.select('a[href*="/travel-guide/attraction/"]'):
        card = _parse_listing_card(link)
        if not card or card["poi_id"] in seen_ids:
            continue
        seen_ids.add(card["poi_id"])
        attractions.append(card)
        if len(attractions) >= limit:
            break

    if not attractions:
        raise ValueError(
            f"No attractions were found for \"{destination}\" on Trip.com. The "
            f"destination may be too small, or the page layout may have changed."
        )

    # The destination's own guide page carries a meta description we can use as an overview.
    overview = ""
    try:
        dest_soup = _get_soup(dest["url"])
        meta = dest_soup.find("meta", attrs={"name": "description"})
        overview = _clean(meta.get("content", "")) if meta else ""
    except Exception as exc:
        print("destination overview unavailable:", exc)

    if include_details:
        urls = [a["url"] for a in attractions]
        with ThreadPoolExecutor(max_workers=4) as pool:
            details = list(pool.map(_safe_detail, urls))
        enriched = []
        for attraction, detail in zip(attractions, details):
            merged = {**attraction, **{k: v for k, v in detail.items() if v not in (None, "")}}
            # A short description is more useful than none: fall back to the list tagline.
            merged["description"] = merged.get("summary") or merged.get("tagline", "")
            merged.pop("summary", None)
            enriched.append(merged)
        attractions = enriched

    print("trip.com attractions:", len(attractions), "for", dest["slug"])

    return {
        "destination": {
            "name": dest["name"],
            "trip_slug": dest["slug"],
            "trip_district_id": dest["district_id"],
            "guide_url": dest["url"],
        },
        "overview": overview or "No destination overview available.",
        "attractions": attractions,
    }


def _safe_detail(url: str) -> Dict:
    """Fetch one attraction, never letting a single failure break the batch."""
    try:
        return _parse_attraction_detail(url)
    except Exception as exc:
        print(f"detail fetch failed for {url}: {exc}")
        return {}


# test the tool
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    print(get_attractions_information.args_schema.model_json_schema())
    result = get_attractions_information.invoke({"destination": "Tokyo", "max_attractions": 3})
    print(json.dumps(result, ensure_ascii=False, indent=2)[:4000])
