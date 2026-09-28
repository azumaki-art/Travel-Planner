import math
from typing import Annotated, Dict, List

import requests
from langchain_core.tools import tool

from models.factory import get_google_maps_api_key

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# Fields requested from Places API (New). Kept explicit because the API bills and
# validates against the field mask.
_FIELD_MASK = ",".join([
    "places.displayName",
    "places.formattedAddress",
    "places.shortFormattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.primaryTypeDisplayName",
    "places.types",
    "places.googleMapsUri",
    "places.currentOpeningHours.openNow",
])

_CONTACT = {
    "User-Agent": "travel-planner-robot/1.0",
    "Accept": "application/json",
}


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    """Straight-line distance in meters between two coordinates."""
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(round(2 * radius * math.asin(math.sqrt(a))))


def _parse_location(location: str) -> tuple:
    """Parse a "longitude,latitude" string into (latitude, longitude)."""
    try:
        lng_str, lat_str = str(location).replace(" ", "").split(",")
        return float(lat_str), float(lng_str)
    except Exception as exc:
        raise ValueError(
            f"Invalid center point \"{location}\". Expected \"longitude,latitude\", "
            f"for example \"139.7967,35.7148\"."
        ) from exc


@tool
def search_nearby_poi(
    location: Annotated[str, "center point coordinates, separated by \",\" with longitude first then latitude, e.g. 139.7967,35.7148"],
    poi_types: Annotated[str, "comma separated Google place types to search for. Use \"restaurant\" or \"cafe\" for dining, \"lodging\" for accommodation, or \"tourist_attraction\" for sights. Example: \"restaurant,cafe\". Leave empty when using keyword."] = "",
    keyword: Annotated[str, "a specific place name to look for, e.g. \"Starbucks\" or \"Hilton\". Use only when the user names a specific place, otherwise leave empty. At least one of keyword or poi_types must be provided."] = "",
    radius: Annotated[int, "search radius in meters, between 1 and 50000."] = 2000,
    max_results: Annotated[int, "maximum number of results to return, between 1 and 20."] = 10,
) -> Dict:
    """Nearby search tool. Finds restaurants, hotels, shops and other points of
interest around a center point, by place type and/or keyword. Returns each place's
name, address, category, rating, review count, price level, opening status, Google
Maps link and straight-line distance from the center point in meters."""
    api_key = get_google_maps_api_key()

    if not keyword and not poi_types:
        raise ValueError("At least one of keyword or poi_types must be provided.")

    center_lat, center_lng = _parse_location(location)
    radius = max(1, min(int(radius), 50000))
    max_results = max(1, min(int(max_results), 20))

    headers = {
        **_CONTACT,
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": _FIELD_MASK,
    }

    if keyword:
        # Free-text search (Places API "Text Search (New)"), biased to the center point.
        payload = {
            "textQuery": keyword,
            "maxResultCount": max_results,
            "languageCode": "en",
            "locationBias": {
                "circle": {
                    "center": {"latitude": center_lat, "longitude": center_lng},
                    "radius": float(radius),
                }
            },
        }
        url = TEXT_SEARCH_URL
        query_label = keyword
    else:
        included = [t.strip() for t in str(poi_types).split(",") if t.strip()]
        payload = {
            "includedTypes": included,
            "maxResultCount": max_results,
            "languageCode": "en",
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": center_lat, "longitude": center_lng},
                    "radius": float(radius),
                }
            },
        }
        url = NEARBY_URL
        query_label = ", ".join(included)

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to search for \"{query_label}\" near {location}. Google Places "
            f"API (New) returned HTTP {response.status_code}: {response.text[:300]}. "
            f"Make sure \"Places API (New)\" is enabled for your key."
        )

    places = response.json().get("places", [])
    print("nearby result count:", len(places))
    if not places:
        return (
            f"No results related to \"{query_label}\" were found near {location}. "
            f"Try expanding the search radius or changing the keyword or place types."
        )

    results = []
    for item in places:
        latlng = item.get("location", {})
        lat, lng = latlng.get("latitude"), latlng.get("longitude")
        distance = (
            _haversine_meters(center_lat, center_lng, lat, lng)
            if lat is not None and lng is not None else None
        )
        opening = item.get("currentOpeningHours", {})
        results.append({
            "name": item.get("displayName", {}).get("text", ""),
            "type": item.get("primaryTypeDisplayName", {}).get("text", "")
                    or ", ".join(item.get("types", [])[:2]),
            "address": item.get("shortFormattedAddress")
                       or item.get("formattedAddress", ""),
            "distance_meters": distance,
            "coordinate": f"{lng},{lat}" if lat is not None else "",
            "rating": item.get("rating", "not available"),
            "review_count": item.get("userRatingCount", "not available"),
            "price_level": item.get("priceLevel", "not available"),
            "open_now": opening.get("openNow", "not available"),
            "google_maps_url": item.get("googleMapsUri", ""),
        })

    results.sort(key=lambda r: r["distance_meters"] if r["distance_meters"] is not None else 10**9)
    return {
        "center_point": location,
        "search_radius_meters": radius,
        "search_result_count": len(results),
        "pois": results,
    }


# test the tool
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    print(search_nearby_poi.args_schema.model_json_schema())
    a = search_nearby_poi.invoke({
        "location": "139.7967,35.7148",
        "poi_types": "restaurant",
        "radius": 1000,
    })
    print(a)
