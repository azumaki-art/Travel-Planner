from typing import Annotated, Dict

import requests
from langchain_core.tools import tool

from models.factory import get_google_maps_api_key

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Routes API (New) requires an explicit field mask.
_FIELD_MASK = ",".join([
    "routes.duration",
    "routes.distanceMeters",
    "routes.travelAdvisory.transitFare",
    "routes.legs.steps.travelMode",
    "routes.legs.steps.distanceMeters",
    "routes.legs.steps.staticDuration",
    "routes.legs.steps.navigationInstruction",
    "routes.legs.steps.transitDetails",
])

_CONTACT = {
    "User-Agent": "travel-planner-robot/1.0",
    "Accept": "application/json",
}

_VALID_MODES = {"TRANSIT", "WALK", "DRIVE", "BICYCLE"}


def _parse_latlng(value: str) -> Dict[str, float]:
    """Parse a "longitude,latitude" string into a Routes API LatLng object."""
    try:
        lng_str, lat_str = str(value).replace(" ", "").split(",")
        return {"latitude": float(lat_str), "longitude": float(lng_str)}
    except Exception as exc:
        raise ValueError(
            f"Invalid coordinate \"{value}\". Expected \"longitude,latitude\", "
            f"for example \"139.7967,35.7148\"."
        ) from exc


def _seconds(duration: str) -> int:
    """Routes API returns durations as strings such as \"1234s\"."""
    try:
        return int(str(duration).rstrip("s"))
    except (TypeError, ValueError):
        return 0


def _humanize(seconds: int) -> str:
    hours, minutes = divmod(seconds // 60, 60)
    if hours and minutes:
        return f"{hours} h {minutes} min"
    if hours:
        return f"{hours} h"
    return f"{minutes} min"


@tool
def route_planning(
    origin: Annotated[str, "longitude and latitude of the departure point, separated by \",\" with longitude first then latitude, e.g. 139.7967,35.7148"],
    destination: Annotated[str, "longitude and latitude of the destination, separated by \",\" with longitude first then latitude, e.g. 139.7967,35.7148"],
    travel_mode: Annotated[str, "how to travel: \"TRANSIT\" for public transport (train, bus, subway), \"WALK\", \"DRIVE\" or \"BICYCLE\". Defaults to TRANSIT."] = "TRANSIT",
) -> Dict:
    """Route planning tool. Plans how to get from one place to another and returns the
total distance in meters, the total travel time in seconds, and the turn by turn or
step by step breakdown including which transit lines to take. Use travel_mode
"TRANSIT" for public transport within a city and "DRIVE" for long intercity trips."""
    api_key = get_google_maps_api_key()

    mode = str(travel_mode or "TRANSIT").strip().upper()
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Invalid travel_mode \"{travel_mode}\". Use one of {sorted(_VALID_MODES)}."
        )

    headers = {
        **_CONTACT,
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": _FIELD_MASK,
    }
    payload = {
        "origin": {"location": {"latLng": _parse_latlng(origin)}},
        "destination": {"location": {"latLng": _parse_latlng(destination)}},
        "travelMode": mode,
        "languageCode": "en",
        "units": "METRIC",
        "computeAlternativeRoutes": False,
    }

    response = requests.post(ROUTES_URL, json=payload, headers=headers, timeout=30)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to obtain a route from {origin} to {destination} by {mode}. "
            f"Google Routes API returned HTTP {response.status_code}: "
            f"{response.text[:300]}. Make sure \"Routes API\" is enabled for your key."
        )

    routes = response.json().get("routes", [])
    print("route result count:", len(routes))
    if not routes:
        return (
            f"No route found from {origin} to {destination} by {mode}. Check that the "
            f"coordinates are correct, or try travel_mode \"DRIVE\"."
        )

    route = routes[0]
    distance_meters = route.get("distanceMeters", 0)
    duration_seconds = _seconds(route.get("duration"))

    transit_fare = route.get("travelAdvisory", {}).get("transitFare", {})
    if transit_fare:
        fare = f"{transit_fare.get('units', '0')} {transit_fare.get('currencyCode', '')}".strip()
    else:
        fare = "not available"

    steps = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            instruction = (step.get("navigationInstruction") or {}).get("instructions", "")
            step_mode = step.get("travelMode", "")
            entry = {
                "travel_mode": step_mode,
                "distance_meters": step.get("distanceMeters", 0),
                "duration_seconds": _seconds(step.get("staticDuration")),
                "instruction": instruction,
            }
            if step_mode == "TRANSIT" and step.get("transitDetails"):
                details = step["transitDetails"]
                line = details.get("transitLine", {})
                entry["transit_line"] = line.get("name", "")
                entry["vehicle_type"] = line.get("vehicle", {}).get("type", "")
                entry["headsign"] = details.get("headsign", "")
                entry["stop_count"] = details.get("stopCount", "")
                entry["departure_stop"] = (
                    details.get("stopDetails", {}).get("departureStop", {}).get("name", "")
                )
                entry["arrival_stop"] = (
                    details.get("stopDetails", {}).get("arrivalStop", {}).get("name", "")
                )
            steps.append(entry)

    result = {
        "origin": origin,
        "destination": destination,
        "travel_mode": mode,
        "distance_meters": distance_meters,
        "distance": f"{distance_meters / 1000:.2f} km",
        "duration_seconds": duration_seconds,
        "duration": _humanize(duration_seconds),
        "transit_fare": fare,
        "steps": steps,
    }
    print("route summary:", {k: result[k] for k in ("distance", "duration", "travel_mode")})
    return result


# test the tool
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    print(route_planning.args_schema.model_json_schema())
    a = route_planning.invoke({
        "origin": "139.7967,35.7148",
        "destination": "139.7454,35.6586",
        "travel_mode": "TRANSIT",
    })
    print(a)
