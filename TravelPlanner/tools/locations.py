import requests
from langchain_core.tools import tool
from typing import Annotated, Dict, List

from models.factory import get_google_maps_api_key

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

_CONTACT = {
    "User-Agent": "travel-planner-robot/1.0",
    "Accept": "application/json",
}


def _component(components: list, wanted: str) -> str:
    for item in components or []:
        if wanted in item.get("types", []):
            return item.get("long_name", "")
    return ""


@tool
def get_location_coordinate(
    location: Annotated[str, "the place name to look up, e.g. \"Senso-ji Temple\""],
    city: Annotated[str, "the city or region the place is in, used to disambiguate names. Leave empty if unknown."] = "",
) -> List[Dict[str, str]]:
    """Location acquisition tool. Gets the latitude and longitude of a place from its
name, optionally narrowed down by city. Because many places share the same name, it
returns a list of every match, each with its full formatted address, coordinates,
city, country and Google place id, so the correct one can be picked.
Coordinates are returned as "longitude,latitude", which is the format expected by
the route planning and nearby search tools."""
    api_key = get_google_maps_api_key()

    address = f"{location}, {city}" if city else location
    params = {
        "address": address,
        "key": api_key,
        "language": "en",
        "region": "us",
    }
    response = requests.get(GEOCODE_URL, params=params, headers=_CONTACT, timeout=30)
    result = response.json()
    print("geocode result status:", result.get("status"))

    status = result.get("status")
    if status == "ZERO_RESULTS":
        return (
            f"No location found for \"{address}\". Try a more specific place name, "
            f"or add the city."
        )
    if status != "OK":
        raise ValueError(
            f"Failed to get the coordinate of \"{address}\". Google Geocoding API "
            f"returned status={status} ({result.get('error_message', 'no message')}). "
            f"Make sure the Geocoding API is enabled for your key."
        )

    coordinates = []
    for item in result.get("results", []):
        geo = item.get("geometry", {}).get("location", {})
        components = item.get("address_components", [])
        coordinates.append({
            "name": item.get("formatted_address", ""),
            "address": item.get("formatted_address", ""),
            "coordinate": f"{geo.get('lng')},{geo.get('lat')}",
            "city": _component(components, "locality")
                    or _component(components, "administrative_area_level_1"),
            "country": _component(components, "country"),
            "place_id": item.get("place_id", ""),
        })
    return coordinates


# test the tool
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    print(get_location_coordinate.args_schema.model_json_schema())
    a = get_location_coordinate.invoke({"location": "Senso-ji Temple", "city": "Tokyo"})
    print(a)
