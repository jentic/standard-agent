import os
import dotenv
import httpx
from typing import Annotated
from .tools_base import FunctionToolProvider

dotenv.load_dotenv()

func_tools = FunctionToolProvider()

API_KEY = os.environ.get("OPENWEATHER_API_KEY")


@func_tools.tool(["weather", "temperature", "humidity", "forecast"])
async def get_weather(lat: float, lon: float, units: str = "metric") -> dict:
    """
    Retrieve real-time weather information for a specific geographic location.
    location can be found using get_coordinate_by_city or get_coordinate_by_zip tools
    """
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": API_KEY, "units": units}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    if resp.status_code != 200:
        return {"error": data.get("message", "Failed to fetch weather")}

    return {
        "city": data["name"],
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "feels_like": data["main"]["feels_like"],
        "weather": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@func_tools.tool(
    [
        "location",
        "city",
        "coordinates",
        "geolocation",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]
)
async def get_coordinate_by_city(
    city: Annotated[str, " city name in string"], test: bool = True
) -> dict:
    """Retrieve the geographic coordinates (latitude and longitude) for a given city."""
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "appid": API_KEY, "limit": 1}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()

    if resp.status_code != 200:
        return {"error": data.get("message", "Failed to fetch weather")}
    if len(data) > 0:
        return {
            # 'name': data[0]['name'],
            "lat": data[0]["lat"],
            "lon": data[0]["lon"],
            # 'country': data[0]['country']
        }
    return {"error": data.get("message", "information is not available")}


@func_tools.tool(
    [
        "location",
        "coordinates",
        "geolocation",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]
)
async def get_coordinate_by_zip(zip: str, country_code: str) -> dict:
    """Fetch the coordinate for a given Zip/post code and country code[ISO 3166]"""
    url = "http://api.openweathermap.org/geo/1.0/zip"
    params = {"zip": f"{zip},{country_code}", "appid": API_KEY}

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params)
        data = resp.json()
    if resp.status_code != 200:
        return {"error": data.get("message", "Failed to fetch weather")}
    if not data:
        return {"error": data.get("message", "information is not available")}
    return data
