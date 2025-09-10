from __future__ import annotations

import os
import sys
from dotenv import load_dotenv, find_dotenv

from assignment_omni.clients.weather import OpenWeatherClient, WeatherQuery


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path or None)
    print(f"CWD: {os.getcwd()}")
    print(f".env detected: {env_path or 'not found'}")
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if api_key:
        masked = api_key[:3] + "***" + api_key[-3:] if len(api_key) >= 6 else "***"
        print(f"OPENWEATHER_API_KEY: present (len={len(api_key)}), preview={masked}")
    else:
        print("OPENWEATHER_API_KEY: NOT SET")
        print("SKIP: Add OPENWEATHER_API_KEY to .env to test client.")
        sys.exit(0)

    client = OpenWeatherClient(api_key)
    city = os.getenv("TEST_CITY") or "London"
    lang = os.getenv("TEST_LANG") or None
    lat = os.getenv("TEST_LAT")
    lon = os.getenv("TEST_LON")
    lat_f = float(lat) if lat else None
    lon_f = float(lon) if lon else None
    q = WeatherQuery(city=city if not (lat_f and lon_f) else None, lat=lat_f, lon=lon_f, lang=lang)
    print("Request params (no key):", {k: v for k, v in q.to_params().items() if k != "appid"})
    print("Fetching weather...")
    data = client.fetch_weather(q)
    name = data.get("name", city)
    temp = data.get("main", {}).get("temp")
    desc = (data.get("weather") or [{}])[0].get("description")
    print(f"OK: {name} temp={temp}°C desc={desc}")


if __name__ == "__main__":
    main()



