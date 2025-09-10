from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel


class WeatherQuery(BaseModel):
    city: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    units: str = "metric"
    lang: Optional[str] = None

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {"units": self.units}
        if self.city:
            params["q"] = self.city
        if self.lat is not None and self.lon is not None:
            params["lat"] = self.lat
            params["lon"] = self.lon
        if self.lang:
            params["lang"] = self.lang
        return params


class OpenWeatherClient:
    def __init__(self, api_key: str, timeout_s: float = 10.0) -> None:
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout_s)
        self._base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    @property
    def api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    def fetch_weather(self, query: WeatherQuery) -> Dict[str, Any]:
        params = query.to_params()
        params["appid"] = self._api_key
        resp = self._client.get(self._base_url, params=params)
        resp.raise_for_status()
        return resp.json()


