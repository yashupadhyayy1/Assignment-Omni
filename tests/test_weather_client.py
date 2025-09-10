import pytest
from assignment_omni.clients.weather import WeatherQuery, OpenWeatherClient


def test_weather_query_params_city():
    """Test weather query parameter creation for city-based queries."""
    q = WeatherQuery(city="London")
    params = q.to_params()
    assert params["q"] == "London"
    assert params["units"] == "metric"
    # appid is added by the client, not the query object


def test_weather_query_params_coordinates():
    """Test weather query parameter creation for coordinate-based queries."""
    q = WeatherQuery(lat=51.5074, lon=-0.1278)
    params = q.to_params()
    assert params["lat"] == 51.5074
    assert params["lon"] == -0.1278
    assert params["units"] == "metric"
    assert "q" not in params


def test_weather_query_validation():
    """Test weather query validation logic."""
    # Valid city query
    query = WeatherQuery(city="New York")
    assert query.city == "New York"
    assert query.lat is None
    assert query.lon is None
    
    # Valid coordinate query
    query = WeatherQuery(lat=40.7128, lon=-74.0060)
    assert query.city is None
    assert query.lat == 40.7128
    assert query.lon == -74.0060


def test_weather_client_initialization():
    """Test weather client initialization."""
    client = OpenWeatherClient("test_api_key")
    assert client.api_key == "test_api_key"
    
    # Test with empty API key
    client_empty = OpenWeatherClient("")
    assert client_empty.api_key == ""


def test_weather_query_parameter_encoding():
    """Test weather query parameter encoding for special characters."""
    # Test special characters in city names
    query = WeatherQuery(city="São Paulo")
    params = query.to_params()
    assert params["q"] == "São Paulo"
    
    # Test spaces in city names
    query = WeatherQuery(city="New York City")
    params = query.to_params()
    assert params["q"] == "New York City"


if __name__ == "__main__":
    pytest.main([__file__])
