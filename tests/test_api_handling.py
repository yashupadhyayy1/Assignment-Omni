"""
Test cases for API handling functionality.
Tests weather API client, error handling, and response processing.
"""

import pytest
import os
from unittest.mock import Mock, patch
from assignment_omni.clients.weather import OpenWeatherClient, WeatherQuery


class TestWeatherAPIHandling:
    """Test cases for weather API handling."""
    
    def test_weather_query_creation(self):
        """Test WeatherQuery object creation and parameter validation."""
        # Test city-based query
        query = WeatherQuery(city="London")
        params = query.to_params()
        
        assert params["q"] == "London"
        assert params["units"] == "metric"
        # appid is added by the client, not the query object
        
        # Test coordinate-based query
        query_coords = WeatherQuery(lat=51.5074, lon=-0.1278)
        params_coords = query_coords.to_params()
        
        assert params_coords["lat"] == 51.5074
        assert params_coords["lon"] == -0.1278
        assert "q" not in params_coords
    
    def test_weather_query_validation(self):
        """Test WeatherQuery validation logic."""
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
    
    @patch('assignment_omni.clients.weather.httpx.Client.get')
    def test_weather_api_success(self, mock_get):
        """Test successful weather API call."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "London",
            "main": {"temp": 15.5, "humidity": 80},
            "weather": [{"description": "cloudy"}],
            "wind": {"speed": 3.2}
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test API call
        client = OpenWeatherClient("test_api_key")
        query = WeatherQuery(city="London")
        result = client.fetch_weather(query)
        
        assert result["name"] == "London"
        assert result["main"]["temp"] == 15.5
        assert result["weather"][0]["description"] == "cloudy"
        mock_get.assert_called_once()
    
    @patch('assignment_omni.clients.weather.httpx.Client.get')
    def test_weather_api_error_handling(self, mock_get):
        """Test weather API error handling."""
        # Mock API error response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.return_value = mock_response
        
        # Test error handling
        client = OpenWeatherClient("test_api_key")
        query = WeatherQuery(city="InvalidCity")
        
        with pytest.raises(Exception, match="API Error"):
            client.fetch_weather(query)
    
    @patch('assignment_omni.clients.weather.httpx.Client.get')
    def test_weather_api_rate_limiting(self, mock_get):
        """Test weather API rate limiting handling."""
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.raise_for_status.side_effect = Exception("Rate limit exceeded")
        mock_get.return_value = mock_response
        
        client = OpenWeatherClient("test_api_key")
        query = WeatherQuery(city="London")
        
        with pytest.raises(Exception, match="Rate limit exceeded"):
            client.fetch_weather(query)
    
    def test_weather_api_key_validation(self):
        """Test weather API key validation."""
        # Test with valid API key
        client = OpenWeatherClient("valid_key_123")
        assert client.api_key == "valid_key_123"
        
        # Test with empty API key
        client_empty = OpenWeatherClient("")
        assert client_empty.api_key == ""
    
    @patch('assignment_omni.clients.weather.httpx.Client.get')
    def test_weather_api_invalid_city(self, mock_get):
        """Test weather API with invalid city."""
        # Mock 404 response for invalid city
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("City not found")
        mock_get.return_value = mock_response
        
        client = OpenWeatherClient("test_api_key")
        query = WeatherQuery(city="NonExistentCity")
        
        with pytest.raises(Exception, match="City not found"):
            client.fetch_weather(query)
    
    def test_weather_query_parameter_encoding(self):
        """Test weather query parameter encoding."""
        # Test special characters in city names
        query = WeatherQuery(city="São Paulo")
        params = query.to_params()
        assert params["q"] == "São Paulo"
        
        # Test spaces in city names
        query = WeatherQuery(city="New York City")
        params = query.to_params()
        assert params["q"] == "New York City"


class TestAPIErrorHandling:
    """Test cases for general API error handling."""
    
    def test_network_timeout_handling(self):
        """Test network timeout handling."""
        with patch('assignment_omni.clients.weather.httpx.Client.get') as mock_get:
            mock_get.side_effect = Exception("Network timeout")
            
            client = OpenWeatherClient("test_key")
            query = WeatherQuery(city="London")
            
            with pytest.raises(Exception, match="Network timeout"):
                client.fetch_weather(query)
    
    def test_invalid_json_response(self):
        """Test handling of invalid JSON response."""
        with patch('assignment_omni.clients.weather.httpx.Client.get') as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            client = OpenWeatherClient("test_key")
            query = WeatherQuery(city="London")
            
            with pytest.raises(ValueError, match="Invalid JSON"):
                client.fetch_weather(query)


if __name__ == "__main__":
    pytest.main([__file__])
