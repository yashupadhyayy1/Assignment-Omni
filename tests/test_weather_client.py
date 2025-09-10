from assignment_omni.clients.weather import WeatherQuery


def test_weather_query_params_city():
    q = WeatherQuery(city="London")
    params = q.to_params()
    assert params["q"] == "London"
    assert params["units"] == "metric"


