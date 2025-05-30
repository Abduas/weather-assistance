import random
from datetime import datetime, timedelta
import json

def generate_mock_historical_weather(city: str, date_str: str) -> dict:
    """Generate mock historical weather data for testing purposes.
    
    Args:
        city: Name of the city
        date_str: Date in YYYY-MM-DD format
    
    Returns:
        dict: Mock weather data in the same format as the WeatherAPI response
    """
    # Seed random with city name and date for consistent results
    random.seed(f"{city}{date_str}")
    
    # Generate realistic temperature ranges based on seasons
    date = datetime.strptime(date_str, "%Y-%m-%d")
    month = date.month
    
    # Adjust temperature ranges by season (Northern Hemisphere)
    if month in [12, 1, 2]:  # Winter
        base_temp = random.uniform(0, 10)
    elif month in [3, 4, 5]:  # Spring
        base_temp = random.uniform(10, 20)
    elif month in [6, 7, 8]:  # Summer
        base_temp = random.uniform(20, 30)
    else:  # Fall
        base_temp = random.uniform(15, 25)
    
    # Generate daily temperature variations
    max_temp = base_temp + random.uniform(2, 5)
    min_temp = base_temp - random.uniform(2, 5)
    avg_temp = (max_temp + min_temp) / 2
    
    # Generate other weather parameters
    conditions = random.choice([
        "Sunny", "Partly cloudy", "Cloudy", "Light rain",
        "Moderate rain", "Heavy rain", "Clear", "Overcast"
    ])
    
    mock_data = {
        "location": {
            "name": city,
            "country": "MockCountry",
            "lat": random.uniform(-90, 90),
            "lon": random.uniform(-180, 180),
            "localtime": f"{date_str} 12:00"
        },
        "forecast": {
            "forecastday": [{
                "date": date_str,
                "day": {
                    "maxtemp_c": round(max_temp, 1),
                    "mintemp_c": round(min_temp, 1),
                    "avgtemp_c": round(avg_temp, 1),
                    "maxwind_kph": round(random.uniform(5, 30), 1),
                    "totalprecip_mm": round(random.uniform(0, 10), 1),
                    "avghumidity": round(random.uniform(40, 90), 1),
                    "condition": {
                        "text": conditions,
                        "code": random.randint(1000, 1030)
                    }
                },
                "hour": [
                    {
                        "time": f"{date_str} {hour:02d}:00",
                        "temp_c": round(avg_temp + random.uniform(-3, 3), 1),
                        "condition": {
                            "text": conditions,
                            "code": random.randint(1000, 1030)
                        },
                        "wind_kph": round(random.uniform(5, 30), 1),
                        "humidity": round(random.uniform(40, 90), 1)
                    } for hour in range(24)
                ]
            }]
        }
    }
    
    return mock_data 