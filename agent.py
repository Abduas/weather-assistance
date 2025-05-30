from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from dotenv import load_dotenv
import os
import requests
from typing import Optional
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.schema import AgentAction, AgentFinish
import json
from mock_data import generate_mock_historical_weather
from fuzzywuzzy import fuzz
import re
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

def calculate_water_intake(temp: float, humidity: float) -> tuple[float, list[str]]:
    """Calculate recommended water intake based on weather conditions.
    
    Args:
        temp: Temperature in Celsius
        humidity: Humidity percentage
    
    Returns:
        Tuple of (recommended_intake, factors_list)
    """
    # Base water intake (in liters)
    base_intake = 2.5
    factors = []
    
    # Temperature adjustments
    if temp > 30:
        temp_factor = 1.5
        factors.append("Very high temperature (+50% intake)")
    elif temp > 25:
        temp_factor = 1.3
        factors.append("High temperature (+30% intake)")
    elif temp > 20:
        temp_factor = 1.1
        factors.append("Moderate temperature (+10% intake)")
    else:
        temp_factor = 1.0
        factors.append("Normal temperature")
        
    # Humidity adjustments
    if humidity < 30:
        humidity_factor = 1.2
        factors.append("Low humidity (+20% intake)")
    elif humidity > 70:
        humidity_factor = 0.9
        factors.append("High humidity (-10% intake)")
    else:
        humidity_factor = 1.0
        factors.append("Normal humidity")
        
    recommended_intake = base_intake * temp_factor * humidity_factor
    return recommended_intake, factors

def is_valid_city(city: str) -> bool:
    """Check if the input is a valid city name using OpenWeather API. Also if the spelling is incorrect, it should return the closest match.
    
    Args:
        city: Name of the city to validate
    Returns:
        bool: True if city exists, False otherwise
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return False
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params)
        return response.status_code == 200
    except:
        return False

def get_closest_city_match(misspelled_city: str) -> tuple[str, int]:
    """Find the closest matching city using the OpenWeather API and fuzzy matching.
    
    Args:
        misspelled_city: Potentially misspelled city name
    Returns:
        tuple: (closest_match, confidence_score)
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return (misspelled_city, 0)
    
    # Try exact match first
    if is_valid_city(misspelled_city):
        return (misspelled_city, 100)
    
    # List of common city variations to try
    variations = [
        misspelled_city,
        misspelled_city.capitalize(),
        misspelled_city.lower(),
        misspelled_city.upper(),
        # Remove repeated letters (e.g., kasaragodd -> kasaragod)
        re.sub(r'(.)\1+', r'\1', misspelled_city)
    ]
    
    highest_score = 0
    best_match = misspelled_city
    
    for variation in variations:
        try:
            base_url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": variation,
                "appid": api_key,
                "units": "metric"
            }
            response = requests.get(base_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                city_name = data["name"]
                # Calculate similarity score
                score = fuzz.ratio(misspelled_city.lower(), city_name.lower())
                if score > highest_score:
                    highest_score = score
                    best_match = city_name
        except:
            continue
    
    return (best_match, highest_score)

def parse_date(date_str: str) -> datetime:
    """Parse date from various formats.
    
    Args:
        date_str: Date string in various formats (e.g., '24th may 2025', 'may 24 2025')
    Returns:
        datetime object
    """
    # Remove ordinal indicators (st, nd, rd, th)
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str.lower())
    
    # Try different date formats
    formats = [
        '%d %B %Y',
        '%B %d %Y',
        '%d %b %Y',
        '%b %d %Y',
        '%Y-%m-%d',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    raise ValueError(f"Could not parse date: {date_str}")

def process_user_input(query: str) -> str:
    """Process user input to handle spelling mistakes and identify city names and dates.
    
    Args:
        query: User input string
    Returns:
        str: Processed query for the agent
    """
    # Define stop words at the beginning of the function
    stop_words = {"in", "at", "of", "the", "is", "what", "how", "tell", "me", "about", "get", "show", 
                 "was", "on", "before", "ago", "last", "week", "weeks", "day", "days", "month", "months"}
    
    # Handle relative date references
    query_lower = query.lower()
    current_date = datetime.now()
    
    # Initialize corrected words list and weather term flag
    corrected_words = []
    has_weather_term = False
    
    # Handle "yesterday"
    if "yesterday" in query_lower:
        yesterday = current_date - timedelta(days=1)
        # Replace "yesterday" with the actual date
        query_lower = query_lower.replace("yesterday", yesterday.strftime("%d %B %Y"))
    
    # Handle "last week"
    elif "last week" in query_lower:
        last_week = current_date - timedelta(days=7)
        query_lower = query_lower.replace("last week", last_week.strftime("%d %B %Y"))
    
    # Handle "last month"
    elif "last month" in query_lower:
        last_month = current_date - timedelta(days=30)
        query_lower = query_lower.replace("last month", last_month.strftime("%d %B %Y"))
    
    # Common weather-related words and their correct spellings
    weather_words = {
        "weather": ["weather", "wether", "wheather", "whether", "wethr"],
        "temperature": ["temperature", "temp", "temprature", "tempreture"],
        "forecast": ["forecast", "forcast", "fourcast", "forcaste"],
        "humidity": ["humidity", "humidty", "huminity"],
        "wind": ["wind", "wnd", "vind"],
        "rain": ["rain", "rane", "reign"],
        "cloudy": ["cloudy", "cloudi", "claudy"]
    }
    
    # Try to extract date from the query
    date_pattern = r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})'
    date_matches = re.findall(date_pattern, query_lower)
    
    if date_matches:
        try:
            query_date = parse_date(date_matches[0])
            current_date = datetime.now()
            
            # If the date is within 14 days in the future, use forecast
            if query_date > current_date and (query_date - current_date).days <= 14:
                days_ahead = (query_date - current_date).days
                # Process rest of the query for city name
                query_without_date = query_lower.replace(date_matches[0], '').strip()
                # Extract city and process for weather forecast
                for word in query_without_date.split():
                    if word not in stop_words and not any(word in vars for vars in weather_words.values()):
                        closest_city, confidence = get_closest_city_match(word)
                        if confidence > 70:
                            return f"get weather forecast for {closest_city}:{days_ahead}"
            
            # If the date is in the past, use historical weather
            elif query_date < current_date:
                # Process rest of the query for city name
                query_without_date = query_lower.replace(date_matches[0], '').strip()
                # Extract city and process for historical weather
                for word in query_without_date.split():
                    if word not in stop_words and not any(word in vars for vars in weather_words.values()):
                        closest_city, confidence = get_closest_city_match(word)
                        if confidence > 70:
                            return f"get historical weather for {closest_city}:{query_date.strftime('%Y-%m-%d')}"
            
            # If the date is too far in the future
            else:
                return f"The date {query_date.strftime('%d %B %Y')} is too far in the future. I can only provide forecasts for the next 14 days."
                
        except ValueError:
            pass
    
    # Continue with the existing word correction logic
    for word in query_lower.split():
        corrected = word
        # Check if it's a weather-related word with spelling mistake
        for correct_word, variations in weather_words.items():
            if any(fuzz.ratio(word, var) > 75 for var in variations):
                corrected = correct_word
                break
        corrected_words.append(corrected)
    
    # Join words back together
    corrected_query = " ".join(corrected_words)
    
    # Look for potential city names
    potential_cities = [word for word in corrected_words if word not in stop_words and 
                       not any(word in vars for vars in weather_words.values())]
    
    # Try to find the closest matching city
    for city in potential_cities:
        closest_city, confidence = get_closest_city_match(city)
        if confidence > 70:
            corrected_query = corrected_query.replace(city, closest_city)
            if city != closest_city:
                return f"Note: Assuming you meant '{closest_city}' instead of '{city}'.\n{corrected_query}"
    
    # If we found a city but no weather term, add "get weather for"
    if potential_cities:
        return f"get weather for {corrected_query}"
    
    return corrected_query

def get_weather(location: str) -> str:
    """Get the current weather and water intake recommendation for a given location.
    
    Args:
        location: City name (e.g., 'London' or 'New York')
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeather API key not found in environment variables"
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Validate required fields
        required_fields = {
            "main": ["temp", "humidity"],  # Only require essential fields
            "name": None
        }
        
        for field, subfields in required_fields.items():
            if field not in data:
                return f"Error: Missing {field} data from API response"
            if subfields:
                for subfield in subfields:
                    if subfield not in data[field]:
                        return f"Error: Missing {field}.{subfield} data from API response"
        
        # Extract all weather data
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        feels_like = data["main"].get("feels_like", temp)  # Default to temp if feels_like missing
        pressure = data["main"].get("pressure", "N/A")
        description = data["weather"][0]["description"] if "weather" in data and data["weather"] else "No description available"
        city_name = data["name"]
        country = data["sys"]["country"] if "sys" in data and "country" in data["sys"] else ""
        
        # Extract wind data with defaults
        wind_speed = data.get("wind", {}).get("speed", 0)  # Default to 0 if missing
        wind_speed_kmh = wind_speed * 3.6   # convert to km/h
        wind_direction = data.get("wind", {}).get("deg", 0)  # Default to 0 if missing
        
        # Get wind direction as compass point
        def get_wind_direction(degrees):
            directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            index = round(degrees / (360 / len(directions))) % len(directions)
            return directions[index]
        
        wind_compass = get_wind_direction(wind_direction)
        
        # Calculate water intake recommendation
        recommended_intake, factors = calculate_water_intake(temp, humidity)
        
        # Add wind factor to water intake recommendation if wind speed is high
        if wind_speed_kmh > 20:
            recommended_intake *= 1.1  # Increase water intake by 10% in windy conditions
            factors.append("High wind speed (+10% intake)")
        
        # Create wind description based on speed
        wind_description = "Light breeze"
        if wind_speed_kmh > 30:
            wind_description = "Strong wind"
        elif wind_speed_kmh > 20:
            wind_description = "Moderate wind"
        elif wind_speed_kmh > 10:
            wind_description = "Gentle breeze"
        
        # Get current timestamp in local time
        local_time = datetime.fromtimestamp(data["dt"]).strftime("%Y-%m-%d %H:%M:%S")
        
        weather_info = (
            f"Current weather in {city_name}, {country} (as of {local_time}):\n"
            f"Temperature: {temp}°C (Feels like: {feels_like}°C)\n"
            f"Wind: {wind_description} at {wind_speed_kmh:.1f} km/h from {wind_compass}\n"
            f"Humidity: {humidity}%\n"
            f"Pressure: {pressure} hPa\n"
            f"Conditions: {description}\n"
        )
        
        # Add extra weather data if available
        if "rain" in data and "1h" in data["rain"]:
            weather_info += f"Rainfall (last hour): {data['rain']['1h']} mm\n"
        if "clouds" in data and "all" in data["clouds"]:
            weather_info += f"Cloud cover: {data['clouds']['all']}%\n"
        if "visibility" in data:
            visibility_km = data["visibility"] / 1000
            weather_info += f"Visibility: {visibility_km:.1f} km\n"
            
        weather_info += f"\n💧 Water Intake Recommendation 💧\n"
        weather_info += f"Recommended water intake: {recommended_intake:.1f} liters\n\n"
        weather_info += f"Factors affecting recommendation:\n"
        
        # Add factors to the output
        for factor in factors:
            weather_info += f"- {factor}\n"
            
        weather_info += "\nRemember to adjust intake based on physical activity and personal health conditions."
        
        return weather_info
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Error processing weather data: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_weather_forecast(location_days: str) -> str:
    """Get the weather forecast and water intake recommendations for a given location and number of days.
    
    Args:
        location_days: Format should be 'city:days' where days is optional 
        (e.g., 'London:3' or 'London'). Days defaults to 3 if not specified.
    """
    # Parse the input string
    parts = location_days.split(':')
    location = parts[0]
    days = int(parts[1]) if len(parts) > 1 else 3
    
    # Limit forecast days to 14 for free tier and future forecasts
    days = min(days, 14)
    
    api_key = os.getenv("WEATHERAPI_KEY")
    if not api_key:
        return "Error: WeatherAPI key not found in environment variables"
    
    base_url = "http://api.weatherapi.com/v1/forecast.json"
    params = {
        "q": location,
        "key": api_key,
        "days": days,
        "aqi": "no"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        location_name = data["location"]["name"]
        country = data["location"]["country"]
        forecast_days = data["forecast"]["forecastday"]
        
        forecast_info = f"Weather forecast for {location_name}, {country}:\n\n"
        
        for day in forecast_days:
            date = datetime.strptime(day["date"], "%Y-%m-%d").strftime("%d %B %Y")
            max_temp_c = day["day"]["maxtemp_c"]
            min_temp_c = day["day"]["mintemp_c"]
            avg_temp_c = day["day"]["avgtemp_c"]
            humidity = day["day"]["avghumidity"]
            condition = day["day"]["condition"]["text"]
            rain_chance = day["day"]["daily_chance_of_rain"]
            max_wind_kph = day["day"]["maxwind_kph"]
            
            # Calculate water intake recommendation for average temperature
            recommended_intake, factors = calculate_water_intake(avg_temp_c, humidity)
            
            # Add wind factor if wind speed is high
            if max_wind_kph > 20:
                recommended_intake *= 1.1
                factors.append("High wind speed (+10% intake)")
            
            forecast_info += (
                f"Date: {date}\n"
                f"Temperature: {min_temp_c}°C to {max_temp_c}°C (avg: {avg_temp_c}°C)\n"
                f"Humidity: {humidity}%\n"
                f"Max Wind Speed: {max_wind_kph} km/h\n"
                f"Conditions: {condition}\n"
                f"Chance of Rain: {rain_chance}%\n\n"
                f"💧 Water Intake Recommendation 💧\n"
                f"Recommended water intake: {recommended_intake:.1f} liters\n"
                f"Factors considered:\n"
            )
            
            # Add factors to the output
            for factor in factors:
                forecast_info += f"- {factor}\n"
            
            forecast_info += "\n" + "-"*40 + "\n\n"
        
        return forecast_info.strip()
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching forecast data: {str(e)}"

def get_historical_weather(location_date: str) -> str:
    """Get historical weather data for a given location and date.
    
    Args:
        location_date: Format should be 'city,country:YYYY-MM-DD' 
        (e.g., 'London,UK:2024-01-01')
    """
    # Parse the input string
    parts = location_date.split(':')
    if len(parts) != 2:
        return "Error: Please provide location and date in format 'city,country:YYYY-MM-DD'"
    
    location = parts[0]
    date = parts[1]
    
    try:
        # Check if we should use mock data from environment variable
        use_mock = os.getenv("USE_MOCK_WEATHER", "true").lower() == "true"
        
        if use_mock:
            # Use mock data for development/testing
            data = generate_mock_historical_weather(location.split(',')[0], date)
        else:
            # Use real API for production
            api_key = os.getenv("WEATHERAPI_KEY")
            if not api_key:
                return "Error: WeatherAPI key not found in environment variables"
            
            base_url = "http://api.weatherapi.com/v1/history.json"
            params = {
                "q": location,
                "key": api_key,
                "dt": date,
                "aqi": "no"
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
        
        location_name = data["location"]["name"]
        country = data["location"]["country"]
        historical_data = data["forecast"]["forecastday"][0]
        
        # Get the day's data
        day_data = historical_data["day"]
        max_temp_c = day_data["maxtemp_c"]
        min_temp_c = day_data["mintemp_c"]
        avg_temp_c = day_data["avgtemp_c"]
        condition = day_data["condition"]["text"]
        total_precip_mm = day_data["totalprecip_mm"]
        max_wind_kph = day_data["maxwind_kph"]
        humidity = day_data["avghumidity"]
        
        # Calculate water intake recommendation
        recommended_intake, factors = calculate_water_intake(avg_temp_c, humidity)
        
        historical_info = (
            f"Historical weather for {location_name}, {country} on {date}:\n\n"
            f"Temperature Range: {min_temp_c}°C to {max_temp_c}°C\n"
            f"Average Temperature: {avg_temp_c}°C\n"
            f"Maximum Wind Speed: {max_wind_kph} km/h\n"
            f"Humidity: {humidity}%\n"
            f"Conditions: {condition}\n"
            f"Total Precipitation: {total_precip_mm}mm\n\n"
            f"💧 Water Intake Recommendation 💧\n"
            f"Recommended water intake: {recommended_intake:.1f} liters\n\n"
            f"Factors affecting recommendation:\n"
        )
        
        # Add factors to the output
        for factor in factors:
            historical_info += f"- {factor}\n"
            
        historical_info += "\nNote: Historical data is simulated for development purposes." if use_mock else ""
        
        return historical_info
    
    except Exception as e:
        return f"Error fetching historical weather data: {str(e)}"

def get_water_recommendation(city: str) -> str:
    """Get weather details and recommend water intake based on conditions.
    
    Args:
        city: Name of the city (e.g., 'London' or 'New York')
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeather API key not found in environment variables"
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # Using metric for consistency
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        
        # Base water intake (in liters)
        base_intake = 2.5  # Average recommended daily water intake
        
        # Adjust for temperature
        if temp > 30:
            temp_factor = 1.5
        elif temp > 25:
            temp_factor = 1.3
        elif temp > 20:
            temp_factor = 1.1
        else:
            temp_factor = 1.0
            
        # Adjust for humidity
        if humidity < 30:
            humidity_factor = 1.2
        elif humidity > 70:
            humidity_factor = 0.9
        else:
            humidity_factor = 1.0
            
        # Calculate recommended intake
        recommended_intake = base_intake * temp_factor * humidity_factor
        
        recommendation = (
            f"Weather in {city}:\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Conditions: {description}\n\n"
            f"Recommended water intake: {recommended_intake:.1f} liters\n\n"
            f"Factors affecting recommendation:\n"
            f"- {'Higher' if temp > 25 else 'Normal'} temperature\n"
            f"- {'Low' if humidity < 30 else 'High' if humidity > 70 else 'Normal'} humidity\n\n"
            f"Remember to adjust intake based on physical activity and personal health conditions."
        )
        return recommendation
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

# Custom AzureChatOpenAI class to avoid temperature parameter
class CustomAzureChatOpenAI(AzureChatOpenAI):
    @property
    def _default_params(self) -> dict:
        params = super()._default_params
        if "temperature" in params:
            del params["temperature"]
        return params

def is_weather_related(query: str) -> bool:
    """Check if the input is related to weather."""
    weather_keywords = ["weather", "temperature", "forecast", "climate", "rain", 
                       "sunny", "cloudy", "humidity", "hot", "cold", "prediction",
                       "historical", "past", "yesterday", "last week", "last month"]
    return any(keyword in query.lower() for keyword in weather_keywords)

def create_agent():
    # Initialize the Azure OpenAI language model
    llm = CustomAzureChatOpenAI(
        deployment_name="o3-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview"
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create the weather tools
    current_weather_tool = StructuredTool.from_function(
        func=get_weather,
        name="get_weather",
        description="Get the current weather information including temperature, humidity, wind speed, and water intake recommendation for a location. Input should be a city name (e.g., 'London' or 'New York'). Use this for queries about weather conditions, wind speed, or temperature in a specific city."
    )
    
    forecast_tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast",
        description="Get the weather forecast and water intake recommendations for a location. Input should be a city name with optional days (e.g., 'London:3' for 3-day forecast, defaults to 3 days)"
    )
    
    historical_tool = StructuredTool.from_function(
        func=get_historical_weather,
        name="get_historical_weather",
        description="Get historical weather data for a location and date. Input must be in format 'city,country:YYYY-MM-DD' (e.g., 'London,UK:2024-01-15')"
    )
    
    water_recommendation_tool = StructuredTool.from_function(
        func=get_water_recommendation,
        name="get_water_recommendation",
        description="Get weather details and recommended water intake based on conditions. Input should be a city name (e.g., 'London' or 'New York')"
    )
    
    # Initialize the agent using LangChain's standard initialize_agent function
    agent_executor = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=[current_weather_tool, forecast_tool, historical_tool, water_recommendation_tool],
        llm=llm,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

def main():
    # Create the agent
    agent = create_agent()
    
    print("Weather Assistant initialized! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        try:
            # Process the user input and handle spelling mistakes
            processed_input = process_user_input(user_input)
            
            # Check if there's a spelling correction note
            if processed_input.startswith("Note: "):
                correction_note, actual_query = processed_input.split("\n")
                print("\nAssistant:", correction_note)
                processed_input = actual_query
            
            # Extract the city name from the processed input
            words = processed_input.split()
            city_index = -1
            if "for" in words:
                city_index = words.index("for") + 1
            
            if city_index >= 0 and city_index < len(words):
                city = words[city_index]
                if is_valid_city(city):
                    response = get_weather(city)
                    print("\nAssistant:", response)
                else:
                    response = agent.invoke({"input": processed_input})
                    print("\nAssistant:", response["output"])
            else:
                response = agent.invoke({"input": processed_input})
                print("\nAssistant:", response["output"])
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()

