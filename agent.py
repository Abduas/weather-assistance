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
    """Check if the input is a valid city name using OpenWeather API.
    
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

def process_user_input(query: str) -> str:
    """Process user input to determine if it's a city name or weather-related query.
    
    Args:
        query: User input string
    Returns:
        str: Processed query for the agent
    """
    # Remove common weather-related words to isolate potential city name
    weather_words = ["weather", "temperature", "forecast", "climate", "in", "at", "of", "the"]
    query_words = query.lower().split()
    
    # Extract potential city name by removing weather-related words
    potential_cities = [word for word in query_words if word not in weather_words]
    
    # If we have potential cities, check if any are valid
    for city in potential_cities:
        if is_valid_city(city):
            return f"get weather for {city}"
            
    # If the original query contains weather words, return it as is
    if any(word in query.lower() for word in weather_words):
        return query
        
    # If it's a single word, check if it's a city
    if len(query_words) == 1 and is_valid_city(query):
        return f"get weather for {query}"
    
    return query

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
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        city_name = data["name"]
        country = data["sys"]["country"]
        
        # Calculate water intake recommendation
        recommended_intake, factors = calculate_water_intake(temp, humidity)
        
        weather_info = (
            f"Current weather in {city_name}, {country}:\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Conditions: {description}\n\n"
            f"💧 Water Intake Recommendation 💧\n"
            f"Recommended water intake: {recommended_intake:.1f} liters\n\n"
            f"Factors affecting recommendation:\n"
        )
        
        # Add factors to the output
        for factor in factors:
            weather_info += f"- {factor}\n"
            
        weather_info += "\nRemember to adjust intake based on physical activity and personal health conditions."
        
        return weather_info
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

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
    
    # Limit forecast days to 3 for free tier
    days = min(days, 3)
    
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
            date = day["date"]
            max_temp_c = day["day"]["maxtemp_c"]
            min_temp_c = day["day"]["mintemp_c"]
            avg_temp_c = day["day"]["avgtemp_c"]
            humidity = day["day"]["avghumidity"]
            condition = day["day"]["condition"]["text"]
            rain_chance = day["day"]["daily_chance_of_rain"]
            
            # Calculate water intake recommendation for average temperature
            recommended_intake, factors = calculate_water_intake(avg_temp_c, humidity)
            
            forecast_info += (
                f"Date: {date}\n"
                f"Temperature: {min_temp_c}°C to {max_temp_c}°C (avg: {avg_temp_c}°C)\n"
                f"Humidity: {humidity}%\n"
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
    
    try:
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
        
        historical_info = (
            f"Historical weather for {location_name}, {country} on {date}:\n\n"
            f"Maximum Temperature: {max_temp_c}°C\n"
            f"Minimum Temperature: {min_temp_c}°C\n"
            f"Average Temperature: {avg_temp_c}°C\n"
            f"Conditions: {condition}\n"
            f"Total Precipitation: {total_precip_mm}mm\n"
            f"Maximum Wind Speed: {max_wind_kph} km/h\n"
            f"Average Humidity: {humidity}%\n"
        )
        
        return historical_info
    
    except requests.exceptions.RequestException as e:
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
        description="Get the current weather and water intake recommendation for a location. Input should be a city name (e.g., 'London' or 'New York')"
    )
    
    forecast_tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast",
        description="Get the weather forecast and water intake recommendations for a location. Input should be a city name with optional days (e.g., 'London:3' for 3-day forecast, defaults to 3 days)"
    )
    
    historical_tool = StructuredTool.from_function(
        func=get_historical_weather,
        name="get_historical_weather",
        description="Get historical weather data for a location. Input should be a city name with country code and date in format 'city,country:YYYY-MM-DD' (e.g., 'London,UK:2024-01-01')"
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
        
        # Process the user input
        processed_input = process_user_input(user_input)
        
        try:
            # If it's a valid city, get the weather directly
            if is_valid_city(user_input):
                response = get_weather(user_input)
                print("\nAssistant:", response)
            else:
                # Otherwise, let the agent handle the query
                response = agent.invoke({"input": processed_input})
                print("\nAssistant:", response["output"])
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main() 