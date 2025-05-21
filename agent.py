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

# Load environment variables
load_dotenv()

def get_weather(location_with_units: str) -> str:
    """Get the current weather for a given location.
    
    Args:
        location_with_units: Format should be 'city,country:units' or just 'city,country' 
        (e.g., 'London,UK:imperial' or 'London,UK'). Units are optional and default to metric.
    """
    # Parse the input string
    parts = location_with_units.split(':')
    location = parts[0]
    units = parts[1] if len(parts) > 1 else "metric"
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OpenWeather API key not found in environment variables"
    
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": api_key,
        "units": units
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]
        
        weather_info = (
            f"Current weather in {location}:\n"
            f"Temperature: {temp}°{'C' if units == 'metric' else 'F'}\n"
            f"Humidity: {humidity}%\n"
            f"Conditions: {description}"
        )
        return weather_info
    
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

def get_weather_forecast(location_days: str) -> str:
    """Get the weather forecast for a given location and number of days.
    
    Args:
        location_days: Format should be 'city,country:days' where days is optional 
        (e.g., 'London,UK:3' or 'London,UK'). Days defaults to 3 if not specified.
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
            condition = day["day"]["condition"]["text"]
            rain_chance = day["day"]["daily_chance_of_rain"]
            
            forecast_info += (
                f"Date: {date}\n"
                f"Max Temperature: {max_temp_c}°C\n"
                f"Min Temperature: {min_temp_c}°C\n"
                f"Conditions: {condition}\n"
                f"Chance of Rain: {rain_chance}%\n\n"
            )
        
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
        description="Get the current weather for a location. Input should be a city name with optional country code and units (e.g., 'London,UK' or 'London,UK:imperial' for Fahrenheit, defaults to metric/Celsius)"
    )
    
    forecast_tool = StructuredTool.from_function(
        func=get_weather_forecast,
        name="get_weather_forecast",
        description="Get the weather forecast for a location. Input should be a city name with optional country code and number of days (e.g., 'London,UK:3' for 3-day forecast, defaults to 3 days if not specified)"
    )
    
    historical_tool = StructuredTool.from_function(
        func=get_historical_weather,
        name="get_historical_weather",
        description="Get historical weather data for a location. Input should be a city name with country code and date in format 'city,country:YYYY-MM-DD' (e.g., 'London,UK:2024-01-01')"
    )
    
    # Initialize the agent using LangChain's standard initialize_agent function
    agent_executor = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=[current_weather_tool, forecast_tool, historical_tool],
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
    
    print("Azure OpenAI Agent initialized! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        try:
            # Check if weather-related before sending to agent
            if not is_weather_related(user_input):
                print("\nAgent: I can only help you with weather-related questions. Please ask me about the weather in a specific location.")
                print("-" * 50)
                continue
                
            # Get response from agent
            response = agent.invoke({"input": user_input})
            print("\nAgent:", response["output"])
            print("-" * 50)
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("-" * 50)

if __name__ == "__main__":
    main() 