import streamlit as st
from agent import create_agent, is_weather_related
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Weather Assistant",
    page_icon="🌤️",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create header
st.title("🌤️ Weather Assistant")
st.markdown("""
Ask me about the weather in any city! You can:
- Get current weather (e.g., "What's the weather in London?")
- Specify units (e.g., "What's the weather in Tokyo in imperial units?")
- Get weather forecasts (e.g., "What's the 3-day forecast for Paris?")
- Check historical weather (e.g., "What was the weather in New York on 2024-01-01?")
- Ask about multiple cities (e.g., "Compare weather in Paris and New York")
- Check future conditions (e.g., "Will it rain in London tomorrow?")

**Note:** 
- Weather forecasts are available for up to 3 days
- Historical data is available for past dates
- Temperature in forecasts and historical data is shown in Celsius
""")

# Add tabs for different views
tab1, tab2 = st.tabs(["💬 Chat", "ℹ️ About"])

with tab1:
    # Initialize agent
    @st.cache_resource
    def get_agent():
        return create_agent()

    agent = get_agent()

    # Chat input
    user_input = st.chat_input("Ask about weather or forecast...")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if weather-related before sending to agent
                    if not is_weather_related(user_input):
                        output = "I can only help you with weather-related questions. Please ask me about the weather or forecast in a specific location."
                    else:
                        response = agent.invoke({"input": user_input})
                        output = response.get("output", "I couldn't process that request.")
                    
                    st.write(output)
                    # Add assistant message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": output})
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    # Add error message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Add a clear button to the sidebar
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with tab2:
    st.markdown("""
    ### About this Weather Assistant

    This is an AI-powered weather assistant that can help you with:

    **Current Weather Information:**
    - Real-time temperature
    - Humidity levels
    - Current weather conditions
    - Support for both metric and imperial units

    **Weather Forecasts:**
    - Up to 3-day weather forecasts
    - Daily high and low temperatures
    - Precipitation chances
    - Weather conditions outlook

    **Historical Weather Data:**
    - Past weather conditions
    - Historical temperature records
    - Precipitation data
    - Wind speed and humidity records

    **Data Sources:**
    - Current weather data: OpenWeather API
    - Weather forecasts and historical data: WeatherAPI.com

    **Tips for Best Results:**
    1. Be specific about the location (city and country)
    2. Specify units if you prefer imperial (default is metric)
    3. For forecasts, specify the number of days (1-3)
    4. For historical data, use the format YYYY-MM-DD
    5. You can ask natural questions about current conditions, forecasts, and historical data

    **Example Queries:**
    ```
    - What's the weather like in Paris, France?
    - Show me the weather in New York in imperial units
    - What's the 3-day forecast for Tokyo, Japan?
    - Will it rain in London tomorrow?
    - What was the weather in Singapore on 2024-01-01?
    - Show me historical weather for Dubai on 2024-02-15
    ```
    """)

# Add some styling
st.markdown("""
<style>
    .stChat {
        padding: 1rem;
    }
    .stChatMessage {
        margin-bottom: 1rem;
    }
    .stTabs {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)
