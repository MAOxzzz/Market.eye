"""
CrewAI integration with Google's Gemini API.
This file configures CrewAI to use Gemini as the LLM provider instead of OpenAI.
"""

import os
import logging
from crewai import Agent
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiCrewManager:
    """Manager for creating CrewAI agents with Gemini API."""
    
    def __init__(self, api_key=None):
        """
        Initialize with Gemini API key.
        
        Args:
            api_key (str, optional): Google Gemini API key
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "AIzaSyCcB3tENdnFNKBIvciETMF196ldlUBmnyk")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Set the API key in the environment for CrewAI to use
        os.environ["OPENAI_API_KEY"] = "dummy_key"  # Dummy key to prevent errors
        
        # Initialize the Gemini LLM
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.4
        )
        
        logger.info("GeminiCrewManager initialized with Gemini API")
    
    def create_agent(self, role, goal, backstory, tools=None, verbose=True):
        """
        Create a CrewAI agent using Gemini as the LLM.
        
        Args:
            role (str): Agent role
            goal (str): Agent goal
            backstory (str): Agent backstory
            tools (list): List of tools for the agent
            verbose (bool): Whether to enable verbose mode
            
        Returns:
            Agent: CrewAI agent configured with Gemini
        """
        # Set up a dummy OpenAI key in env vars (needed by CrewAI internally)
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-gemini-integration"
        
        # Create agent with Gemini LLM only, no tools yet
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=verbose,
            llm=self.gemini_llm,
            allow_delegation=False
        )
        
        # Add tools to the agent after creation if they exist
        if tools:
            agent.tools = tools
        
        return agent

# Example usage
if __name__ == "__main__":
    # Initialize the manager
    manager = GeminiCrewManager()
    
    # Create a test agent
    test_agent = manager.create_agent(
        role="Test Agent",
        goal="Test Gemini integration with CrewAI",
        backstory="You are a test agent created to verify that Gemini works with CrewAI."
    )
    
    print("Test agent created successfully!") 