"""
CrewAI integration with Google's Gemini API.
This file configures CrewAI to use Gemini as the LLM provider instead of OpenAI.
"""

import os
import logging
from crewai import Agent
import google.generativeai as genai
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Import config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom LLM class to wrap Gemini model for CrewAI
class GeminiLLM(BaseModel):
    """Custom LLM interface for Gemini model."""
    
    api_key: str
    model_name: str = "gemini-pro"
    temperature: float = 0.4
    max_tokens: int = 1024
    top_p: float = 0.95
    top_k: int = 40
    
    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "max_output_tokens": self.max_tokens,
                }
            )
            logger.info(f"Initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def create_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using Gemini model.
        
        Args:
            messages: List of message dictionaries with role and content
            
        Returns:
            Dictionary with response content
        """
        try:
            # Convert messages from OpenAI format to Gemini format
            gemini_messages = []
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    # For system messages, prepend to first user message or add as user message
                    if gemini_messages:
                        # Find first user message
                        for i, msg in enumerate(gemini_messages):
                            if msg.role == "user":
                                gemini_messages[i].parts[0].text = f"{content}\n\n{gemini_messages[i].parts[0].text}"
                                break
                        else:
                            # No user message found, add as user message
                            gemini_messages.append(genai.types.ContentPart(role="user", text=content))
                    else:
                        # No messages yet, add as user message
                        gemini_messages.append(genai.types.ContentPart(role="user", text=content))
                else:
                    # Map roles from OpenAI to Gemini
                    gemini_role = "user" if role == "user" else "model"
                    gemini_messages.append(genai.types.ContentPart(role=gemini_role, text=content))
            
            # Generate response
            response = self.model.generate_content(gemini_messages)
            
            # Format response to match OpenAI structure
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response.text
                        }
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            # Return a fallback response
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"I'm sorry, I encountered an error: {str(e)}"
                        }
                    }
                ]
            }

class GeminiCrewManager:
    """Manager for creating CrewAI agents with Gemini API."""
    
    def __init__(self, api_key=None):
        """
        Initialize with Gemini API key.
        
        Args:
            api_key (str, optional): Google Gemini API key
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. CrewAI agents will not function correctly.")
        
        # Initialize the Gemini LLM
        self.gemini_llm = GeminiLLM(
            api_key=self.api_key,
            model_name=config.GEMINI_MODEL,
            temperature=0.4
        )
        
        logger.info("GeminiCrewManager initialized with direct Gemini API integration")
    
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