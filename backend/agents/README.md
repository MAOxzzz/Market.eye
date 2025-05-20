# AI Agents

This directory contains the AI agents that power Market Eye AI's analysis capabilities.

## Components

- `data_collector_agent.py` - Responsible for gathering and preparing stock data
- `data_processor_agent.py` - Analyzes trends and generates initial forecasts
- `llm_recommendation_agent.py` - Creates investment advice using Google's Gemini API
- `stock_analysis_crew.py` - Orchestrates agent collaboration for comprehensive analysis
- `crewai_gemini_integration.py` - Integrates Google's Gemini API with CrewAI framework
- `tools/` - Specialized tools that agents can use for analysis tasks

## Usage

Agents are orchestrated through the CrewAI framework and can be used individually or as a crew:

```python
from agents.stock_analysis_crew import StockAnalysisCrew

# Initialize the crew
crew = StockAnalysisCrew()

# Run analysis on selected tickers
results = crew.analyze_stocks(['AAPL', 'MSFT', 'GOOGL'])
```
