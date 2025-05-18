# Market Eye AI - CrewAI Agents

This module implements the CrewAI agents for the Market Eye AI system.

## Overview

The CrewAI implementation consists of three specialized agents that work together to collect data, process it, and generate investment recommendations:

1. **Data Collector Agent**: Responsible for fetching stock data from Kaggle and preparing it for analysis
2. **Data Processor Agent**: Analyzes the data, calculates metrics, and generates forecasts using LSTM models
3. **LLM Recommendation Agent**: Generates investment recommendations using the Gemini API

## Architecture

```
agents/
├── tools/                     # Tools used by the agents
│   ├── kaggle_tool.py         # Tool for accessing Kaggle datasets
│   ├── data_analysis_tool.py  # Tool for analyzing stock data
│   └── gemini_tool.py         # Tool for generating recommendations
├── data_collector_agent.py    # Agent 1: Data Collector
├── data_processor_agent.py    # Agent 2: Data Processor
├── llm_recommendation_agent.py # Agent 3: LLM Recommendation Generator
├── stock_analysis_crew.py     # Orchestration of all agents
└── api.py                     # API endpoints for CrewAI functionality
```

## How It Works

1. **Data Collection**:

   - The Data Collector agent fetches stock data from the Kaggle "World Stock Prices" dataset
   - It cleans the data and prepares it for analysis
   - The agent can handle specific tickers or process all available data

2. **Data Processing**:

   - The Data Processor agent calculates key metrics (highest price, lowest price, annual growth)
   - It compares sectors (Technology, Finance, Sportswear)
   - It trains LSTM models to predict future stock prices
   - It generates January 2025 price forecasts
   - It calculates error metrics (MSE, RMSE) when actual data is available

3. **Recommendation Generation**:
   - The LLM Recommendation agent builds dynamic prompts from the analysis
   - It includes metrics, forecasts, and model accuracy in the prompts
   - It calls the Gemini API to generate BUY/SELL/HOLD recommendations
   - It extracts and formats the recommendations for display

## API Endpoints

The CrewAI functionality is exposed through FastAPI endpoints:

- `POST /crewai/analyze` - Start a stock analysis job
- `GET /crewai/status/{request_id}` - Check the status of an analysis
- `GET /crewai/results/{request_id}` - Get the results of a completed analysis
- `GET /crewai/active` - List all active analyses

## Usage Example

```python
from agents.stock_analysis_crew import StockAnalysisCrew

# Create the crew
crew = StockAnalysisCrew()

# Run analysis for specific tickers
results = crew.analyze_stocks(['AAPL', 'MSFT', 'GOOGL'])

# Access the recommendations
for ticker, rec in results['recommendations']['ticker_recommendations'].items():
    rec_type = rec.get('recommendation_type', 'UNKNOWN')
    print(f"{ticker}: {rec_type}")

# See the market summary
print(results['recommendations']['market_summary'])
```

## Configuration

- **Kaggle API**: To use the Kaggle data source, you need to configure your Kaggle API credentials in `~/.kaggle/kaggle.json`
- **Gemini API**: Set your Gemini API key as an environment variable `GEMINI_API_KEY` or pass it directly when initializing the recommendation agent

## Adding New Agents

To add a new agent to the crew:

1. Create a new tool in the `tools/` directory
2. Create a new agent class extending the CrewAI Agent
3. Add the agent to the `stock_analysis_crew.py` file
4. Update the crew workflow to include the new agent
