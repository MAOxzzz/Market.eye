# Machine Learning Models

This directory contains the machine learning models used for stock price forecasting.

## Components

- LSTM Models - Deep learning models for time series forecasting
- Weights and parameters for trained models
- Model training and evaluation scripts

## Model Architecture

The primary forecasting models use LSTM (Long Short-Term Memory) neural networks:

- **Input**: Sequence of 30 days of historical stock prices
- **Architecture**: LSTM(50) → Dense(1)
- **Output**: Predicted stock price for the next day

## Usage

Models are accessed through the forecasting API:

```python
from forecasting.forecast_api import ForecastAPI

# Initialize forecasting API
forecast_api = ForecastAPI()

# Get forecast for a ticker
historical_df, forecast_df = forecast_api.get_historical_and_forecast('AAPL', days=30)
```
