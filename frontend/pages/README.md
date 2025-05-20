# UI Pages and Components

This directory contains the Streamlit UI pages and components for Market Eye AI.

## Components

- Authentication UI
- Stock Viewer
- Investment Advisor
- Stock Forecaster
- CrewAI Analysis
- Dataset Management
- Report Generator

## Implementation

The UI is built with Streamlit and organized into tabs for better user experience:

```python
# Create tabs for different features
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "View Stocks",
    "Get Advice",
    "Stock Forecast",
    "CrewAI Analysis",
    "Dataset Management",
    "Save Report"
])
```

Each tab implements specific functionality and connects to the backend services as needed.
