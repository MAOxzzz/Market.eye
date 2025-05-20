# PDF Report Generator for Market Eye AI

This module provides comprehensive PDF report generation capabilities for the Market Eye AI system. It fulfills Functional Requirement #6, creating well-formatted reports with analytical tables, figures, forecast comparisons, error metrics, and LLM-generated insights.

## Features

The PDF Report Generator includes:

- **Analytical Tables**: Company and sector comparison tables with key metrics
- **Visualizations**: Historical vs. forecast price charts
- **Forecast vs. Real-Data Comparisons**: Side-by-side performance evaluation
- **Error Metrics**: MSE, RMSE, R², and accuracy metrics with visualizations
- **LLM-Generated Text**: AI-powered recommendations and insights
- **Professional Formatting**: Well-structured and properly formatted reports

## Usage

### Basic Usage

```python
from forecasting.pdf_report_generator import PDFReportGenerator

# Initialize the report generator
report_generator = PDFReportGenerator(output_dir="reports")

# Generate a report with whatever data you have available
report_path = report_generator.create_analytical_report(
    user="username",
    ticker="AAPL",
    recommendation="BUY - Apple shows strong growth potential...",
    analytics_data=analytics_data,  # Optional
    forecast_data=forecast_data,    # Optional
    historical_data=historical_data,  # Optional
    error_metrics=error_metrics,    # Optional
    llm_insights=llm_insights       # Optional
)

# Get an HTML download link for the report
download_link = report_generator.get_download_link(report_path)
```

### Data Structures

#### Analytics Data

```python
analytics_data = {
    'company_comparison': pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'highest_price': [200.0, 180.0, 150.0],
        'lowest_price': [150.0, 120.0, 100.0],
        'annual_growth': [15.2, 12.5, 10.8],
        'volatility': [20.5, 18.2, 22.1],
        'current_price': [180.5, 170.2, 140.3]
    }),
    'sector_comparison': pd.DataFrame({
        'avg_annual_growth': [12.8, 8.5, 10.2],
        'avg_volatility': [18.5, 15.2, 20.1],
        'total_market_cap': [1200.5, 800.2, 500.1],
        'num_companies': [10, 15, 8]
    }, index=['Technology', 'Finance', 'Consumer'])
}
```

#### Forecast Data

```python
forecast_data = {
    'ticker': 'AAPL',
    'forecast_df': pd.DataFrame({
        'Date': pd.date_range(start='2023-02-01', periods=10),
        'Predicted_Close': [105.2, 106.3, 107.1, ...]
    }),
    'metrics': {
        'rmse': 2.45,
        'mse': 6.00,
        'mae': 1.95,
        'r2': 0.85,
        'accuracy_5pct': 85.0
    }
}
```

#### Historical Data

```python
historical_data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=30),
    'Close': [100.2, 101.3, 99.8, ...],
    'Ticker': ['AAPL'] * 30
})
```

#### Error Metrics

```python
error_metrics = {
    'mse': 6.00,
    'rmse': 2.45,
    'mae': 1.95,
    'r2': 0.85,
    'accuracy_5pct': 85.0,
    'visualization': 'path/to/visualization.png'  # Optional
}
```

#### LLM Insights

```python
llm_insights = {
    'summary': "Apple Inc. has shown strong performance...",
    'recommendations': "BUY - Apple Inc. shows strong growth potential...",
    'risk_assessment': "Low to moderate risk. Supply chain constraints remain..."
}
```

## Error Metrics Visualization

The module includes a utility function to generate error metrics visualizations:

```python
from forecasting.pdf_report_generator import generate_error_metrics_chart

# Generate error visualization
error_viz_buffer = generate_error_metrics_chart(
    actual=historical_data['Close'].values[-10:],
    predicted=forecast_data['forecast_df']['Predicted_Close'].values[:10],
    title="AAPL - Forecast Error Analysis"
)

# Save to file
with open('error_visualization.png', 'wb') as f:
    f.write(error_viz_buffer.getvalue())

# Add to error metrics
error_metrics['visualization'] = 'error_visualization.png'
```

## Integration with Streamlit

The PDF report generator is fully integrated with the Streamlit interface. Reports can be generated from:

1. The "Stock Analysis" tab
2. The "Forecast Report" tab
3. The "CrewAI Analysis" tab

Each report type includes relevant information based on the analysis performed.

## Implementation Details

The PDF report generator uses:

- **FPDF**: For PDF document creation and formatting
- **Matplotlib**: For chart and visualization generation
- **Pandas**: For data manipulation and table creation

The generated reports are saved in the "reports" directory and provided to users as downloadable links.
