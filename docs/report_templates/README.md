# Report Templates

This directory contains templates used for generating PDF reports in Market Eye AI.

## Templates

- Stock Analysis Report - Template for comprehensive stock analysis reports
- Forecast Report - Template for forecast-focused reports
- CrewAI Analysis Report - Template for CrewAI agent-based analysis reports

## Implementation

Reports are generated using the FPDF library and custom templating:

```python
from forecasting.pdf_report_generator import PDFReportGenerator

# Create a PDF report generator
report_generator = PDFReportGenerator(output_dir="reports")

# Generate a report using templates
report_path = report_generator.create_analytical_report(
    user="username",
    ticker="AAPL",
    analytics_data=analytics_data,
    forecast_data=forecast_data,
    recommendation=recommendation
)
```

## Report Structure

Each report typically includes:

- Header with logo and report title
- Stock information and summary
- Performance analytics and visualizations
- Forecast data and error metrics
- AI-generated investment recommendations
- Date and time of generation
