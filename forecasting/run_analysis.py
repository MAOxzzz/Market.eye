import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting.stock_analyzer import StockAnalyzer
from forecasting.lstm_forecaster import LSTMForecaster
import pandas as pd
from datetime import datetime

def main():
    # Initialize paths
    data_path = 'data/stock_data/test_data.csv'  # Using test data by default
    output_dir = 'forecasting/analysis'
    model_dir = 'forecasting/models'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if test data exists, if not create it
    if not os.path.exists(data_path):
        print("Test data not found. Creating sample data...")
        from test_forecasting import create_test_data
        create_test_data()
    
    # Initialize analyzer and forecaster
    analyzer = StockAnalyzer(data_path, output_dir)
    forecaster = LSTMForecaster(data_path, model_dir)
    
    # Load data
    print("Loading data...")
    analyzer.load_data()
    forecaster.load_data()
    
    # Define tickers to analyze (use a smaller set for testing)
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Generate analytics
    print("\nGenerating analytics...")
    company_comparison = analyzer.generate_cross_company_comparison(tickers)
    sector_comparison = analyzer.generate_sector_comparison()
    analyzer.plot_comparison_charts(tickers)
    
    # Save analytics
    company_comparison.to_csv(os.path.join(output_dir, 'company_comparison.csv'))
    sector_comparison.to_csv(os.path.join(output_dir, 'sector_comparison.csv'))
    
    # Train and evaluate models
    print("\nTraining LSTM models...")
    results = []
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        
        # Train model with fewer epochs for testing
        training_info = forecaster.train_model(ticker, epochs=10, batch_size=32)
        print(f"Training completed: {training_info['message']}")
        
        # Evaluate model
        evaluation = forecaster.evaluate_model(ticker)
        print(f"Evaluation metrics:")
        print(f"MSE: {evaluation['mse']:.2f}")
        print(f"RMSE: {evaluation['rmse']:.2f}")
        print(f"MAE: {evaluation['mae']:.2f}")
        print(f"Percentage Error: {evaluation['percentage_error']:.2f}%")
        
        # Generate January 2025 forecast
        forecast = forecaster.forecast_january_2025(ticker)
        print(f"January 2025 forecast generated and saved")
        
        # Store results
        results.append({
            'ticker': ticker,
            'mse': evaluation['mse'],
            'rmse': evaluation['rmse'],
            'mae': evaluation['mae'],
            'percentage_error': evaluation['percentage_error'],
            'forecast_path': os.path.join(model_dir, f'{ticker}_jan_2025_forecast.csv'),
            'evaluation_plot': evaluation['plot_path'],
            'forecast_plot': os.path.join(model_dir, f'{ticker}_jan_2025_forecast.png')
        })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'forecasting_results.csv'), index=False)
    
    print("\nAnalysis and forecasting completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Models and forecasts saved in: {model_dir}")

if __name__ == "__main__":
    main() 