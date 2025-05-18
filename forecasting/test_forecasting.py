"""
Test script for verifying the forecasting components of the Market Eye AI project.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import forecasting components
from forecasting.stock_forecaster import StockForecaster
from forecasting.data_updater import backfill_dataset
from forecasting.forecast_api import ForecastAPI

def test_data_updater():
    """Test the data updater functionality."""
    print("\n=== Testing Data Updater ===")
    
    # Create data directory if it doesn't exist
    os.makedirs('forecasting/data', exist_ok=True)
    
    # Copy the original dataset if it doesn't exist in the data directory
    if not os.path.exists('forecasting/data/test_data.csv'):
        df = pd.read_csv('data/stock_data/Dataset.csv')
        df.to_csv('forecasting/data/test_data.csv', index=False)
        print("Original dataset copied to forecasting/data/test_data.csv")
    
    # Count initial rows
    initial_df = pd.read_csv('forecasting/data/test_data.csv')
    initial_rows = len(initial_df)
    print(f"Initial dataset has {initial_rows} rows")
    
    # Backfill a small amount (just 7 days to keep test fast)
    print("Backfilling dataset with 7 days...")
    backfill_dataset(
        'forecasting/data/test_data.csv',
        end_date=pd.to_datetime(initial_df['Date'].max()) + pd.Timedelta(days=7),
        output_path='forecasting/data/test_data_updated.csv'
    )
    
    # Count updated rows
    updated_df = pd.read_csv('forecasting/data/test_data_updated.csv')
    updated_rows = len(updated_df)
    
    # Verify data was added
    added_rows = updated_rows - initial_rows
    expected_added = len(initial_df['Ticker'].unique()) * 7  # 3 tickers x 7 days
    
    if added_rows == expected_added:
        print(f"✅ Data updater test PASSED - Added {added_rows} rows as expected")
    else:
        print(f"❌ Data updater test FAILED - Expected to add {expected_added} rows, but added {added_rows}")
    
    return updated_df

def test_forecaster(test_ticker='AAPL'):
    """Test the stock forecaster functionality."""
    print("\n=== Testing Stock Forecaster ===")
    
    # Initialize forecaster with test data
    forecaster = StockForecaster('forecasting/data/test_data_updated.csv', 
                                model_dir='forecasting/test_models')
    
    # Create test models directory
    os.makedirs('forecasting/test_models', exist_ok=True)
    
    # Load data
    print("Loading and cleaning data...")
    data = forecaster.load_data()
    
    # Check data loaded properly
    if len(data) > 0:
        print(f"✅ Data loaded successfully with {len(data)} rows")
    else:
        print("❌ Failed to load data")
        return False
    
    # Train a model with just a few epochs for testing
    print(f"Training model for {test_ticker} (with reduced epochs for testing)...")
    start_time = time.time()
    history = forecaster.train_model(test_ticker, epochs=2)
    training_time = time.time() - start_time
    
    # Check model was created
    model_path = os.path.join(forecaster.model_dir, f'{test_ticker}_model.h5')
    scaler_path = os.path.join(forecaster.model_dir, f'{test_ticker}_scaler.pkl')
    
    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)
    
    if model_exists and scaler_exists:
        print(f"✅ Model training PASSED - Created model files in {training_time:.2f} seconds")
    else:
        print(f"❌ Model training FAILED - Missing {'model' if not model_exists else 'scaler'} file")
        return False
    
    # Evaluate model
    print("Evaluating model...")
    metrics = forecaster.evaluate_model(test_ticker)
    
    # Check evaluation produced metrics
    if metrics and 'rmse' in metrics:
        print(f"✅ Model evaluation PASSED - RMSE: {metrics['rmse']:.4f}")
    else:
        print("❌ Model evaluation FAILED - Could not get metrics")
        return False
    
    # Test forecasting
    print("Generating forecast...")
    forecast = forecaster.forecast_future(test_ticker, days=7)
    
    # Check forecast has expected columns and length
    if len(forecast) == 7 and 'Predicted_Close' in forecast.columns:
        print(f"✅ Forecasting PASSED - Generated {len(forecast)} days of predictions")
    else:
        print(f"❌ Forecasting FAILED - Expected 7 days, got {len(forecast)}")
        return False
    
    return True

def test_forecast_api(test_ticker='AAPL'):
    """Test the forecast API functionality."""
    print("\n=== Testing Forecast API ===")
    
    # Initialize API
    api = ForecastAPI(model_dir='forecasting/test_models', 
                     data_dir='forecasting/data')
    
    # Test available tickers
    tickers = api.get_available_tickers()
    if test_ticker in tickers:
        print(f"✅ API initialization PASSED - Found {len(tickers)} available ticker(s)")
    else:
        print(f"❌ API initialization FAILED - Could not find {test_ticker} in available tickers")
        return False
    
    # Test forecast generation
    forecast = api.get_forecast(test_ticker, days=5)
    if forecast is not None and len(forecast) == 5:
        print(f"✅ API forecast PASSED - Generated {len(forecast)} days of predictions")
    else:
        print("❌ API forecast FAILED - Could not generate forecast")
        return False
    
    # Test plotting
    fig = api.plot_forecast(test_ticker, days=5)
    if fig is not None:
        print("✅ API plotting PASSED - Generated forecast plot")
        plt.close(fig)
    else:
        print("❌ API plotting FAILED - Could not generate plot")
        return False
    
    return True

def test_data_loading():
    """Test that we can load a dataset for processing."""
    # Load the sample dataset
    df = pd.read_csv('data/stock_data/Dataset.csv')
    
    # Check that the data is loaded properly
    assert not df.empty, "Dataset should not be empty"
    assert 'Date' in df.columns, "Dataset should have Date column"
    assert 'Ticker' in df.columns, "Dataset should have Ticker column"
    assert 'Close' in df.columns, "Dataset should have Close column"

if __name__ == "__main__":
    print("=== Running Forecasting Component Tests ===")
    
    # Run data updater test
    updated_df = test_data_updater()
    
    # Run forecaster test
    forecaster_success = test_forecaster()
    
    # Run API test if forecaster was successful
    if forecaster_success:
        api_success = test_forecast_api()
    else:
        print("Skipping API tests due to forecaster test failure")
        api_success = False
    
    # Report overall status
    print("\n=== Test Summary ===")
    if forecaster_success and api_success:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED - See details above") 