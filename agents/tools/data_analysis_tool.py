import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalysisTool:
    """Tool for analyzing stock data and generating metrics."""
    
    def __init__(self, output_dir="data/analytics"):
        """
        Initialize the data analysis tool.
        
        Args:
            output_dir (str): Directory to save analysis outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Map ticker symbols to company names and sectors
        self.company_info = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
            'GOOGL': {'name': 'Alphabet Inc.', 'sector': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Technology'},
            'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
            'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Finance'},
            'BAC': {'name': 'Bank of America Corp.', 'sector': 'Finance'},
            'WFC': {'name': 'Wells Fargo & Co.', 'sector': 'Finance'},
            'GS': {'name': 'Goldman Sachs Group Inc.', 'sector': 'Finance'},
            'NKE': {'name': 'Nike Inc.', 'sector': 'Sportswear'},
            'UAA': {'name': 'Under Armour Inc.', 'sector': 'Sportswear'},
            'ADDYY': {'name': 'Adidas AG', 'sector': 'Sportswear'},
            'LULU': {'name': 'Lululemon Athletica Inc.', 'sector': 'Sportswear'}
        }
        
        # Model parameters
        self.sequence_length = 30
        
    def calculate_metrics(self, df):
        """
        Calculate key metrics for stock data.
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            dict: Dictionary with metrics per ticker
        """
        logger.info("Calculating stock metrics...")
        results = {}
        
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker]
            
            # Calculate highest and lowest prices
            highest_price = ticker_data['High'].max()
            lowest_price = ticker_data['Low'].min()
            
            # Calculate annual growth for 2020 if available
            growth_2020 = self._calculate_annual_growth(ticker_data, year=2020)
            
            # Store results
            results[ticker] = {
                'highest_price': highest_price,
                'lowest_price': lowest_price,
                'annual_growth_2020': growth_2020,
                'company_name': self.company_info.get(ticker, {}).get('name', 'Unknown'),
                'sector': self.company_info.get(ticker, {}).get('sector', 'Unknown')
            }
            
        logger.info(f"Metrics calculated for {len(results)} tickers")
        return results
    
    def _calculate_annual_growth(self, ticker_data, year=2020):
        """Calculate annual growth percentage for a given year."""
        try:
            # Check if Date column is datetime type
            if 'Date' not in ticker_data.columns:
                logger.warning("Date column not found in ticker data")
                return None
                
            # Make sure Date is datetime type
            if not pd.api.types.is_datetime64_any_dtype(ticker_data['Date']):
                logger.warning("Date column is not in datetime format, attempting conversion")
                ticker_data = ticker_data.copy()
                ticker_data['Date'] = pd.to_datetime(ticker_data['Date'], errors='coerce')
                ticker_data = ticker_data.dropna(subset=['Date'])
                
                if len(ticker_data) == 0:
                    logger.warning("No valid dates after conversion")
                    return None
            
            # Filter data for the specified year
            year_data = ticker_data[ticker_data['Date'].dt.year == year]
            
            if len(year_data) < 2:
                logger.info(f"Not enough data points for year {year}, found {len(year_data)}")
                return None
            
            # Get first and last closing prices of the year
            first_close = year_data.iloc[0]['Close']
            last_close = year_data.iloc[-1]['Close']
            
            # Calculate growth percentage
            growth_pct = ((last_close - first_close) / first_close) * 100
            return growth_pct
        except Exception as e:
            logger.error(f"Error calculating annual growth: {e}")
            return None
    
    def compare_sectors(self, df):
        """
        Compare performance across sectors.
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            dict: Sector comparison results
        """
        logger.info("Performing sector comparison...")
        sectors = {}
        
        # Group tickers by sector
        for ticker in df['Ticker'].unique():
            sector = self.company_info.get(ticker, {}).get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)
        
        results = {}
        for sector, tickers in sectors.items():
            sector_data = df[df['Ticker'].isin(tickers)]
            
            # Skip if no data for this sector
            if len(sector_data) == 0:
                continue
                
            # Calculate sector metrics
            avg_close = sector_data.groupby('Date')['Close'].mean()
            
            # Calculate sector growth over entire period
            if not avg_close.empty:
                first_close = avg_close.iloc[0]
                last_close = avg_close.iloc[-1]
                growth_pct = ((last_close - first_close) / first_close) * 100
            else:
                growth_pct = None
            
            # Store results
            results[sector] = {
                'tickers': tickers,
                'average_close_latest': avg_close.iloc[-1] if not avg_close.empty else None,
                'overall_growth_pct': growth_pct,
                'volatility': sector_data.groupby('Ticker')['Close'].std().mean()
            }
        
        logger.info(f"Comparison completed for {len(results)} sectors")
        return results
    
    def prepare_lstm_data(self, ticker_data, sequence_length=30):
        """Prepare data for LinearRegression model."""
        close_prices = ticker_data['Close'].values.reshape(-1, 1)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        
        # Create sequences - for linear regression we'll use a simpler approach
        X = np.array(range(len(scaled_data))).reshape(-1, 1)
        y = scaled_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test, scaler
    
    def build_lstm_model(self, input_shape):
        """Build a linear regression model for stock price prediction."""
        model = LinearRegression()
        return model
    
    def train_forecasting_model(self, df, ticker, epochs=20, batch_size=32):
        """
        Train a Linear Regression model for stock price forecasting.
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Ticker symbol to forecast
            epochs (int): Not used for LinearRegression, kept for API compatibility
            batch_size (int): Not used for LinearRegression, kept for API compatibility
            
        Returns:
            dict: Training results and evaluation metrics
        """
        logger.info(f"Training forecasting model for {ticker}")
        
        # Check for existing model
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        model_path = os.path.join(self.output_dir, 'models', f"{ticker}_model.pkl")
        
        # Get ticker data
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) < self.sequence_length + 10:
            logger.warning(f"Not enough data for {ticker} to train a model")
            return {
                'status': 'error',
                'message': f"Not enough data for {ticker} to train a model"
            }
        
        # Prepare data
        X_train, y_train, X_test, y_test, scaler = self.prepare_lstm_data(ticker_data, self.sequence_length)
        
        # Build model
        model = self.build_lstm_model((self.sequence_length, 1))
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Model evaluation - MSE: {mse}, RMSE: {rmse}")
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)
        
        # Return results
        return {
            'status': 'success',
            'ticker': ticker,
            'mse': mse,
            'rmse': rmse,
            'model_path': model_path,
            'training_history': {
                'loss': None,
                'val_loss': None
            },
            'scaler': scaler  # Save for future predictions
        }
    
    def forecast_future(self, df, ticker, days=30):
        """
        Generate price forecasts for a ticker.
        
        Args:
            df (pd.DataFrame): Stock data
            ticker (str): Ticker symbol to forecast
            days (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: Forecast data with dates and predicted prices
        """
        logger.info(f"Generating {days}-day forecast for {ticker}")
        
        # Get ticker data
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        
        if len(ticker_data) == 0:
            logger.warning(f"No data available for ticker {ticker}")
            return None
        
        # Load or train model
        model_path = os.path.join(self.output_dir, 'models', f"{ticker}_model.pkl")
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                model = model_data['model']
                scaler = model_data['scaler']
                logger.info(f"Loaded existing model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                # Train a new model if loading fails
                results = self.train_forecasting_model(df, ticker)
                if results.get('status') == 'error':
                    return None
                
                with open(results['model_path'], 'rb') as f:
                    model_data = pickle.load(f)
                model = model_data['model']
                scaler = model_data['scaler']
        else:
            # Train a new model
            results = self.train_forecasting_model(df, ticker)
            if results.get('status') == 'error':
                return None
            
            with open(results['model_path'], 'rb') as f:
                model_data = pickle.load(f)
            model = model_data['model']
            scaler = model_data['scaler']
        
        # Generate future dates
        last_date = ticker_data['Date'].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        # Prepare prediction inputs for LinearRegression
        last_idx = len(ticker_data) - 1
        future_x = np.array(range(last_idx + 1, last_idx + days + 1)).reshape(-1, 1)
        
        # Make predictions
        future_predictions_scaled = model.predict(future_x)
        
        # Convert predictions back to original scale
        future_predictions = scaler.inverse_transform(future_predictions_scaled)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions.flatten(),
            'Ticker': ticker
        })
        
        logger.info(f"Generated {len(forecast_df)} days of forecasts for {ticker}")
        return forecast_df
    
    def calculate_january_2025_metrics(self, predictions_df, actual_df=None):
        """
        Calculate metrics for January 2025 forecasts vs. actual data when available.
        
        Args:
            predictions_df (pd.DataFrame): Dataframe with predictions
            actual_df (pd.DataFrame): Dataframe with actual data (if available)
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            # Ensure Date column is datetime type
            if 'Date' not in predictions_df.columns:
                logger.error("Date column not found in predictions dataframe")
                return {'error': 'Date column not found in predictions dataframe'}
                
            # Convert if necessary
            if not pd.api.types.is_datetime64_any_dtype(predictions_df['Date']):
                predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], errors='coerce')
                predictions_df = predictions_df.dropna(subset=['Date'])
            
            # Filter for January 2025
            jan_2025_predictions = predictions_df[
                (predictions_df['Date'].dt.year == 2025) & 
                (predictions_df['Date'].dt.month == 1)
            ]
            
            results = {
                'predictions': jan_2025_predictions.to_dict(orient='records')
            }
            
            # If actual data is available, calculate error metrics
            if actual_df is not None:
                # Ensure Date column is datetime type in actual_df
                if 'Date' not in actual_df.columns:
                    logger.error("Date column not found in actual dataframe")
                    return {'error': 'Date column not found in actual dataframe'}
                    
                # Convert if necessary
                if not pd.api.types.is_datetime64_any_dtype(actual_df['Date']):
                    actual_df['Date'] = pd.to_datetime(actual_df['Date'], errors='coerce')
                    actual_df = actual_df.dropna(subset=['Date'])
                
                jan_2025_actual = actual_df[
                    (actual_df['Date'].dt.year == 2025) & 
                    (actual_df['Date'].dt.month == 1)
                ]
                
                # Merge predictions with actual data
                comparison = pd.merge(
                    jan_2025_predictions,
                    jan_2025_actual[['Date', 'Ticker', 'Close']],
                    on=['Date', 'Ticker'],
                    how='inner'
                )
                
                if not comparison.empty:
                    # Calculate metrics
                    mse = mean_squared_error(comparison['Close'], comparison['Predicted_Close'])
                    rmse = np.sqrt(mse)
                    
                    results.update({
                        'mse': mse,
                        'rmse': rmse,
                        'comparison': comparison.to_dict(orient='records')
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e)}

    def generate_analysis_report(self, df, tickers=None):
        """
        Generate a comprehensive analysis report for the specified tickers.
        
        Args:
            df (pd.DataFrame): Stock data
            tickers (list): List of ticker symbols to analyze
            
        Returns:
            dict: Comprehensive analysis report
        """
        # If no tickers specified, use all available
        if not tickers:
            tickers = df['Ticker'].unique().tolist()
            
        logger.info(f"Generating analysis report for {len(tickers)} tickers...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(df)
        
        # Compare sectors
        sector_comparison = self.compare_sectors(df)
        
        # Generate forecasts
        forecasts = {}
        for ticker in tickers:
            forecast_df = self.forecast_future(df, ticker)
            if forecast_df is not None:
                forecasts[ticker] = forecast_df
        
        # Prepare report
        report = {
            'timestamp': datetime.now().isoformat(),
            'tickers_analyzed': tickers,
            'metrics': metrics,
            'sector_comparison': sector_comparison,
            'forecasts': {t: f.to_dict(orient='records') for t, f in forecasts.items()}
        }
        
        # Save report
        report_path = os.path.join(self.output_dir, 'analysis_report.json')
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Analysis report saved to {report_path}")
        return report


if __name__ == "__main__":
    # Example usage
    from agents.tools.kaggle_tool import KaggleTool
    
    try:
        # Load data
        kaggle_tool = KaggleTool()
        stock_data = kaggle_tool.load_stock_data(tickers=['AAPL', 'MSFT', 'GOOGL', 'JPM', 'NKE'])
        
        # Analyze data
        analysis_tool = DataAnalysisTool()
        metrics = analysis_tool.calculate_metrics(stock_data)
        
        print("Metrics:")
        for ticker, values in metrics.items():
            print(f"{ticker}: Highest={values['highest_price']:.2f}, Lowest={values['lowest_price']:.2f}, 2020 Growth={values.get('annual_growth_2020', 'N/A')}")
        
        # Compare sectors
        sectors = analysis_tool.compare_sectors(stock_data)
        print("\nSector Comparison:")
        for sector, values in sectors.items():
            print(f"{sector}: Overall Growth={values.get('overall_growth_pct', 'N/A')}, Volatility={values.get('volatility', 'N/A')}")
        
        # Generate forecasts for Apple
        apple_forecast = analysis_tool.forecast_future(stock_data, 'AAPL', days=5)
        if apple_forecast is not None:
            print("\nApple 5-day Forecast:")
            print(apple_forecast)
            
    except Exception as e:
        print(f"Error: {e}") 