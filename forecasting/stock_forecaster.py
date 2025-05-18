import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import os
import datetime
import pickle


class StockForecaster:
    def __init__(self, data_path, model_dir='forecasting/models'):
        """
        Initialize the stock forecaster.
        
        Args:
            data_path (str): Path to the stock data CSV file
            model_dir (str): Directory to save the trained models
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.df = None
        self.models = {}
        self.scalers = {}
        self.sequence_length = 30
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def load_data(self):
        """Load and clean the stock data."""
        self.df = pd.read_csv(self.data_path)
        
        # Clean data
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')
        
        # Drop rows with missing values
        self.df.dropna(subset=['Date', 'Close', 'Volume'], inplace=True)
        
        # Sort by ticker and date
        self.df.sort_values(by=['Ticker', 'Date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # Filter only desired tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.df = self.df[self.df['Ticker'].isin(tickers)]
        
        return self.df
    
    def prepare_data(self, ticker, test_split=0.2):
        """
        Prepare the data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            test_split (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        # Filter data for the specified ticker
        df_ticker = self.df[self.df['Ticker'] == ticker].sort_values('Date')
        
        # Check if we have enough data
        if len(df_ticker) <= self.sequence_length:
            raise ValueError(f"Not enough data for {ticker}. Need at least {self.sequence_length+1} data points, but only found {len(df_ticker)}.")
        
        # Get prices and reshape
        prices = df_ticker['Close'].values.reshape(-1, 1)
        
        # Normalize prices
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences for simple regression
        X = np.array(range(len(scaled_prices))).reshape(-1, 1)
        y = scaled_prices
        
        # Split data
        split = int((1 - test_split) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Save scaler for later use
        self.scalers[ticker] = scaler
        
        return X_train, y_train, X_test, y_test, scaler
    
    def train_model(self, ticker, epochs=None, batch_size=None):
        """
        Train a model for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            epochs: Not used in linear regression (kept for API compatibility)
            batch_size: Not used in linear regression (kept for API compatibility)
            
        Returns:
            dict: Training information
        """
        # Prepare data
        X_train, y_train, X_test, y_test, scaler = self.prepare_data(ticker)
        
        # Build and train simple linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{ticker}_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store model
        self.models[ticker] = model
        
        return {"message": f"Linear regression model trained for {ticker}"}
    
    def evaluate_model(self, ticker):
        """
        Evaluate the model for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Evaluation metrics
        """
        # Get test data
        _, _, X_test, y_test, scaler = self.prepare_data(ticker)
        
        # Load model if not in memory
        if ticker not in self.models:
            model_path = os.path.join(self.model_dir, f'{ticker}_model.pkl')
            with open(model_path, 'rb') as f:
                self.models[ticker] = pickle.load(f)
        
        # Make predictions
        y_pred = self.models[ticker].predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Custom accuracy (predictions within 5% of true values)
        tolerance = 0.05
        accuracy_like = np.mean(np.abs(y_pred.flatten() - y_test.flatten()) / y_test.flatten() < tolerance)
        
        # Plot results
        plt.figure(figsize=(10, 5))
        plt.plot(y_test, label='True Prices')
        plt.plot(y_pred, label='Predicted Prices')
        plt.title(f"{ticker} - True vs Predicted Prices")
        plt.xlabel("Time")
        plt.ylabel("Normalized Price")
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f'{ticker}_evaluation.png')
        plt.savefig(plot_path)
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_5pct': accuracy_like * 100,
            'plot_path': plot_path
        }
    
    def forecast_future(self, ticker, days=30):
        """
        Forecast future prices for a ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            days (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: DataFrame with predicted prices
        """
        # Filter data for the ticker
        df_ticker = self.df[self.df['Ticker'] == ticker].sort_values('Date')
        
        # Get all prices
        prices = df_ticker['Close'].values.reshape(-1, 1)
        
        # Load scaler if not in memory
        if ticker not in self.scalers:
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scalers[ticker] = pickle.load(f)
        
        # Scale prices
        scaler = self.scalers[ticker]
        scaled_prices = scaler.transform(prices)
        
        # Load model if not in memory
        if ticker not in self.models:
            model_path = os.path.join(self.model_dir, f'{ticker}_model.pkl')
            with open(model_path, 'rb') as f:
                self.models[ticker] = pickle.load(f)
        
        # Forecast future dates
        X_future = np.array(range(len(prices), len(prices) + days)).reshape(-1, 1)
        future_scaled = self.models[ticker].predict(X_future)
        
        # Convert back to original scale
        future_prices = scaler.inverse_transform(future_scaled)
        
        # Generate future dates
        last_date = df_ticker['Date'].iloc[-1]
        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(days)]
        
        # Create DataFrame with predictions
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Ticker': ticker,
            'Predicted_Close': future_prices.flatten()
        })
        
        return forecast_df
    
    def generate_prediction_report(self, tickers=None):
        """
        Generate a report with predictions for all or specified tickers.
        
        Args:
            tickers (list): List of tickers to include in the report
            
        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        all_predictions = []
        
        for ticker in tickers:
            # Make sure model exists
            model_path = os.path.join(self.model_dir, f'{ticker}_model.pkl')
            if not os.path.exists(model_path):
                print(f"No model found for {ticker}. Training now...")
                self.train_model(ticker)
            
            # Get forecast for next 30 days
            forecast = self.forecast_future(ticker, days=30)
            all_predictions.append(forecast)
        
        # Combine predictions
        predictions_df = pd.concat(all_predictions)
        
        # Format dates
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date']).dt.strftime('%Y-%m-%d')
        
        return predictions_df


if __name__ == "__main__":
    # Example usage
    forecaster = StockForecaster('data/stock_data/Dataset.csv')
    forecaster.load_data()
    
    # Train models for each ticker
    for ticker in ['AAPL', 'MSFT', 'GOOGL']:
        print(f"Training model for {ticker}...")
        forecaster.train_model(ticker)
        
        print(f"Evaluating model for {ticker}...")
        metrics = forecaster.evaluate_model(ticker)
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
    # Generate forecast
    predictions = forecaster.generate_prediction_report()
    print(predictions.head())
    
    # Save predictions to CSV
    predictions.to_csv('forecasting/data/stock_predictions.csv', index=False)
    print("Predictions saved to: forecasting/data/stock_predictions.csv") 