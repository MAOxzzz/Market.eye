import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import keras
from keras import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import os
import datetime
import pickle

class LSTMForecaster:
    def __init__(self, data_path, model_dir='forecasting/models'):
        """
        Initialize the LSTM forecaster.
        
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
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce', utc=True)
        self.df['Close'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['Volume'] = pd.to_numeric(self.df['Volume'], errors='coerce')
        
        # Drop rows with missing values
        self.df.dropna(subset=['Date', 'Close', 'Volume'], inplace=True)
        
        # Sort by ticker and date
        self.df.sort_values(by=['Ticker', 'Date'], inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        return self.df
    
    def create_sequences(self, data, seq_length):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def prepare_data(self, ticker, test_split=0.2):
        """
        Prepare the data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            test_split (float): Proportion of data to use for testing
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        # Filter data for the specified ticker
        df_ticker = self.df[self.df['Ticker'] == ticker].sort_values('Date')
        
        # Check if we have enough data
        if len(df_ticker) <= self.sequence_length:
            raise ValueError(f"Not enough data for {ticker}. Need at least {self.sequence_length+1} data points.")
        
        # Get prices and reshape
        prices = df_ticker['Close'].values.reshape(-1, 1)
        
        # Normalize prices
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)
        
        # Create sequences
        X, y = self.create_sequences(scaled_prices, self.sequence_length)
        
        # Split data
        split = int((1 - test_split) * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Save scaler for later use
        self.scalers[ticker] = scaler
        
        return X_train, y_train, X_test, y_test, scaler
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train_model(self, ticker, epochs=50, batch_size=32):
        """
        Train an LSTM model for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training information
        """
        # Prepare data
        X_train, y_train, X_test, y_test, scaler = self.prepare_data(ticker)
        
        # Build and train LSTM model
        model = self.build_lstm_model(input_shape=(X_train.shape[1], 1))
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
        model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Store model
        self.models[ticker] = model
        
        return {
            "message": f"LSTM model trained for {ticker}",
            "epochs_trained": len(history.history['loss']),
            "final_loss": history.history['loss'][-1],
            "final_val_loss": history.history['val_loss'][-1]
        }
    
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
            model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
            self.models[ticker] = load_model(model_path)
        
        # Make predictions
        y_pred = self.models[ticker].predict(X_test)
        
        # Inverse transform predictions and actual values
        y_test_orig = scaler.inverse_transform(y_test)
        y_pred_orig = scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        # Calculate percentage error
        percentage_error = np.mean(np.abs(y_pred_orig - y_test_orig) / y_test_orig) * 100
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_orig, label='True Prices')
        plt.plot(y_pred_orig, label='Predicted Prices')
        plt.title(f"{ticker} - True vs Predicted Prices")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f'{ticker}_evaluation.png')
        plt.savefig(plot_path)
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'percentage_error': percentage_error,
            'plot_path': plot_path
        }
    
    def forecast_january_2025(self, ticker):
        """
        Forecast prices for January 2025.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: DataFrame with predicted prices for January 2025
        """
        # Load model if not in memory
        if ticker not in self.models:
            model_path = os.path.join(self.model_dir, f'{ticker}_lstm_model.h5')
            self.models[ticker] = load_model(model_path)
        
        # Get the last sequence_length days of data
        df_ticker = self.df[self.df['Ticker'] == ticker].sort_values('Date')
        last_sequence = df_ticker['Close'].values[-self.sequence_length:].reshape(-1, 1)
        
        # Scale the data
        if ticker not in self.scalers:
            scaler_path = os.path.join(self.model_dir, f'{ticker}_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scalers[ticker] = pickle.load(f)
        
        scaled_sequence = self.scalers[ticker].transform(last_sequence)
        
        # Generate January 2025 dates
        last_date = df_ticker['Date'].iloc[-1]
        jan_2025_dates = pd.date_range(
            start='2025-01-01',
            end='2025-01-31',
            freq='B'  # Business days
        )
        
        # Make predictions
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(len(jan_2025_dates)):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            pred = self.models[ticker].predict(X, verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scalers[ticker].inverse_transform(predictions)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': jan_2025_dates,
            'Predicted_Price': predictions.flatten()
        })
        
        # Save predictions
        results.to_csv(os.path.join(self.model_dir, f'{ticker}_jan_2025_forecast.csv'), index=False)
        
        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(df_ticker['Date'][-30:], df_ticker['Close'][-30:], label='Historical Prices')
        plt.plot(results['Date'], results['Predicted_Price'], label='January 2025 Forecast')
        plt.title(f"{ticker} - January 2025 Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        
        # Save plot
        plot_path = os.path.join(self.model_dir, f'{ticker}_jan_2025_forecast.png')
        plt.savefig(plot_path)
        plt.close()
        
        return results 