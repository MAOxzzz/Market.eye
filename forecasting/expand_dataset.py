"""
Utility script to expand the dataset with more generated data for LSTM training.
This is useful when you don't have enough real data to train the models.
"""

import pandas as pd
import numpy as np
import datetime
import os
import argparse

def expand_dataset(input_file, output_file, min_rows_per_ticker=100):
    """
    Expand the dataset to have at least min_rows_per_ticker rows for each ticker.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the expanded dataset
        min_rows_per_ticker (int): Minimum number of rows per ticker
    
    Returns:
        pd.DataFrame: Expanded dataframe
    """
    # Load the dataset (use chunksize for large files)
    try:
        df = pd.read_csv(input_file)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove any rows with NaN dates
        df = df.dropna(subset=['Date'])
        
        print(f"Original dataset has {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Attempting to load in chunks...")
        
        # Try loading in chunks (for very large files)
        chunks = []
        for chunk in pd.read_csv(input_file, chunksize=10000):
            chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
            chunk = chunk.dropna(subset=['Date'])
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print(f"Successfully loaded dataset in chunks. Total rows: {len(df)}")
    
    # Get unique tickers
    tickers = df['Ticker'].unique()
    print(f"Found {len(tickers)} tickers: {', '.join(tickers)}")
    
    # Create a list to store the expanded data
    expanded_data = []
    
    # For each ticker
    for ticker in tickers:
        # Get data for this ticker
        ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
        print(f"{ticker}: {len(ticker_data)} rows")
        
        # If we already have enough data, just add it as is
        if len(ticker_data) >= min_rows_per_ticker:
            expanded_data.append(ticker_data)
            print(f"{ticker} has enough data. No expansion needed.")
            continue
        
        # Otherwise, generate more data
        print(f"Generating more data for {ticker} to reach {min_rows_per_ticker} rows")
        
        # Add the original data
        expanded_data.append(ticker_data)
        
        # How many more rows do we need?
        rows_to_add = min_rows_per_ticker - len(ticker_data)
        
        # Generate new rows by adding dates before the first date
        first_date = ticker_data['Date'].min()
        
        # In case of errors, use the first valid row
        try:
            initial_row = ticker_data.iloc[0]
            recent_price = initial_row['Close']
            avg_volume = ticker_data['Volume'].mean()
        except Exception as e:
            print(f"Error accessing data for {ticker}: {str(e)}")
            print("Using default values")
            recent_price = 100.0  # Default price
            avg_volume = 1000000  # Default volume
        
        new_rows = []
        for i in range(rows_to_add):
            # Generate a new date (going backwards from the first date)
            new_date = first_date - datetime.timedelta(days=i+1)
            
            # Generate a price with small random variation from the first price
            # Use smaller variation the further back we go to avoid big jumps
            variation_pct = np.random.uniform(-0.01, 0.01)  # ±1% daily change
            new_close = recent_price * (1 + variation_pct)
            recent_price = new_close  # Update for next iteration
            
            # Calculate other prices based on the close price
            price_range = new_close * 0.01  # 1% range for the day
            new_open = new_close * (1 - np.random.uniform(-0.005, 0.005))
            new_high = max(new_open, new_close) + np.random.uniform(0, price_range)
            new_low = min(new_open, new_close) - np.random.uniform(0, price_range)
            
            # Use the average volume with some randomness
            new_volume = int(avg_volume * np.random.uniform(0.8, 1.2))
            
            # Create a new row
            new_row = {
                'Date': new_date,
                'Ticker': ticker,
                'Open': round(new_open, 2),
                'High': round(new_high, 2),
                'Low': round(new_low, 2),
                'Close': round(new_close, 2),
                'Volume': new_volume
            }
            
            new_rows.append(new_row)
        
        # Add the new rows to the expanded data
        if new_rows:
            new_data = pd.DataFrame(new_rows)
            expanded_data.append(new_data)
    
    # Combine all the data (in batches to avoid memory issues)
    print("Combining expanded data...")
    combined_df = pd.concat(expanded_data, ignore_index=True)
    
    # Sort by ticker and date
    print("Sorting data...")
    combined_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    # Save the expanded dataset
    print(f"Saving expanded dataset to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print(f"Expanded dataset has {len(combined_df)} rows")
    print(f"Saved to {output_file}")
    
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Expand stock dataset with synthetic data')
    parser.add_argument('--input', default='data/stock_data/Dataset.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/stock_data/expanded_dataset.csv', help='Output CSV file')
    parser.add_argument('--factor', type=int, default=3, help='Expansion factor')
    args = parser.parse_args()
    
    expander = DatasetExpander(args.input)
    expander.expand_dataset(args.factor, args.output)
    print(f"Expanded dataset saved to {args.output}") 