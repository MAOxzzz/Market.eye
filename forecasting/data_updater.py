import pandas as pd
import numpy as np
import datetime
import os
import time
import schedule
import sys
import argparse


def update_dataset(dataset_path, output_path=None):
    """
    Update the stock dataset with new simulated data points.
    
    Args:
        dataset_path (str): Path to the original dataset
        output_path (str): Path to save the updated dataset
    
    Returns:
        pd.DataFrame: Updated dataset
    """
    # If no output path is provided, update the original file
    if output_path is None:
        output_path = dataset_path
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Convert date to datetime with utc=True to handle timezone consistently
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    # No need to check tz anymore since we're explicitly setting utc=True
    # Now localize to None to make timezone-naive
    df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Get the unique tickers
    tickers = df['Ticker'].unique()
    
    # Get the latest date in the dataset
    latest_date = df['Date'].max()
    if latest_date.tzinfo is not None:
        latest_date = latest_date.replace(tzinfo=None)
    
    # Calculate the next date
    next_date = latest_date + datetime.timedelta(days=1)
    today = datetime.datetime.now().date()
    
    # Only update if needed (if the dataset doesn't have today's data)
    if next_date.date() <= today:
        print(f"Updating dataset with new data point for {next_date.strftime('%Y-%m-%d')}")
        
        new_rows = []
        
        for ticker in tickers:
            # Get the latest data for this ticker
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            latest_row = ticker_data.iloc[-1]
            
            # Simulate a new price with some random variation (±3%)
            latest_close = latest_row['Close']
            change_percent = np.random.uniform(-0.03, 0.03)
            new_close = latest_close * (1 + change_percent)
            
            # Calculate other prices based on the close price
            price_range = new_close * 0.02  # 2% range for the day
            new_open = new_close * (1 - np.random.uniform(-0.01, 0.01))
            new_high = max(new_open, new_close) + np.random.uniform(0, price_range)
            new_low = min(new_open, new_close) - np.random.uniform(0, price_range)
            
            # Simulate volume with some randomness based on previous volume
            new_volume = int(latest_row['Volume'] * np.random.uniform(0.7, 1.3))
            
            # Create a new row
            new_row = {
                'Date': next_date,
                'Ticker': ticker,
                'Open': round(new_open, 2),
                'High': round(new_high, 2),
                'Low': round(new_low, 2),
                'Close': round(new_close, 2),
                'Volume': new_volume
            }
            
            new_rows.append(new_row)
        
        # Add the new rows to the dataframe
        new_data = pd.DataFrame(new_rows)
        df = pd.concat([df, new_data], ignore_index=True)
        
        # Sort the dataframe
        df.sort_values(by=['Ticker', 'Date'], inplace=True)
        
        # Save the updated dataset
        df.to_csv(output_path, index=False)
        
        print(f"Dataset updated with {len(new_rows)} new data points.")
        return df
    else:
        print("Dataset is already up-to-date.")
        return df


def setup_daily_update(dataset_path, output_path=None, update_time="00:00", end_date=None):
    """
    Schedule a daily update for the dataset.
    
    Args:
        dataset_path (str): Path to the original dataset
        output_path (str): Path to save the updated dataset
        update_time (str): Time to perform the update (24-hour format)
        end_date (str): Date to stop updates (format: 'YYYY-MM-DD')
    """
    # Parse end date if provided
    if end_date:
        end_date = pd.to_datetime(end_date, utc=True)
    else:
        # Default to December 31, 2024
        end_date = pd.to_datetime("2024-12-31", utc=True)
        
    def job():
        current_date = datetime.datetime.now()
        print(f"Running scheduled update at {current_date}")
        
        # Check if we've reached the end date
        if end_date and current_date.date() > end_date.date():
            print(f"Reached end date ({end_date.strftime('%Y-%m-%d')}). Stopping scheduled updates.")
            return schedule.CancelJob
            
        update_dataset(dataset_path, output_path)
    
    # Schedule the job
    schedule.every().day.at(update_time).do(job)
    
    print(f"Daily update scheduled at {update_time} until {end_date.strftime('%Y-%m-%d')}")
    
    # Run the job once immediately
    job()
    
    # Keep the script running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("Update scheduler stopped.")


def backfill_dataset(dataset_path, end_date=None, output_path=None):
    """
    Backfill the dataset with simulated data up to the specified end date.
    
    Args:
        dataset_path (str): Path to the original dataset
        end_date (str): End date for backfilling (format: 'YYYY-MM-DD')
        output_path (str): Path to save the updated dataset
    
    Returns:
        pd.DataFrame: Updated dataset
    """
    # If no output path is provided, update the original file
    if output_path is None:
        output_path = dataset_path
    
    # If no end date is provided, use today's date
    if end_date is None:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Parse the end date and ensure it's timezone-naive
    end_date = pd.to_datetime(end_date, utc=True)
    if end_date.tzinfo is not None:
        end_date = end_date.replace(tzinfo=None)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Convert Date to datetime with utc=True to handle timezone consistently
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    # No need to check tz anymore since we're explicitly setting utc=True
    # Now localize to None to make timezone-naive
    df['Date'] = df['Date'].dt.tz_localize(None)
    
    # Get the latest date in the dataset
    latest_date = df['Date'].max()
    if latest_date.tzinfo is not None:
        latest_date = latest_date.replace(tzinfo=None)
    
    # If already up to date, return
    if latest_date >= end_date:
        print(f"Dataset already contains data up to {end_date}.")
        return df
    
    # Generate dates to add
    current_date = latest_date
    dates_to_add = []
    while current_date < end_date:
        current_date += datetime.timedelta(days=1)
        dates_to_add.append(current_date)
    
    print(f"Backfilling dataset with {len(dates_to_add)} days of data.")
    
    # Create a temporary copy to simulate updates
    temp_df = df.copy()
    
    # For each date to add
    for next_date in dates_to_add:
        new_rows = []
        
        for ticker in df['Ticker'].unique():
            # Get the latest data for this ticker
            ticker_data = temp_df[temp_df['Ticker'] == ticker].sort_values('Date')
            latest_row = ticker_data.iloc[-1]
            
            # Simulate a new price
            latest_close = latest_row['Close']
            change_percent = np.random.uniform(-0.03, 0.03)  # ±3% daily change
            new_close = latest_close * (1 + change_percent)
            
            # Calculate other prices
            price_range = new_close * 0.02  # 2% range for the day
            new_open = new_close * (1 - np.random.uniform(-0.01, 0.01))
            new_high = max(new_open, new_close) + np.random.uniform(0, price_range)
            new_low = min(new_open, new_close) - np.random.uniform(0, price_range)
            
            # Simulate volume
            new_volume = int(latest_row['Volume'] * np.random.uniform(0.7, 1.3))
            
            # Create a new row
            new_row = {
                'Date': next_date,
                'Ticker': ticker,
                'Open': round(new_open, 2),
                'High': round(new_high, 2),
                'Low': round(new_low, 2),
                'Close': round(new_close, 2),
                'Volume': new_volume
            }
            
            new_rows.append(new_row)
        
        # Add the new rows to the temporary dataframe
        new_data = pd.DataFrame(new_rows)
        temp_df = pd.concat([temp_df, new_data], ignore_index=True)
    
    # Update the original dataframe
    df = temp_df.copy()
    
    # Sort the dataframe
    df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    # Save the updated dataset
    df.to_csv(output_path, index=False)
    
    print(f"Dataset updated with data up to {end_date.strftime('%Y-%m-%d')}.")
    return df


def main():
    """Main function to handle command line arguments and run the appropriate functions."""
    parser = argparse.ArgumentParser(description='Stock Market Dataset Updater')
    
    # Define command line arguments
    parser.add_argument('--schedule', action='store_true', help='Set up daily scheduled updates')
    parser.add_argument('--backfill', action='store_true', help='Backfill the dataset to a specific date')
    parser.add_argument('--update', action='store_true', help='Update the dataset with one day of data')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date for backfilling (YYYY-MM-DD)')
    parser.add_argument('--update-time', type=str, default='00:00', help='Time to run daily updates (HH:MM)')
    parser.add_argument('--input', type=str, help='Input dataset path')
    parser.add_argument('--output', type=str, help='Output dataset path')
    
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    os.makedirs('data/stock_data', exist_ok=True)
    
    # Determine source dataset (in order of preference)
    if args.input:
        source_dataset = args.input
    else:
        source_datasets = [
            'data/stock_data/enhanced_dataset.csv',
            'data/stock_data/uploaded_dataset.csv',
            'data/stock_data/Dataset.csv'
        ]
        
        source_dataset = None
        for dataset in source_datasets:
            if os.path.exists(dataset):
                source_dataset = dataset
                print(f"Using source dataset: {source_dataset}")
                break
    
    if source_dataset is None:
        print("No suitable source dataset found. Please make sure one of the datasets exists.")
        sys.exit(1)
    
    # Determine output dataset
    output_dataset = args.output if args.output else 'data/stock_data/updated_dataset.csv'
    
    # Copy the original dataset if it doesn't exist in the data directory
    if not os.path.exists(output_dataset):
        original_df = pd.read_csv(source_dataset)
        original_df.to_csv(output_dataset, index=False)
        print(f"Source dataset copied to {output_dataset}")
    
    # Execute the requested action
    if args.backfill:
        backfill_dataset(
            output_dataset,
            end_date=args.end_date,
            output_path=output_dataset
        )
    elif args.update:
        update_dataset(
            output_dataset,
            output_path=output_dataset
        )
    elif args.schedule:
        setup_daily_update(
            output_dataset,
            output_path=output_dataset,
            update_time=args.update_time,
            end_date=args.end_date
        )
    else:
        # If no action specified, show usage
        parser.print_help()


if __name__ == "__main__":
    main() 