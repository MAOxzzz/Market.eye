import os
import subprocess
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleTool:
    """Tool for downloading and managing stock data from Kaggle."""
    
    def __init__(self, dataset_name="jacksoncrow/stock-market-dataset", download_path="data/stock_data"):
        """
        Initialize the Kaggle tool.
        
        Args:
            dataset_name (str): Name of the Kaggle dataset
            download_path (str): Path to download the dataset to
        """
        self.dataset_name = dataset_name
        self.download_path = download_path
        self.ensure_directory()
        
    def ensure_directory(self):
        """Ensure the download directory exists."""
        os.makedirs(self.download_path, exist_ok=True)
        
    def check_kaggle_credentials(self):
        """Check if Kaggle API credentials exist."""
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_file = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_file):
            logger.warning("Kaggle API credentials not found. Please configure them.")
            return False
        return True
        
    def download_dataset(self):
        """Download the dataset from Kaggle."""
        if not self.check_kaggle_credentials():
            logger.error("Kaggle credentials not found. Cannot download dataset.")
            return False
            
        try:
            logger.info(f"Downloading dataset {self.dataset_name}...")
            command = f"kaggle datasets download {self.dataset_name} -p {self.download_path} --unzip"
            subprocess.run(command, shell=True, check=True)
            logger.info("Dataset downloaded and extracted successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dataset: {e}")
            return False
    
    def load_stock_data(self, tickers=None):
        """
        Load stock data for specified tickers.
        
        Args:
            tickers (list): List of ticker symbols to load. If None, load all available data.
            
        Returns:
            pd.DataFrame: Dataframe with stock data for the specified tickers.
        """
        # Check if dataset exists, download if not
        dataset_path = os.path.join(self.download_path, "Dataset.csv")
        if not os.path.exists(dataset_path):
            if not self.download_dataset():
                raise FileNotFoundError(f"Dataset not found at {dataset_path} and could not be downloaded.")
        
        # Load the dataset
        logger.info(f"Loading stock data from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        
        # Clean the data
        df = self.clean_data(df)
        
        # Filter by tickers if specified
        if tickers:
            df = df[df['Ticker'].isin(tickers)]
            logger.info(f"Filtered data for tickers: {tickers}")
        
        return df
    
    def clean_data(self, df):
        """
        Clean and prepare the stock data.
        
        Args:
            df (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Cleaned and prepared stock data
        """
        # Convert date to datetime
        try:
            # Check if Date column exists
            if 'Date' not in df.columns:
                logger.error("Date column not found in dataset")
                return df
                
            # Convert date to datetime with error handling
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Drop rows where Date conversion failed
            df = df.dropna(subset=['Date'])
            
            # Sort by ticker and date
            df = df.sort_values(['Ticker', 'Date'])
            
            # Handle missing values
            df = df.dropna(subset=['Close', 'Volume'])
            
            # Verify Date is datetime type before using dt accessor
            if pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Add year and month columns for easier filtering
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
            else:
                logger.warning("Date column could not be converted to datetime, skipping year/month extraction")
            
            return df
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            # Return original dataframe if cleaning fails
            return df
    
    def update_data(self, tickers=None):
        """
        Update the stock data with the latest available data.
        
        Args:
            tickers (list): List of ticker symbols to update
            
        Returns:
            pd.DataFrame: Updated stock data
        """
        # First load existing data
        df = self.load_stock_data(tickers)
        
        # Get the latest date in the data
        latest_date = df['Date'].max()
        
        # TODO: Implement API calls to update data with latest market data
        # For now, we'll just return the existing data with a log message
        logger.info(f"Data update would fetch data newer than {latest_date}. This is a placeholder.")
        
        # For demonstration, add metadata about the update
        df.attrs['last_updated'] = datetime.now().isoformat()
        df.attrs['update_status'] = 'placeholder'
        
        return df
    
    def save_data(self, df, filename="updated_dataset.csv"):
        """
        Save the stock data to a CSV file.
        
        Args:
            df (pd.DataFrame): Stock data to save
            filename (str): Name of the file to save to
            
        Returns:
            str: Path to the saved file
        """
        save_path = os.path.join(self.download_path, filename)
        df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
        return save_path


if __name__ == "__main__":
    # Example usage
    kaggle_tool = KaggleTool()
    try:
        # Try to load data for Apple, Microsoft and Google
        stock_data = kaggle_tool.load_stock_data(tickers=['AAPL', 'MSFT', 'GOOGL'])
        print(f"Loaded {len(stock_data)} rows of data")
        print(stock_data.head())
    except Exception as e:
        print(f"Error: {e}") 