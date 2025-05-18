from crewai import Agent
from agents.tools.kaggle_tool import KaggleTool
import logging
from agents.crewai_gemini_integration import GeminiCrewManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollectorAgent:
    """Agent responsible for collecting stock data from Kaggle."""
    
    def __init__(self):
        """Initialize the data collector agent."""
        self.tool = KaggleTool()
        self.crew_manager = GeminiCrewManager()
        self.agent = self._create_agent()
        
    def _create_agent(self):
        """Create the CrewAI agent."""
        # Define tools as a list of callable functions
        tools = [
            self._load_data, 
            self._update_data, 
            self._save_data
        ]
        
        return self.crew_manager.create_agent(
            role="Stock Data Collector",
            goal="Collect accurate and up-to-date stock data for analysis",
            backstory="""You are an expert in financial data collection, specializing in 
            stock market data. You have years of experience gathering, cleaning, and 
            preparing stock price data for analysis. Your skill at finding and processing 
            reliable data sources is unmatched.""",
            verbose=True,
            tools=tools
        )
    
    def _load_data(self, tickers=None):
        """
        Load stock data for specified tickers.
        
        Args:
            tickers (list, optional): List of ticker symbols to load
            
        Returns:
            dict: Dictionary with loaded data information
        """
        try:
            # Convert comma-separated string to list if needed
            if isinstance(tickers, str):
                tickers = [t.strip() for t in tickers.split(',')]
                
            logger.info(f"Loading data for tickers: {tickers}")
            df = self.tool.load_stock_data(tickers)
            
            # Return summary information about the loaded data
            return {
                'status': 'success',
                'tickers': list(df['Ticker'].unique()),
                'rows': len(df),
                'date_range': [df['Date'].min().strftime('%Y-%m-%d'), df['Date'].max().strftime('%Y-%m-%d')],
                'message': f"Successfully loaded {len(df)} rows of data for {len(df['Ticker'].unique())} tickers"
            }
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _update_data(self, tickers=None):
        """
        Update stock data with latest data.
        
        Args:
            tickers (list, optional): List of ticker symbols to update
            
        Returns:
            dict: Dictionary with update information
        """
        try:
            # Convert comma-separated string to list if needed
            if isinstance(tickers, str):
                tickers = [t.strip() for t in tickers.split(',')]
                
            logger.info(f"Updating data for tickers: {tickers}")
            df = self.tool.update_data(tickers)
            
            # Return summary information about the updated data
            return {
                'status': 'success',
                'tickers': list(df['Ticker'].unique()),
                'rows': len(df),
                'date_range': [df['Date'].min().strftime('%Y-%m-%d'), df['Date'].max().strftime('%Y-%m-%d')],
                'last_updated': df.attrs.get('last_updated', 'N/A'),
                'message': f"Successfully updated data for {len(df['Ticker'].unique())} tickers"
            }
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _save_data(self, dataframe, filename="processed_dataset.csv"):
        """
        Save stock data to a file.
        
        Args:
            dataframe: DataFrame to save
            filename (str): Name of the file to save
            
        Returns:
            dict: Dictionary with save information
        """
        try:
            logger.info(f"Saving data to {filename}")
            save_path = self.tool.save_data(dataframe, filename)
            
            return {
                'status': 'success',
                'path': save_path,
                'message': f"Successfully saved data to {save_path}"
            }
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def collect_stock_data(self, tickers=None):
        """
        Main method to collect stock data.
        
        Args:
            tickers (list, optional): List of ticker symbols to collect
            
        Returns:
            pandas.DataFrame: Collected stock data
        """
        try:
            # Load or download the data
            df = self.tool.load_stock_data(tickers)
            logger.info(f"Collected {len(df)} rows of data for {len(df['Ticker'].unique())} tickers")
            return df
        except Exception as e:
            logger.error(f"Error collecting stock data: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    agent = DataCollectorAgent()
    try:
        data = agent.collect_stock_data(['AAPL', 'MSFT', 'GOOGL'])
        print(f"Collected {len(data)} rows of data")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}") 