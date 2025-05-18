from crewai import Agent
from agents.tools.data_analysis_tool import DataAnalysisTool
import pandas as pd
import logging
from agents.crewai_gemini_integration import GeminiCrewManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessorAgent:
    """Agent responsible for analyzing stock data and generating forecasts."""
    
    def __init__(self):
        """Initialize the data processor agent."""
        self.tool = DataAnalysisTool()
        self.crew_manager = GeminiCrewManager()
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the CrewAI agent."""
        # Define tools as a list of callable functions
        tools = [
            self._calculate_metrics,
            self._compare_sectors,
            self._train_model,
            self._generate_forecast,
            self._evaluate_january_2025,
            self._generate_report
        ]
        
        return self.crew_manager.create_agent(
            role="Stock Data Processor & Forecaster",
            goal="Analyze stock data and produce accurate price forecasts",
            backstory="""You are a quantitative analyst with deep expertise in financial
            modeling and machine learning. You have a track record of accurately predicting
            market trends and stock price movements. Your analyses provide valuable insights
            for investment decisions.""",
            verbose=True,
            tools=tools
        )
    
    def _calculate_metrics(self, dataframe):
        """
        Calculate key metrics for stock data.
        
        Args:
            dataframe: Stock data DataFrame
            
        Returns:
            dict: Dictionary with metrics per ticker
        """
        try:
            logger.info("Calculating stock metrics")
            metrics = self.tool.calculate_metrics(dataframe)
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _compare_sectors(self, dataframe):
        """
        Compare performance across sectors.
        
        Args:
            dataframe: Stock data DataFrame
            
        Returns:
            dict: Sector comparison results
        """
        try:
            logger.info("Comparing sectors")
            results = self.tool.compare_sectors(dataframe)
            return results
        except Exception as e:
            logger.error(f"Error comparing sectors: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _train_model(self, dataframe, ticker, epochs=20, batch_size=32):
        """
        Train a forecasting model for a ticker.
        
        Args:
            dataframe: Stock data DataFrame
            ticker (str): Ticker symbol to forecast
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training results and evaluation metrics
        """
        try:
            logger.info(f"Training model for {ticker}")
            results = self.tool.train_forecasting_model(dataframe, ticker, epochs, batch_size)
            return results
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _generate_forecast(self, dataframe, ticker, days=30):
        """
        Generate price forecasts for a ticker.
        
        Args:
            dataframe: Stock data DataFrame
            ticker (str): Ticker symbol to forecast
            days (int): Number of days to forecast
            
        Returns:
            dict: Forecast data
        """
        try:
            logger.info(f"Generating {days}-day forecast for {ticker}")
            forecast_df = self.tool.forecast_future(dataframe, ticker, days)
            
            if forecast_df is None:
                return {
                    'status': 'error',
                    'message': f"Unable to generate forecast for {ticker}"
                }
                
            return {
                'status': 'success',
                'ticker': ticker,
                'days': days,
                'forecast': forecast_df.to_dict(orient='records')
            }
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _evaluate_january_2025(self, predictions_df, actual_df=None):
        """
        Calculate metrics for January 2025 forecasts.
        
        Args:
            predictions_df: DataFrame with predictions
            actual_df: DataFrame with actual data (if available)
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info("Evaluating January 2025 forecast metrics")
            metrics = self.tool.calculate_january_2025_metrics(predictions_df, actual_df)
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating metrics: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _generate_report(self, dataframe, tickers=None):
        """
        Generate a comprehensive analysis report.
        
        Args:
            dataframe: Stock data DataFrame
            tickers (list): List of ticker symbols to analyze
            
        Returns:
            dict: Comprehensive analysis report
        """
        try:
            logger.info(f"Generating report for {tickers}")
            report = self.tool.generate_analysis_report(dataframe, tickers)
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def process_stock_data(self, dataframe, tickers=None):
        """
        Main method to process stock data.
        
        Args:
            dataframe: Stock data DataFrame
            tickers (list): List of ticker symbols to process
            
        Returns:
            dict: Processing results
        """
        try:
            # If no tickers specified, use all available
            if not tickers:
                tickers = dataframe['Ticker'].unique().tolist()
            elif isinstance(tickers, str):
                tickers = [t.strip() for t in tickers.split(',')]
                
            logger.info(f"Processing data for: {tickers}")
            
            # Calculate metrics
            metrics = self.tool.calculate_metrics(dataframe)
            
            # Compare sectors
            sector_comparison = self.tool.compare_sectors(dataframe)
            
            # Generate forecasts
            forecasts = {}
            for ticker in tickers:
                # Train model and generate forecast
                forecast_df = self.tool.forecast_future(dataframe, ticker, days=30)
                if forecast_df is not None:
                    forecasts[ticker] = forecast_df
            
            # Compile results
            results = {
                'metrics': metrics,
                'sector_comparison': sector_comparison,
                'forecasts': {ticker: forecast.to_dict(orient='records') for ticker, forecast in forecasts.items()},
                'tickers_processed': tickers
            }
            
            logger.info(f"Processed data for {len(tickers)} tickers with {len(forecasts)} forecasts")
            return results
        except Exception as e:
            logger.error(f"Error processing stock data: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from agents.data_collector_agent import DataCollectorAgent
    
    try:
        # First collect the data
        collector = DataCollectorAgent()
        data = collector.collect_stock_data(['AAPL', 'MSFT', 'GOOGL'])
        
        # Then process it
        processor = DataProcessorAgent()
        results = processor.process_stock_data(data)
        
        # Print results summary
        print("\nMetrics:")
        for ticker, metrics in results['metrics'].items():
            print(f"{ticker}: Highest={metrics['highest_price']:.2f}, Lowest={metrics['lowest_price']:.2f}")
        
        print("\nForecasts:")
        for ticker, forecast in results['forecasts'].items():
            print(f"{ticker}: {len(forecast)} days forecasted")
            
    except Exception as e:
        print(f"Error: {e}") 