from crewai import Agent
from agents.tools.gemini_tool import GeminiTool
import pandas as pd
import logging
from agents.crewai_gemini_integration import GeminiCrewManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMRecommendationAgent:
    """Agent responsible for generating investment recommendations using Gemini API."""
    
    def __init__(self, api_key=None):
        """Initialize the LLM recommendation agent."""
        self.tool = GeminiTool(api_key)
        self.crew_manager = GeminiCrewManager(api_key=api_key)
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create the CrewAI agent."""
        # Define tools as a list of callable functions
        tools = [
            self._generate_recommendation,
            self._generate_market_summary
        ]
        
        return self.crew_manager.create_agent(
            role="Investment Advisor",
            goal="Generate accurate and valuable investment recommendations",
            backstory="""You are a seasoned investment advisor with decades of experience
            in financial markets. You have a unique ability to analyze complex financial
            data and translate it into clear, actionable advice. Your clients trust your
            judgment and consistently benefit from your recommendations.""",
            verbose=True,
            tools=tools
        )
    
    def _generate_recommendation(self, ticker, metrics, forecast, sector_comparison=None, mse_rmse=None):
        """
        Generate an investment recommendation for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            metrics (dict): Metrics for the ticker
            forecast (dict): Forecast data
            sector_comparison (dict): Sector comparison data
            mse_rmse (dict): Model error metrics
            
        Returns:
            dict: Generated recommendation
        """
        try:
            logger.info(f"Generating recommendation for {ticker}")
            
            # Convert forecast back to DataFrame if it's a dict
            if isinstance(forecast, dict) or isinstance(forecast, list):
                forecast = pd.DataFrame(forecast)
            
            recommendation = self.tool.generate_recommendation(
                ticker,
                metrics,
                forecast,
                sector_comparison,
                mse_rmse
            )
            
            return recommendation
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {
                'status': 'error',
                'ticker': ticker,
                'message': str(e)
            }
    
    def _generate_market_summary(self, metrics_dict, sector_comparison):
        """
        Generate an overall market summary.
        
        Args:
            metrics_dict (dict): Dictionary of metrics for all tickers
            sector_comparison (dict): Sector comparison data
            
        Returns:
            str: Generated market summary
        """
        try:
            logger.info("Generating market summary")
            summary = self.tool.generate_market_summary(metrics_dict, sector_comparison)
            return summary
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return f"Error generating market summary: {e}"
    
    def generate_recommendations(self, analysis_results):
        """
        Main method to generate recommendations based on analysis results.
        
        Args:
            analysis_results (dict): Results from the data processor agent
            
        Returns:
            dict: Generated recommendations
        """
        try:
            metrics = analysis_results.get('metrics', {})
            sector_comparison = analysis_results.get('sector_comparison', {})
            forecasts = analysis_results.get('forecasts', {})
            
            logger.info(f"Generating recommendations for {len(metrics)} tickers")
            
            # Generate individual ticker recommendations
            recommendations = {}
            for ticker, ticker_metrics in metrics.items():
                forecast = forecasts.get(ticker, None)
                
                # Generate recommendation
                recommendation = self.tool.generate_recommendation(
                    ticker,
                    ticker_metrics,
                    forecast,
                    sector_comparison
                )
                
                recommendations[ticker] = recommendation
            
            # Generate market summary
            market_summary = self.tool.generate_market_summary(metrics, sector_comparison)
            
            # Compile results
            results = {
                'ticker_recommendations': recommendations,
                'market_summary': market_summary,
                'tickers_analyzed': list(metrics.keys())
            }
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return results
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from agents.data_collector_agent import DataCollectorAgent
    from agents.data_processor_agent import DataProcessorAgent
    
    try:
        # First collect the data
        collector = DataCollectorAgent()
        data = collector.collect_stock_data(['AAPL', 'MSFT'])
        
        # Then process it
        processor = DataProcessorAgent()
        analysis_results = processor.process_stock_data(data)
        
        # Finally, generate recommendations
        recommender = LLMRecommendationAgent()
        recommendations = recommender.generate_recommendations(analysis_results)
        
        # Print recommendation summary
        print("\nRecommendations:")
        for ticker, rec in recommendations['ticker_recommendations'].items():
            rec_type = rec.get('recommendation_type', 'UNKNOWN')
            print(f"{ticker}: {rec_type}")
            
        # Print market summary
        print("\nMarket Summary:")
        print(recommendations['market_summary'][:100] + "...")
        
    except Exception as e:
        print(f"Error: {e}") 