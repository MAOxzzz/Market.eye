from crewai import Crew, Process, Task
import pandas as pd
import logging
import os

from agents.data_collector_agent import DataCollectorAgent
from agents.data_processor_agent import DataProcessorAgent
from agents.llm_recommendation_agent import LLMRecommendationAgent
from agents.crewai_gemini_integration import GeminiCrewManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockAnalysisCrew:
    """Crew for orchestrating stock analysis with CrewAI agents."""
    
    def __init__(self, gemini_api_key=None):
        """
        Initialize the stock analysis crew.
        
        Args:
            gemini_api_key (str, optional): API key for Gemini
        """
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "AIzaSyCcB3tENdnFNKBIvciETMF196ldlUBmnyk")
        self.output_dir = "data/crew_outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize the Gemini Crew Manager
        self.crew_manager = GeminiCrewManager(api_key=self.api_key)
        
        # Initialize agents
        self.data_collector = DataCollectorAgent()
        self.data_processor = DataProcessorAgent()
        self.recommendation_generator = LLMRecommendationAgent(api_key=self.api_key)
        
        # Save agent references for crew creation
        self.collector_agent = self.data_collector.agent
        self.processor_agent = self.data_processor.agent
        self.recommender_agent = self.recommendation_generator.agent
    
    def run_analysis(self, tickers=None, save_results=True):
        """
        Run the full stock analysis pipeline.
        
        Args:
            tickers (list): List of ticker symbols to analyze
            save_results (bool): Whether to save results to disk
            
        Returns:
            dict: Analysis results
        """
        logger.info(f"Starting stock analysis for tickers: {tickers}")
        
        # Define tasks for the crew
        collect_task = Task(
            description=f"Collect stock data for {tickers if tickers else 'all available stocks'}",
            agent=self.collector_agent,
            expected_output="A DataFrame containing stock data for the requested tickers"
        )
        
        process_task = Task(
            description="Analyze the stock data and generate forecasts",
            agent=self.processor_agent,
            expected_output="A dictionary containing metrics, sector comparisons, and price forecasts"
        )
        
        recommend_task = Task(
            description="Generate investment recommendations based on the analysis",
            agent=self.recommender_agent,
            expected_output="A dictionary containing recommendations for each ticker and an overall market summary"
        )
        
        # Create the crew
        crew = Crew(
            agents=[self.collector_agent, self.processor_agent, self.recommender_agent],
            tasks=[collect_task, process_task, recommend_task],
            verbose=True,
            process=Process.sequential  # Run tasks in sequence
        )
        
        # Start the crew
        try:
            result = crew.kickoff()
            
            if save_results:
                self._save_results(result)
                
            return result
        except Exception as e:
            logger.error(f"Error running stock analysis crew: {e}")
            raise
    
    def analyze_stocks(self, tickers=None):
        """
        Analyze stocks without using CrewAI task framework.
        More direct approach for debugging and development.
        
        Args:
            tickers (list): List of ticker symbols to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            # Step 1: Collect data
            logger.info(f"Collecting data for {tickers}")
            try:
                stock_data = self.data_collector.collect_stock_data(tickers)
            except Exception as e:
                logger.error(f"Error in data collection step: {e}")
                raise Exception(f"Failed to collect stock data: {str(e)}")
            
            # Step 2: Process data
            logger.info("Processing data and generating forecasts")
            try:
                analysis_results = self.data_processor.process_stock_data(stock_data, tickers)
            except Exception as e:
                logger.error(f"Error in data processing step: {e}")
                raise Exception(f"Failed to process stock data: {str(e)}")
            
            # Step 3: Generate recommendations
            logger.info("Generating investment recommendations")
            try:
                recommendations = self.recommendation_generator.generate_recommendations(analysis_results)
            except Exception as e:
                logger.error(f"Error in recommendation generation step: {e}")
                raise Exception(f"Failed to generate recommendations: {str(e)}")
            
            # Ensure Date column is proper datetime type before accessing min/max
            date_range = ["N/A", "N/A"]
            try:
                if 'Date' in stock_data.columns:
                    # Verify Date is datetime type
                    if pd.api.types.is_datetime64_any_dtype(stock_data['Date']):
                        date_range = [
                            stock_data['Date'].min().strftime('%Y-%m-%d'),
                            stock_data['Date'].max().strftime('%Y-%m-%d')
                        ]
                    else:
                        logger.warning("Date column is not in datetime format. Cannot extract min/max dates.")
            except Exception as e:
                logger.warning(f"Error extracting date range: {e}")
            
            # Combine all results
            results = {
                'data_collection': {
                    'tickers': list(stock_data['Ticker'].unique()),
                    'rows': len(stock_data),
                    'date_range': date_range
                },
                'analysis': analysis_results,
                'recommendations': recommendations
            }
            
            # Save results
            self._save_results(results)
            
            logger.info("Stock analysis completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in stock analysis: {e}")
            raise
    
    def _save_results(self, results):
        """Save analysis results to disk."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert any non-serializable objects
        def json_serializer(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="records")
            try:
                return obj.__dict__
            except:
                return str(obj)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(results, f, default=json_serializer, indent=2)
            
        logger.info(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    try:
        # Create the crew
        crew = StockAnalysisCrew()
        
        # Run analysis for Apple and Microsoft
        results = crew.analyze_stocks(['AAPL', 'MSFT'])
        
        # Print recommendation summary
        print("\nRecommendations:")
        recommendations = results['recommendations']['ticker_recommendations']
        for ticker, rec in recommendations.items():
            rec_type = rec.get('recommendation_type', 'UNKNOWN')
            print(f"{ticker}: {rec_type}")
            
        # Print a snippet of the market summary
        summary = results['recommendations']['market_summary']
        print("\nMarket Summary:")
        print(summary[:200] + "...")
        
    except Exception as e:
        print(f"Error: {e}") 