import os
import json
import logging
import google.generativeai as genai
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiTool:
    """Tool for generating market insights and recommendations using Google's Gemini API."""
    
    def __init__(self, api_key=None, output_dir="data/recommendations"):
        """
        Initialize the Gemini tool.
        
        Args:
            api_key (str): Google Generative AI API key, if None, will try to load from environment
            output_dir (str): Directory to save recommendations
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "AIzaSyCcB3tENdnFNKBIvciETMF196ldlUBmnyk")
        self.output_dir = output_dir
        self.model = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure Gemini
        self._configure()
    
    def _configure(self):
        """Configure the Gemini API."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
            raise
    
    def build_recommendation_prompt(self, ticker, metrics, forecast, sector_comparison=None, mse_rmse=None):
        """
        Build a dynamic prompt for stock recommendations.
        
        Args:
            ticker (str): Ticker symbol
            metrics (dict): Metrics for the ticker
            forecast (pd.DataFrame): Forecast data
            sector_comparison (dict): Sector comparison data
            mse_rmse (dict): Model error metrics
            
        Returns:
            str: Generated prompt
        """
        company_name = metrics.get('company_name', ticker)
        sector = metrics.get('sector', 'Unknown')
        highest_price = metrics.get('highest_price', 'N/A')
        lowest_price = metrics.get('lowest_price', 'N/A')
        growth_2020 = metrics.get('annual_growth_2020', 'N/A')
        
        # Build the beginning of the prompt
        prompt_parts = [
            f"You are an expert financial analyst specializing in stock market analysis. I need you to analyze {company_name} ({ticker}) in the {sector} sector.",
            "\nHere is the key data:"
        ]
        
        # Add metrics section
        prompt_parts.append(f"\n## Historical Performance")
        prompt_parts.append(f"- Highest Price: ${highest_price}")
        prompt_parts.append(f"- Lowest Price: ${lowest_price}")
        prompt_parts.append(f"- 2020 Annual Growth: {growth_2020}%")
        
        # Add sector comparison if available
        if sector_comparison and sector in sector_comparison:
            sector_data = sector_comparison[sector]
            prompt_parts.append(f"\n## Sector Analysis ({sector})")
            prompt_parts.append(f"- Sector's Overall Growth: {sector_data.get('overall_growth_pct', 'N/A')}%")
            prompt_parts.append(f"- Sector's Average Volatility: {sector_data.get('volatility', 'N/A')}")
            prompt_parts.append(f"- Other companies in this sector: {', '.join(sector_data.get('tickers', []))}")
        
        # Add forecast data if available
        if forecast is not None:
            # Get the first and last predicted prices
            if len(forecast) > 0:
                first_price = forecast.iloc[0]['Predicted_Close']
                last_price = forecast.iloc[-1]['Predicted_Close']
                price_change = last_price - first_price
                pct_change = (price_change / first_price) * 100
                
                prompt_parts.append(f"\n## Price Forecast")
                prompt_parts.append(f"- Forecast Period: {forecast.iloc[0]['Date'].strftime('%Y-%m-%d')} to {forecast.iloc[-1]['Date'].strftime('%Y-%m-%d')}")
                prompt_parts.append(f"- Starting Price: ${first_price:.2f}")
                prompt_parts.append(f"- Ending Price: ${last_price:.2f}")
                prompt_parts.append(f"- Predicted Change: ${price_change:.2f} ({pct_change:.2f}%)")
        
        # Add model accuracy metrics if available
        if mse_rmse:
            prompt_parts.append(f"\n## Model Accuracy")
            prompt_parts.append(f"- Mean Squared Error (MSE): {mse_rmse.get('mse', 'N/A')}")
            prompt_parts.append(f"- Root Mean Squared Error (RMSE): {mse_rmse.get('rmse', 'N/A')}")
        
        # Add the request for specific recommendation format
        prompt_parts.append(f"\n## Request")
        prompt_parts.append("Based on this analysis, provide a comprehensive investment recommendation. Your response MUST include:")
        prompt_parts.append("1. A clear BUY, HOLD, or SELL recommendation as the first line")
        prompt_parts.append("2. A brief 2-3 sentence explanation of your recommendation")
        prompt_parts.append("3. Key risk factors to consider")
        prompt_parts.append("4. A short-term (1-3 months) price outlook")
        prompt_parts.append("5. A long-term (1-2 years) strategic perspective")
        
        # Combine all parts
        prompt = "\n".join(prompt_parts)
        logger.info(f"Generated prompt for {ticker} ({len(prompt.split())} words)")
        return prompt
    
    def generate_recommendation(self, ticker, metrics, forecast, sector_comparison=None, mse_rmse=None):
        """
        Generate an investment recommendation.
        
        Args:
            ticker (str): Ticker symbol
            metrics (dict): Metrics for the ticker
            forecast (pd.DataFrame): Forecast data
            sector_comparison (dict): Sector comparison data
            mse_rmse (dict): Model error metrics
            
        Returns:
            dict: Generated recommendation and metadata
        """
        # Build the prompt
        prompt = self.build_recommendation_prompt(
            ticker, 
            metrics, 
            forecast, 
            sector_comparison, 
            mse_rmse
        )
        
        # Generate the recommendation
        try:
            logger.info(f"Calling Gemini API for {ticker} recommendation")
            response = self.model.generate_content(prompt)
            recommendation = response.text
            
            # Extract the recommendation type (BUY, HOLD, SELL)
            rec_type = "UNKNOWN"
            if recommendation.strip().upper().startswith("BUY"):
                rec_type = "BUY"
            elif recommendation.strip().upper().startswith("HOLD"):
                rec_type = "HOLD"
            elif recommendation.strip().upper().startswith("SELL"):
                rec_type = "SELL"
            
            # Create result with metadata
            result = {
                'ticker': ticker,
                'company_name': metrics.get('company_name', ticker),
                'sector': metrics.get('sector', 'Unknown'),
                'recommendation_type': rec_type,
                'recommendation_text': recommendation,
                'timestamp': datetime.now().isoformat(),
                'prompt_used': prompt
            }
            
            # Save the recommendation
            self._save_recommendation(result)
            
            logger.info(f"Generated {rec_type} recommendation for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendation for {ticker}: {e}")
            return {
                'ticker': ticker,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_recommendation(self, recommendation):
        """Save the recommendation to a file."""
        ticker = recommendation['ticker']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(recommendation, f, indent=2)
            
        logger.info(f"Saved recommendation to {filepath}")
    
    def generate_market_summary(self, metrics_dict, sector_comparison):
        """
        Generate an overall market summary.
        
        Args:
            metrics_dict (dict): Dictionary of metrics for all tickers
            sector_comparison (dict): Sector comparison data
            
        Returns:
            str: Generated market summary
        """
        # Build a prompt for overall market summary
        prompt_parts = [
            "You are an expert financial analyst. Generate a comprehensive market summary based on the following data:"
        ]
        
        # Add sector comparison data
        prompt_parts.append("\n## Sector Performance")
        for sector, data in sector_comparison.items():
            growth = data.get('overall_growth_pct', 'N/A')
            tickers = data.get('tickers', [])
            prompt_parts.append(f"- {sector}: {growth}% growth, Companies: {', '.join(tickers)}")
        
        # Add individual stock highlights
        prompt_parts.append("\n## Individual Stock Highlights")
        for ticker, metrics in metrics_dict.items():
            name = metrics.get('company_name', ticker)
            sector = metrics.get('sector', 'Unknown')
            growth = metrics.get('annual_growth_2020', 'N/A')
            prompt_parts.append(f"- {name} ({ticker}): {sector} sector, 2020 Growth: {growth}%")
        
        # Request specific format for the summary
        prompt_parts.append("\n## Request")
        prompt_parts.append("Please provide a comprehensive market summary that includes:")
        prompt_parts.append("1. An overall market health assessment")
        prompt_parts.append("2. Sector-by-sector performance analysis")
        prompt_parts.append("3. Top performing stocks and why")
        prompt_parts.append("4. Underperforming stocks and potential reasons")
        prompt_parts.append("5. Key market trends and indicators to watch")
        
        # Combine all parts
        prompt = "\n".join(prompt_parts)
        
        try:
            logger.info("Calling Gemini API for market summary")
            response = self.model.generate_content(prompt)
            summary = response.text
            
            # Save the summary
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.output_dir, f"market_summary_{timestamp}.txt")
            
            with open(filepath, 'w') as f:
                f.write(summary)
                
            logger.info(f"Generated market summary saved to {filepath}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return f"Error generating market summary: {e}"


if __name__ == "__main__":
    # Example usage
    try:
        # Create dummy data for testing
        ticker = "AAPL"
        metrics = {
            'company_name': 'Apple Inc.',
            'sector': 'Technology',
            'highest_price': 182.94,
            'lowest_price': 53.15,
            'annual_growth_2020': 80.75
        }
        
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create dummy forecast
        start_date = datetime.now()
        dates = [start_date + timedelta(days=i) for i in range(30)]
        forecast = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Predicted_Close': [150 + i*0.5 for i in range(30)]
        })
        
        # Create dummy sector comparison
        sector_comparison = {
            'Technology': {
                'tickers': ['AAPL', 'MSFT', 'GOOGL'],
                'overall_growth_pct': 25.5,
                'volatility': 0.12
            }
        }
        
        # Create dummy MSE/RMSE
        mse_rmse = {
            'mse': 0.025,
            'rmse': 0.158
        }
        
        # Test recommendation generation
        gemini_tool = GeminiTool()
        recommendation = gemini_tool.generate_recommendation(
            ticker, 
            metrics, 
            forecast, 
            sector_comparison, 
            mse_rmse
        )
        
        print(f"Recommendation Type: {recommendation['recommendation_type']}")
        print(f"Recommendation Text:\n{recommendation['recommendation_text']}")
        
    except Exception as e:
        print(f"Error: {e}") 