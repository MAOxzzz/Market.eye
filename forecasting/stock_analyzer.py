import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

class StockAnalyzer:
    def __init__(self, data_path: str, output_dir: str = 'forecasting/analysis'):
        """
        Initialize the stock analyzer.
        
        Args:
            data_path (str): Path to the stock data CSV file
            output_dir (str): Directory to save analysis outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'JPM': 'Finance',
            'BAC': 'Finance',
            'GS': 'Finance',
            'NKE': 'Sportswear',
            'UA': 'Sportswear',
            'LULU': 'Sportswear'
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the stock data."""
        self.df = pd.read_csv(self.data_path)
        
        # Convert dates with utc=True to avoid warnings and ensure proper conversion
        self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce', utc=True)
        
        # Drop rows with invalid dates
        self.df = self.df.dropna(subset=['Date'])
        
        # Ensure Date column is datetime type before extracting year
        if pd.api.types.is_datetime64_any_dtype(self.df['Date']):
            self.df['Year'] = self.df['Date'].dt.year
        else:
            # Fallback if date conversion somehow failed
            print("Warning: Date conversion did not result in datetime type. Year extraction skipped.")
            self.df['Year'] = pd.DatetimeIndex(self.df['Date']).year
        
        return self.df
    
    def compute_ticker_analytics(self, ticker: str) -> Dict:
        """
        Compute analytics for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Analytics including highest price, lowest price, and growth metrics
        """
        df_ticker = self.df[self.df['Ticker'] == ticker]
        
        # Basic price metrics
        highest_price = df_ticker['Close'].max()
        lowest_price = df_ticker['Close'].min()
        
        # Calculate annual growth
        df_ticker = df_ticker.sort_values('Date')
        first_price = df_ticker.iloc[0]['Close']
        last_price = df_ticker.iloc[-1]['Close']
        years = (df_ticker.iloc[-1]['Date'] - df_ticker.iloc[0]['Date']).days / 365.25
        annual_growth = ((last_price / first_price) ** (1/years) - 1) * 100
        
        # Calculate volatility (standard deviation of returns)
        returns = df_ticker['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        return {
            'ticker': ticker,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
            'annual_growth': annual_growth,
            'volatility': volatility,
            'current_price': last_price,
            'first_price': first_price,
            'years_analyzed': years
        }
    
    def generate_cross_company_comparison(self, tickers: List[str]) -> pd.DataFrame:
        """
        Generate comparison metrics across companies.
        
        Args:
            tickers (List[str]): List of ticker symbols to compare
            
        Returns:
            pd.DataFrame: Comparison metrics
        """
        comparison_data = []
        for ticker in tickers:
            analytics = self.compute_ticker_analytics(ticker)
            comparison_data.append(analytics)
        
        return pd.DataFrame(comparison_data)
    
    def generate_sector_comparison(self) -> Dict[str, pd.DataFrame]:
        """
        Generate comparison metrics across sectors.
        
        Returns:
            dict: Sector-wise comparison metrics
        """
        sector_metrics = {}
        
        for sector in set(self.sectors.values()):
            sector_tickers = [t for t, s in self.sectors.items() if s == sector]
            sector_df = self.df[self.df['Ticker'].isin(sector_tickers)]
            
            # Calculate sector-wide metrics
            sector_metrics[sector] = {
                'avg_annual_growth': sector_df.groupby('Ticker')['Close'].apply(
                    lambda x: ((x.iloc[-1] / x.iloc[0]) ** (1/len(x)) - 1) * 100
                ).mean(),
                'avg_volatility': sector_df.groupby('Ticker')['Close'].pct_change().std() * np.sqrt(252) * 100,
                'total_market_cap': sector_df.groupby('Ticker')['Close'].last().sum(),
                'num_companies': len(sector_tickers)
            }
        
        return pd.DataFrame(sector_metrics).T
    
    def plot_comparison_charts(self, tickers: List[str]):
        """
        Generate and save comparison charts.
        
        Args:
            tickers (List[str]): List of ticker symbols to compare
        """
        # Price comparison chart
        plt.figure(figsize=(12, 6))
        for ticker in tickers:
            df_ticker = self.df[self.df['Ticker'] == ticker]
            plt.plot(df_ticker['Date'], df_ticker['Close'], label=ticker)
        
        plt.title('Stock Price Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'price_comparison.png'))
        plt.close()
        
        # Growth comparison chart
        try:
            # Create a figure directly instead of using concat which can fail with duplicate dates
            plt.figure(figsize=(12, 6))
            
            for ticker in tickers:
                df_ticker = self.df[self.df['Ticker'] == ticker].sort_values('Date')
                
                # Make sure we have data to calculate growth
                if len(df_ticker) > 0:
                    first_price = df_ticker['Close'].iloc[0]
                    growth = ((df_ticker['Close'] / first_price) - 1) * 100
                    
                    # Plot directly without creating Series objects
                    plt.plot(df_ticker['Date'], growth, label=ticker)
            
            plt.title('Growth Comparison (Base 100)')
            plt.xlabel('Date')
            plt.ylabel('Growth (%)')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'growth_comparison.png'))
            plt.close()
        except Exception as e:
            print(f"Error generating growth comparison chart: {str(e)}")
            # Create a simple placeholder chart if there's an error
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, f"Could not generate growth chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(os.path.join(self.output_dir, 'growth_comparison.png'))
            plt.close()
    
    def generate_analysis_report(self, tickers: List[str] = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            tickers (List[str]): List of ticker symbols to analyze
            
        Returns:
            str: Path to the generated report
        """
        if tickers is None:
            tickers = list(self.sectors.keys())
        
        # Generate all analyses
        company_comparison = self.generate_cross_company_comparison(tickers)
        sector_comparison = self.generate_sector_comparison()
        self.plot_comparison_charts(tickers)
        
        # Save comparisons to CSV
        company_comparison.to_csv(os.path.join(self.output_dir, 'company_comparison.csv'))
        sector_comparison.to_csv(os.path.join(self.output_dir, 'sector_comparison.csv'))
        
        return self.output_dir 