#!/usr/bin/env python
"""
Test script for the CrewAI implementation.
This script tests the Data Collector, Data Processor, and LLM Recommendation agents.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import from the agents module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.stock_analysis_crew import StockAnalysisCrew

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crewai_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_crew_analyze_stocks():
    """
    Test the full stock analysis pipeline using direct method calls.
    """
    logger.info("Starting CrewAI test: analyze_stocks")
    
    try:
        # Create the crew
        crew = StockAnalysisCrew()
        
        # Run analysis for a single ticker to keep test time reasonable
        # Use just Apple for a quick test
        results = crew.analyze_stocks(['AAPL'])
        
        # Basic validation of the results
        assert 'data_collection' in results, "Missing data_collection in results"
        assert 'analysis' in results, "Missing analysis in results"
        assert 'recommendations' in results, "Missing recommendations in results"
        
        # Access the recommendations
        recommendations = results['recommendations']['ticker_recommendations']
        for ticker, rec in recommendations.items():
            rec_type = rec.get('recommendation_type', 'UNKNOWN')
            logger.info(f"{ticker}: {rec_type}")
            
        # Check the market summary
        summary = results['recommendations'].get('market_summary', '')
        logger.info(f"Market Summary: {summary[:100]}...")
        
        logger.info("CrewAI test: analyze_stocks PASSED")
        return True
    except Exception as e:
        logger.error(f"CrewAI test: analyze_stocks FAILED with error: {e}")
        return False

def test_individual_agents():
    """
    Test each agent individually.
    """
    logger.info("Starting CrewAI test: individual_agents")
    
    try:
        # Create the crew to access the individual agents
        crew = StockAnalysisCrew()
        
        # Test Data Collector
        logger.info("Testing Data Collector agent")
        stock_data = crew.data_collector.collect_stock_data(['AAPL'])
        assert len(stock_data) > 0, "Data Collector returned empty dataset"
        
        # Test Data Processor
        logger.info("Testing Data Processor agent")
        analysis_results = crew.data_processor.process_stock_data(stock_data)
        assert 'metrics' in analysis_results, "Missing metrics in analysis results"
        
        # Test LLM Recommendation Generator
        logger.info("Testing LLM Recommendation Generator agent")
        recommendations = crew.recommendation_generator.generate_recommendations(analysis_results)
        assert 'ticker_recommendations' in recommendations, "Missing ticker_recommendations in recommendations"
        
        logger.info("CrewAI test: individual_agents PASSED")
        return True
    except Exception as e:
        logger.error(f"CrewAI test: individual_agents FAILED with error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting CrewAI tests")
    
    # Run the tests
    test_individual_agents()
    test_crew_analyze_stocks()
    
    logger.info("CrewAI tests completed") 