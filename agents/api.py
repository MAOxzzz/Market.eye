from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
import pandas as pd
import logging
import os
from typing import List, Optional
from pydantic import BaseModel
import json
from datetime import datetime

from agents.stock_analysis_crew import StockAnalysisCrew

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/crewai", tags=["CrewAI"])

# Define models
class TickerRequest(BaseModel):
    tickers: List[str]
    save_results: bool = True

class AnalysisResponse(BaseModel):
    request_id: str
    status: str
    message: str
    result_path: Optional[str] = None

# Store for background tasks
analysis_tasks = {}

def get_crew():
    """Dependency to get a StockAnalysisCrew instance."""
    return StockAnalysisCrew()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_stocks(
    request: TickerRequest,
    background_tasks: BackgroundTasks,
    crew: StockAnalysisCrew = Depends(get_crew)
):
    """
    Endpoint to analyze stocks using CrewAI agents.
    
    This starts a background task for the analysis and returns a request ID.
    """
    # Generate request ID
    request_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Define the task
    async def run_analysis_task():
        try:
            # Run analysis
            result = crew.analyze_stocks(request.tickers)
            
            # Update task status
            analysis_tasks[request_id].update({
                "status": "completed",
                "result": result
            })
            
            logger.info(f"Analysis completed for request {request_id}")
        except Exception as e:
            logger.error(f"Error in analysis task: {e}")
            analysis_tasks[request_id].update({
                "status": "failed",
                "error": str(e)
            })
    
    # Store task information
    analysis_tasks[request_id] = {
        "tickers": request.tickers,
        "status": "running",
        "created_at": datetime.now().isoformat(),
        "save_results": request.save_results
    }
    
    # Start background task
    background_tasks.add_task(run_analysis_task)
    
    return AnalysisResponse(
        request_id=request_id,
        status="running",
        message=f"Analysis started for tickers: {', '.join(request.tickers)}"
    )

@router.get("/status/{request_id}", response_model=AnalysisResponse)
async def get_analysis_status(request_id: str):
    """Get the status of an ongoing or completed analysis."""
    if request_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis request not found")
    
    task_info = analysis_tasks[request_id]
    
    if task_info["status"] == "completed":
        # Get path to results if available
        result_path = None
        if task_info.get("save_results", True):
            result_path = os.path.join("data/crew_outputs", f"analysis_results_{request_id}.json")
        
        return AnalysisResponse(
            request_id=request_id,
            status="completed",
            message="Analysis completed successfully",
            result_path=result_path
        )
    elif task_info["status"] == "failed":
        return AnalysisResponse(
            request_id=request_id,
            status="failed",
            message=f"Analysis failed: {task_info.get('error', 'Unknown error')}"
        )
    else:
        return AnalysisResponse(
            request_id=request_id,
            status="running",
            message=f"Analysis in progress for tickers: {', '.join(task_info['tickers'])}"
        )

@router.get("/results/{request_id}")
async def get_analysis_results(request_id: str):
    """Get the results of a completed analysis."""
    if request_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis request not found")
    
    task_info = analysis_tasks[request_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis not completed yet. Current status: {task_info['status']}")
    
    # Return the results
    return task_info.get("result", {"status": "error", "message": "Results not available"})

@router.get("/active")
async def get_active_analyses():
    """Get a list of all active analysis tasks."""
    active_tasks = {
        request_id: {
            "tickers": info["tickers"],
            "status": info["status"],
            "created_at": info["created_at"]
        }
        for request_id, info in analysis_tasks.items()
        if info["status"] == "running"
    }
    return active_tasks 