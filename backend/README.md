# Backend Components

This directory contains all server-side and processing components for Market Eye AI.

## Structure

- `agents/` - AI agents and intelligent components that analyze stocks and generate recommendations
- `models/` - Machine learning models and prediction algorithms
- `database/` - Database schemas, migrations, and connection management

## Usage

The backend components can be accessed through the FastAPI endpoints defined in `main.py` at the root of the project.

```bash
# Start the backend server
uvicorn main:app --reload
```
