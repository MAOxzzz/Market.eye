# Database Components

This directory contains database schema definitions and management code.

## Components

- SQLite database schema
- Database migrations
- Connection management utilities
- Data access objects

## Schema

The database includes the following tables:

- `users` - User authentication and profile data
- `activity_logs` - User activity tracking
- `forecasts` - Saved forecast results
- `recommendations` - Investment recommendations

## Usage

Database connections are managed in the application:

```python
# Get database connection
def get_db_connection():
    conn = sqlite3.connect('market_eye.db')
    conn.row_factory = sqlite3.Row
    return conn

# Use connection
conn = get_db_connection()
cursor = conn.cursor()
```
