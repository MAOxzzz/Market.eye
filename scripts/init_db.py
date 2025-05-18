# scripts/init_db.py

import sqlite3
import os

def init_db(
    db_path: str = os.path.join(os.path.dirname(__file__), '..', 'market_eye.db'),
    schema_path: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'schema.sql')
):
    # Load the SQL schema
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # Connect (or create) the SQLite database
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
        print(f"✔ Database initialized at: {os.path.abspath(db_path)}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
 