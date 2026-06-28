import sqlite3
import os
from typing import List, Dict, Any

DB_PATH = "memory.db"

def init_db(db_path: str = DB_PATH):
    """Initializes the database and creates the custom customer_interactions table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT NOT NULL,
            customer_name TEXT,
            query TEXT NOT NULL,
            department TEXT,
            response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"-> Custom SQLite database initialized at {db_path} (table: customer_interactions)")

def log_interaction(thread_id: str, customer_name: str, query: str, department: str, response: str, db_path: str = DB_PATH):
    """Logs a single customer interaction to the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO customer_interactions (thread_id, customer_name, query, department, response)
        VALUES (?, ?, ?, ?, ?)
    """, (thread_id, customer_name, query, department, response))
    conn.commit()
    conn.close()
    # Also log to console for visibility
    print(f"-> Logged interaction to SQLite memory (Thread: {thread_id}, Dept: {department})")

def get_interaction_history(thread_id: str, db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves the customer interaction history for a given thread_id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT query, department, response, timestamp 
        FROM customer_interactions 
        WHERE thread_id = ? 
        ORDER BY timestamp ASC
    """, (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "query": row["query"],
            "department": row["department"],
            "response": row["response"],
            "timestamp": row["timestamp"]
        })
    return history

def get_history_summary_string(thread_id: str, db_path: str = DB_PATH) -> str:
    """Formats the conversation history as a readable summary string for the LLM."""
    history = get_interaction_history(thread_id, db_path)
    if not history:
        return "No previous interactions found."
        
    formatted = []
    for i, item in enumerate(history, 1):
        formatted.append(
            f"Interaction #{i}:\n"
            f"  - Customer Query: \"{item['query']}\"\n"
            f"  - Route/Department: {item['department']}\n"
            f"  - System Response: \"{item['response']}\""
        )
    return "\n\n".join(formatted)
