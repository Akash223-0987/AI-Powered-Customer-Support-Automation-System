import sqlite3
try:
    conn = sqlite3.connect("memory.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, thread_id, department, query FROM customer_interactions")
    rows = cursor.fetchall()
    print("Database Rows:")
    for row in rows:
        print(row)
    conn.close()
except Exception as e:
    print("Error:", e)
