import sqlite3
import os

# Get project root and database path
project_root = os.path.dirname(os.path.dirname(__file__))
db_dir = os.path.join(project_root, 'MCP-Server', 'database')
os.makedirs(db_dir, exist_ok=True)  # Ensure 'database' folder exists

def create_connection():
    db_path = os.path.join(db_dir, 'sales.db')
    conn = sqlite3.connect(db_path)
    return conn

def create_sales_table(conn):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY,
        product TEXT NOT NULL,
        sale_amount REAL NOT NULL,
        sale_date TEXT NOT NULL
    )
    """)
    conn.commit()

def insert_sample_data(conn):
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT INTO sales (product, sale_amount, sale_date)
    VALUES (?, ?, ?)
    """, [
        ("Widget A", 120.50, "2025-06-10"),
        ("Widget B", 250.00, "2025-06-11"),
        ("Widget A", 180.75, "2025-06-12"),
        ("Widget C", 300.10, "2025-06-13"),
        ("Widget B", 150.00, "2025-06-14"),
    ])
    conn.commit()

if __name__ == "__main__":
    conn = create_connection()
    create_sales_table(conn)
    insert_sample_data(conn)
    conn.close()
    print("sales.db created with sales table and sample data.")