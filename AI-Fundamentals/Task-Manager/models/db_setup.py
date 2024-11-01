import sqlite3
import os

project_root = os.path.dirname(os.path.dirname(__file__))

def create_connection():
    db_path = os.path.join(project_root, 'database', 'task_manager.db')
    conn = sqlite3.connect(db_path)
    return conn

def create_tables():
    conn = create_connection()
    cursor = conn.cursor()

    # Create user table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')

    # Create tasks table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    description TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'Pending',
    FOREIGN KEY (user_id) REFERENCES users (id)
                   );
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_tables()
    print("Database setup completed")