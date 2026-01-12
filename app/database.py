import os
import pymysql

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "ai_tutor_db")

if not DB_HOST or not DB_USER or not DB_NAME:
    raise ValueError("Database configuration incomplete. Please set DB_HOST, DB_USER, and DB_NAME in .env file")

DB_CONFIG = {
    "host": DB_HOST,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
}

def init_db():
    """Initialize ai_tutor_db and ensure users table exists."""

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(f"Connected to existing {DB_NAME} database.")
    except (pymysql.err.ProgrammingError, pymysql.err.OperationalError):
        temp_conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        temp_cursor = temp_conn.cursor()
        temp_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        temp_conn.commit()
        temp_conn.close()

        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE,
            password VARCHAR(1024)
        )
        '''
    )

    conn.commit()
    conn.close()
    print(f"MySQL database initialized successfully ({DB_NAME}.users)")
