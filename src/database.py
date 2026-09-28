"""
database.py
--------------------------------------------------------------------
PURPOSE:
    Handles all PostgreSQL database operations for the
    Social Media Trend Monitoring Agent using psycopg2.

    TABLES:
        1. news_data
              id      SERIAL PRIMARY KEY
              title   TEXT
              source  TEXT

        2. twitter_sentiment
              id        SERIAL PRIMARY KEY
              text      TEXT
              sentiment VARCHAR(50)

    FUNCTIONS:
        - create_connection()   -> connect to PostgreSQL
        - create_tables()       -> create both tables if they don't exist
        - insert_news_data()    -> insert processed news rows
        - insert_twitter_data() -> insert tweet text + predicted sentiment
        - get_table_counts()    -> return row counts (used by dashboard)
--------------------------------------------------------------------
"""

import psycopg2
import pandas as pd

# ------------------------------------------------------------------
# DATABASE CONFIGURATION
# Update these values to match your local PostgreSQL setup.
# (See README.md for full PostgreSQL setup instructions.)
# ------------------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "database": "social_media_trend_db",
    "user": "postgres",
    "password": "Prathmesh@25"   # <-- CHANGE THIS before running
}


# ==================================================================
# 1. CREATE CONNECTION
# ==================================================================
def create_connection():
    """
    Create and return a psycopg2 connection object to PostgreSQL.
    Returns None if the connection fails (so the caller can handle it
    gracefully instead of crashing the Streamlit app).
    """
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        return conn
    except Exception as e:
        print(f"[ERROR] Could not connect to PostgreSQL: {e}")
        return None


# ==================================================================
# 2. CREATE TABLES
# ==================================================================
def create_tables() -> bool:
    """
    Create the 'news_data' and 'twitter_sentiment' tables if they
    do not already exist. Returns True on success, False on failure.
    """
    conn = create_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()

        # Table 1: news_data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_data (
                id SERIAL PRIMARY KEY,
                title TEXT,
                source TEXT
            );
        """)

        # Table 2: twitter_sentiment
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS twitter_sentiment (
                id SERIAL PRIMARY KEY,
                text TEXT,
                sentiment VARCHAR(50)
            );
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Tables created successfully (if they did not already exist).")
        return True

    except Exception as e:
        print(f"[ERROR] Could not create tables: {e}")
        conn.close()
        return False


# ==================================================================
# 3. INSERT NEWS DATA
# ==================================================================
def insert_news_data(news_df: pd.DataFrame) -> bool:
    """
    Insert processed news records (title, source) into the news_data table.
    """
    conn = create_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()

        for _, row in news_df.iterrows():
            cursor.execute(
                "INSERT INTO news_data (title, source) VALUES (%s, %s);",
                (row.get("title", ""), row.get("source", "Unknown"))
            )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Inserted {len(news_df)} news records into news_data table.")
        return True

    except Exception as e:
        print(f"[ERROR] Could not insert news data: {e}")
        conn.close()
        return False


# ==================================================================
# 4. INSERT TWITTER SENTIMENT DATA
# ==================================================================
def insert_twitter_data(predictions_df: pd.DataFrame) -> bool:
    """
    Insert tweet text and predicted sentiment into the twitter_sentiment table.
    Expects a DataFrame with columns: 'text' and 'predicted_sentiment'.
    """
    conn = create_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()

        for _, row in predictions_df.iterrows():
            cursor.execute(
                "INSERT INTO twitter_sentiment (text, sentiment) VALUES (%s, %s);",
                (row.get("text", ""), row.get("predicted_sentiment", "Unknown"))
            )

        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] Inserted {len(predictions_df)} twitter records into twitter_sentiment table.")
        return True

    except Exception as e:
        print(f"[ERROR] Could not insert twitter data: {e}")
        conn.close()
        return False


# ==================================================================
# 5. GET TABLE COUNTS (used by the "Database Status" dashboard page)
# ==================================================================
def get_table_counts() -> dict:
    """
    Return the number of records currently stored in each table.
    """
    conn = create_connection()
    if conn is None:
        return {"news_data": 0, "twitter_sentiment": 0}

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM news_data;")
        news_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM twitter_sentiment;")
        twitter_count = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return {"news_data": news_count, "twitter_sentiment": twitter_count}

    except Exception as e:
        print(f"[ERROR] Could not fetch table counts: {e}")
        conn.close()
        return {"news_data": 0, "twitter_sentiment": 0}


# ==================================================================
# Quick manual test
# ==================================================================
if __name__ == "__main__":
    create_tables()
    print(get_table_counts())
