"""SQLite Database Module - Handles fast, concurrent session logging."""

import os
import sqlite3
import threading
import pandas as pd
import config
from logger_setup import LogCategory
import logging


class SessionDatabase:
    """Singleton: Manages the SQLite connection and table structures."""
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(SessionDatabase, cls).__new__(cls)
                cls._instance._init_db()
            return cls._instance

    def _init_db(self):
        """Creates the database file and sessions table if they do not exist."""
        self.db_path = os.path.join(config.DB_PATH, "hospital_sessions.db")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT,
                        user TEXT,
                        role TEXT,
                        login_time TEXT,
                        mask TEXT,
                        hat TEXT,
                        wash_complete TEXT,
                        wash_duration INTEGER,
                        who_steps TEXT,
                        sink TEXT
                    )
                """)
                conn.commit()
                logging.info("[DB] SQLite database initialized successfully.")
        except Exception as e:
            logging.error(f"[DB ERROR] Failed to initialize SQLite: {e}")

    def insert_session(self, date_str, user, role, login_time, mask, hat, wash_complete, wash_duration, who_steps, sink):
        """Performs a lightweight, fast insertion of a completed session."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions 
                    (date, user, role, login_time, mask, hat, wash_complete, wash_duration, who_steps, sink)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (date_str, user, role, login_time, mask, hat, wash_complete, wash_duration, who_steps, sink))
                conn.commit()
            return True
        except Exception as e:
            logging.error(f"[DB ERROR] Failed to insert session: {e}")
            return False

    def get_sessions_by_date(self, target_date):
        """Returns a Pandas DataFrame of all sessions for a specific day."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM sessions WHERE date = ?"
                return pd.read_sql_query(query, conn, params=(target_date,))
        except Exception as e:
            logging.error(f"[DB ERROR] Failed to query by date: {e}")
            return pd.DataFrame()

    def get_sessions_by_month(self, target_month_str):
        """Returns a Pandas DataFrame of all sessions for a specific month (e.g., '2026-07')."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM sessions WHERE date LIKE ?"
                return pd.read_sql_query(query, conn, params=(target_month_str + '%',))
        except Exception as e:
            logging.error(f"[DB ERROR] Failed to query by month: {e}")
            return pd.DataFrame()