"""Automated Report Generator - Queries SQLite, builds Excel, and uploads to Bale Bot."""

import os
import time
import threading
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta

import config
from session_database import SessionDatabase
from logger_setup import LogCategory


class DailyReportThread(threading.Thread):
    """Background daemon that generates and uploads yesterday's report."""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.reports_dir = os.path.join(config.DB_PATH, "REPORTS")
        os.makedirs(self.reports_dir, exist_ok=True)

    def run(self):
        logging.info(f"{LogCategory.NETWORK} Daily Report Thread started.")
        while self.running:
            now = datetime.now()
            
            # --- 1. SCHEDULED GENERATION ---
            # Run every night at 23:55
            if now.hour == 23 and now.minute == 55:
                target_date = now.strftime("%Y-%m-%d")
                self.generate_and_upload(target_date)
                time.sleep(60)  # Sleep 1 min to prevent duplicate triggers
            
            # --- 2. RETRY PENDING UPLOADS ---
            # Scan for failed uploads at the top of every hour
            elif now.minute == 0:
                self.retry_pending_uploads()
                time.sleep(60)

            # Sleep in short interruptible chunks
            for _ in range(30):
                if not self.running:
                    return
                time.sleep(1)

    def generate_and_upload(self, target_date):
        file_path = os.path.join(self.reports_dir, f"daily_report_{target_date}.xlsx")
        
        # 1. Query SQLite and generate Excel (if not already generated)
        if not os.path.exists(file_path) and not os.path.exists(file_path + ".uploaded"):
            db = SessionDatabase()
            df = db.get_sessions_by_date(target_date)
            
            if df.empty:
                logging.info(f"{LogCategory.DATABASE} No sessions found for {target_date}. Skipping report.")
                return False
                
            # Clean up the output
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
                
            df.to_excel(file_path, index=False)
            logging.info(f"{LogCategory.DATABASE} Generated daily report: {file_path}")

        # 2. Attempt Upload
        if os.path.exists(file_path):
            return self.upload_report(file_path, target_date)
        return True

    def upload_report(self, file_path, target_date):
        if not getattr(config, 'BOT_DOC_URL', None):
            return False
            
        try:
            caption = f"🏥 Automated Daily Hygiene Summary ({target_date})"
            with open(file_path, 'rb') as doc:
                files = {'document': doc}
                payload = {'chat_id': config.BOT_CHAT_ID, 'caption': caption}
                
                resp = requests.post(config.BOT_DOC_URL, data=payload, files=files, timeout=30)
                
                if resp.status_code == 200:
                    logging.info(f"{LogCategory.NETWORK} Successfully uploaded {file_path}")
                    # Mark as successfully processed
                    os.rename(file_path, file_path + ".uploaded")
                    return True
                else:
                    logging.warning(f"{LogCategory.NETWORK} Upload failed for {file_path}: HTTP {resp.status_code}")
                    
        except requests.exceptions.RequestException as e:
            logging.error(f"{LogCategory.NETWORK} Network error uploading {file_path}: {e}")
        
        return False

    def retry_pending_uploads(self):
        """Scans the reports directory for un-uploaded files and retries them."""
        for filename in os.listdir(self.reports_dir):
            if filename.endswith(".xlsx"): 
                file_path = os.path.join(self.reports_dir, filename)
                try:
                    # Extract date from filename format: daily_report_YYYY-MM-DD.xlsx
                    date_str = filename.split('_')[2].split('.')[0]
                    logging.info(f"{LogCategory.NETWORK} Retrying upload for pending report: {filename}")
                    self.upload_report(file_path, date_str)
                except Exception as e:
                    logging.error(f"{LogCategory.NETWORK} Failed to process pending report {filename}: {e}")

    def stop(self):
        self.running = False
        self.join(timeout=2.0)


class MonthlyReportThread(threading.Thread):
    """Background daemon that generates and uploads the previous month's report."""
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.reports_dir = os.path.join(config.DB_PATH, "REPORTS")
        os.makedirs(self.reports_dir, exist_ok=True)

    def run(self):
        logging.info(f"{LogCategory.NETWORK} Monthly Report Thread started.")
        while self.running:
            now = datetime.now()
            
            # Run at 10:00 AM on the 1st day of every month
            if now.day == 1 and now.hour == 10 and now.minute == 0:
                # Calculate the previous month string (YYYY-MM)
                first_of_this_month = now.replace(day=1)
                last_month_date = first_of_this_month - timedelta(days=1)
                target_month = last_month_date.strftime("%Y-%m")
                
                self.generate_and_upload(target_month)
                time.sleep(60) 

            # Retry pending monthly uploads at 10:30 AM
            elif now.minute == 30:
                self.retry_pending_uploads()
                time.sleep(60)

            for _ in range(30):
                if not self.running:
                    return
                time.sleep(1)

    def generate_and_upload(self, target_month):
        file_path = os.path.join(self.reports_dir, f"monthly_report_{target_month}.xlsx")
        
        if not os.path.exists(file_path) and not os.path.exists(file_path + ".uploaded"):
            db = SessionDatabase()
            
            # Use pandas to query SQLite via connection for custom LIKE matching
            query = f"SELECT * FROM sessions WHERE date LIKE '{target_month}-%'"
            df = pd.read_sql_query(query, db.conn)
            
            if df.empty:
                logging.info(f"{LogCategory.DATABASE} No sessions found for {target_month}. Skipping report.")
                return False
                
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
                
            df.to_excel(file_path, index=False)
            logging.info(f"{LogCategory.DATABASE} Generated monthly report: {file_path}")

        if os.path.exists(file_path):
            return self.upload_report(file_path, target_month)
        return True

    def upload_report(self, file_path, target_month):
        if not getattr(config, 'BOT_DOC_URL', None):
            return False
            
        try:
            caption = f"📊 Automated Monthly Handwashing Report for {target_month}"
            with open(file_path, 'rb') as doc:
                files = {'document': doc}
                payload = {'chat_id': config.BOT_CHAT_ID, 'caption': caption}
                
                resp = requests.post(config.BOT_DOC_URL, data=payload, files=files, timeout=60)
                
                if resp.status_code == 200:
                    logging.info(f"{LogCategory.NETWORK} Successfully uploaded {file_path}")
                    os.rename(file_path, file_path + ".uploaded")
                    return True
                else:
                    logging.warning(f"{LogCategory.NETWORK} Upload failed for monthly report: HTTP {resp.status_code}")
        except Exception as e:
            logging.error(f"{LogCategory.NETWORK} Network error uploading monthly report: {e}")
        
        return False

    def retry_pending_uploads(self):
        for filename in os.listdir(self.reports_dir):
            if filename.startswith("monthly_report_") and filename.endswith(".xlsx"):
                file_path = os.path.join(self.reports_dir, filename)
                try:
                    month_str = filename.split('_')[2].split('.')[0]
                    logging.info(f"{LogCategory.NETWORK} Retrying upload for pending monthly report: {filename}")
                    self.upload_report(file_path, month_str)
                except Exception as e:
                    logging.error(f"{LogCategory.NETWORK} Failed to process pending report {filename}: {e}")

    def stop(self):
        self.running = False
        self.join(timeout=2.0)