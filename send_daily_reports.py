"""Daily Report Sender Thread - Guaranteed delivery of Master Daily Summary only."""
import os
import requests
import logging
from datetime import datetime, timedelta
from PyQt5.QtCore import QThread
import config
import time
import pandas as pd

class MonthlyReportThread(QThread):
    """Background thread that sends a combined Excel report on the 1st of every month."""
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        print("[INFO] Monthly Report Scheduler Started...")
        while self.running:
            now = datetime.now()
            
            # WAKE UP CHECK: Is it the 1st day of the month at exactly 08:00 AM?
            if now.day == 1 and now.hour == 8 and now.minute == 0:
                self.generate_and_send_monthly_report(now)
                # Sleep for 65 seconds so it doesn't trigger twice in the same minute
                time.sleep(65)
            else:
                # Sleep for 30 seconds before checking the clock again
                time.sleep(30)

    def generate_and_send_monthly_report(self, current_date):
        # 1. Figure out what the previous month was
        last_month_date = current_date.replace(day=1) - timedelta(days=1)
        target_month_str = last_month_date.strftime("%Y-%m") # e.g., "2026-07"
        
        logs_dir = os.path.join(config.DB_PATH, "LOGS")
        if not os.path.exists(logs_dir):
            return
            
        all_files = os.listdir(logs_dir)
        monthly_data = []
        
        # 2. Gather all daily log files from that specific month
        for file in all_files:
            if target_month_str in file and (file.endswith('.csv') or file.endswith('.xlsx')):
                file_path = os.path.join(logs_dir, file)
                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    monthly_data.append(df)
                except Exception as e:
                    print(f"[ERROR] Could not read log file {file}: {e}")
        
        if not monthly_data:
            print(f"[INFO] No logs found for {target_month_str}. Skipping monthly report.")
            return
            
        # 3. Merge all daily logs into one master dataframe
        merged_df = pd.concat(monthly_data, ignore_index=True)
        
        # Sort by date/time if a 'Time' or 'Date' column exists
        if 'Date' in merged_df.columns:
            merged_df = merged_df.sort_values(by=['Date'])
            
        report_name = f"Hospital_AI_Monthly_Report_{target_month_str}.xlsx"
        report_path = os.path.join(logs_dir, report_name)
        
        # 4. Save the combined Excel file
        merged_df.to_excel(report_path, index=False)
        print(f"[INFO] Master Monthly report generated: {report_path}")
        
        # 5. Send it to the Bot
        self.send_to_bot(report_path, target_month_str)

    def send_to_bot(self, file_path, month_str):
        try:
            # Bale API requires the 'sendDocument' endpoint to upload files
            base_url = config.BOT_API_URL.replace("sendMessage", "sendDocument")
            
            with open(file_path, 'rb') as f:
                files = {'document': f}
                data = {
                    'chat_id': config.BOT_CHAT_ID, 
                    'caption': f"📊 Automated Monthly Handwashing Report for {month_str}"
                }
                
                print("[INFO] Uploading Monthly Report to Bale Group...")
                response = requests.post(base_url, data=data, files=files, timeout=60)
                
                if response.status_code == 200:
                    print(f"[SUCCESS] Monthly report sent to Bale group successfully!")
                else:
                    print(f"[ERROR] Failed to send monthly report. Bot replied: {response.text}")
        except Exception as e:
            print(f"[ERROR] Bot API exception during monthly upload: {e}")

    def stop(self):
        self.running = False
        self.wait()

class DailyReportThread(QThread):
    """Smart thread that sends ONLY the Master Daily Summary to the bot while leaving user reports local."""
    def __init__(self):
        super().__init__()
        self.running = True
        self.tracking_file = os.path.join(config.DB_PATH, "LOGS", "reports_sent_log.txt")

    def run(self):
        logging.info("[REPORTS] Master Daily Report Thread initialized.")
        self.msleep(10000)  # Wait 10 seconds after app launch before checking
        
        while self.running:
            today = datetime.now().date()
            
            # Scan the last 14 days for any unsent master summary reports
            for i in range(14, 0, -1):
                if not self.running:
                    break

                target_date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                
                if not self.already_sent(target_date):
                    if self.report_files_exist(target_date):
                        logging.info(f"[REPORTS CATCH-UP] Found unsent Master Summary for {target_date}. Delivering...")
                        if self.process_and_send(target_date):
                            self.mark_as_sent(target_date)
                            logging.info(f"[REPORTS SUCCESS] Master Summary for {target_date} delivered!")
                        else:
                            logging.warning(f"[REPORTS FAILED] Network error sending Master Summary for {target_date}. Will retry.")
                            break  # Exit loop to retry during next cycle
                    else:
                        # No master report exists for this date (e.g. system was off), mark so we don't re-check
                        self.mark_as_sent(target_date)

            # Sleep for 4 hours before checking again
            self.smart_sleep(4 * 3600)

    def smart_sleep(self, seconds):
        for _ in range(seconds):
            if not self.running:
                break
            self.msleep(1000)

    def already_sent(self, date_str):
        if not os.path.exists(self.tracking_file):
            return False
        with open(self.tracking_file, 'r') as f:
            return date_str in f.read()

    def mark_as_sent(self, date_str):
        with open(self.tracking_file, 'a') as f:
            f.write(f"{date_str}\n")

    def report_files_exist(self, target_date):
        """Checks if the Master Daily Summary Excel file exists on disk."""
        master_excel = os.path.join(config.DB_PATH, "LOGS", f"master_daily_report_{target_date}.xlsx")
        return os.path.exists(master_excel)

    def process_and_send(self, target_date):
        """Uploads ONLY the Master Daily Summary Excel report to the bot."""
        base_url = config.BOT_API_URL.rsplit('/', 1)[0]
        send_doc_url = f"{base_url}/sendDocument"

        master_excel = os.path.join(config.DB_PATH, "LOGS", f"master_daily_report_{target_date}.xlsx")
        if os.path.exists(master_excel):
            try:
                caption = f"📊 MASTER Daily Hygiene Summary ({target_date})"
                with open(master_excel, 'rb') as doc:
                    files = {'document': doc}
                    payload = {'chat_id': config.BOT_CHAT_ID, 'caption': caption}
                    resp = requests.post(send_doc_url, data=payload, files=files, timeout=20)
                    return resp.status_code == 200
            except Exception as e:
                logging.error(f"[REPORTS ERROR] Failed to send Master Report for {target_date}: {e}")
                return False

        return False

    def stop(self):
        self.running = False
        self.wait()     