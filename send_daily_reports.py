"""Daily Report Sender Thread - Guaranteed delivery of Master Daily Summary only."""
import os
import requests
import logging
from datetime import datetime, timedelta
from PyQt5.QtCore import QThread
import config

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