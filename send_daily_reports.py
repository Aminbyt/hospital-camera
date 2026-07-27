"""Daily Report Sender Thread - Guaranteed delivery via persistent tracking."""

import os
import requests
import logging
from datetime import datetime, timedelta
from PyQt5.QtCore import QThread
import config

class DailyReportThread(QThread):
    """Smart thread that guarantees yesterday's reports are sent, surviving reboots and internet drops."""

    def __init__(self):
        super().__init__()
        self.running = True
        # Create a permanent log file on the hard drive to track successful deliveries
        self.tracking_file = os.path.join(config.DB_PATH, "LOGS", "reports_sent_log.txt")

    def run(self):
        logging.info("[REPORTS] Smart Daily Report Thread initialized.")
        self.msleep(10000) # Give the system 10 seconds to boot up before using the network

        while self.running:
            yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

            # 1. Check the hard drive to see if we already successfully sent yesterday's reports
            if not self.already_sent(yesterday_str):
                logging.info(f"[REPORTS] Attempting to send reports for {yesterday_str}...")
                
                # 2. Try to send. If it returns True, internet is working and files are sent!
                if self.process_and_send(yesterday_str):
                    self.mark_as_sent(yesterday_str)
                    logging.info(f"[REPORTS SUCCESS] All reports for {yesterday_str} delivered!")
                else:
                    logging.warning("[REPORTS FAILED] Internet drop or error. Retrying in 15 minutes.")
                    self.smart_sleep(15 * 60) # Wait 15 minutes before retrying
                    continue # Restart the loop to retry immediately

            # 3. If already sent, sleep for 4 hours before checking the date again
            self.smart_sleep(4 * 3600) 

    def smart_sleep(self, seconds):
        """A responsive sleep that can be interrupted instantly when the app closes."""
        for _ in range(seconds):
            if not self.running:
                break
            self.msleep(1000)

    def already_sent(self, date_str):
        """Reads the physical tracking file to survive computer restarts."""
        if not os.path.exists(self.tracking_file):
            return False
        with open(self.tracking_file, 'r') as f:
            return date_str in f.read()

    def mark_as_sent(self, date_str):
        """Saves the date to the hard drive so we never send duplicates."""
        with open(self.tracking_file, 'a') as f:
            f.write(f"{date_str}\n")

    def process_and_send(self, target_date):
        base_url = config.BOT_API_URL.rsplit('/', 1)[0]
        send_doc_url = f"{base_url}/sendDocument"

        if not os.path.exists(config.REG_PATH):
            return True # Nothing to send, consider it a success

        all_success = True
        files_found = False

        for person_name in os.listdir(config.REG_PATH):
            person_dir = os.path.join(config.REG_PATH, person_name)
            
            if os.path.isdir(person_dir):
                excel_name = f"{person_name}_{target_date}.xlsx"
                excel_file = os.path.join(person_dir, excel_name)
                
                # If they washed their hands yesterday, the file exists
                if os.path.exists(excel_file):
                    files_found = True
                    try:
                        clean_display_name = person_name.replace('_', ' ')
                        caption_text = f"📊 Daily Hygiene Report for {clean_display_name} ({target_date})"

                        with open(excel_file, 'rb') as doc:
                            files = {'document': doc}
                            payload = {
                                'chat_id': config.BOT_CHAT_ID,
                                'caption': caption_text
                            }

                            # Attempt to send the file via Bale
                            resp = requests.post(send_doc_url, data=payload, files=files, timeout=20)
                            if resp.status_code != 200:
                                all_success = False # Network failed during upload
                                
                    except Exception as e:
                        logging.error(f"[REPORTS ERROR] Failed to send {excel_name}: {e}")
                        all_success = False

        # If we found files but couldn't send them ALL (e.g., internet drop), return False to trigger the 15-min retry
        if files_found and not all_success:
            return False
            
        return True

    def stop(self):
        self.running = False
        self.wait