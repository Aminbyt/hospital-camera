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
        self.tracking_file = os.path.join(config.DB_PATH, "LOGS", "reports_sent_log.txt")

    def run(self):
        logging.info("[REPORTS] Smart Daily Report Thread initialized.")
        self.msleep(10000)
        while self.running:
            yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            if not self.already_sent(yesterday_str):
                logging.info(f"[REPORTS] Attempting to send reports for {yesterday_str}...")
                if self.process_and_send(yesterday_str):
                    self.mark_as_sent(yesterday_str)
                    logging.info(f"[REPORTS SUCCESS] All reports for {yesterday_str} delivered!")
                else:
                    logging.warning("[REPORTS FAILED] Internet drop or error. Retrying in 15 minutes.")
                    self.smart_sleep(15 * 60)
                    continue
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

    def process_and_send(self, target_date):
        base_url = config.BOT_API_URL.rsplit('/', 1)[0]
        send_doc_url = f"{base_url}/sendDocument"

        all_success = True
        files_found = False

        # --- 1. Send Master Daily Excel Report ---
        master_excel = os.path.join(config.DB_PATH, "LOGS", f"master_daily_report_{target_date}.xlsx")
        if os.path.exists(master_excel):
            files_found = True
            try:
                caption = f"📊 MASTER Daily Hygiene Summary ({target_date})"
                with open(master_excel, 'rb') as doc:
                    files = {'document': doc}
                    payload = {'chat_id': config.BOT_CHAT_ID, 'caption': caption}
                    resp = requests.post(send_doc_url, data=payload, files=files, timeout=20)
                    if resp.status_code != 200:
                        all_success = False
            except Exception as e:
                logging.error(f"[REPORTS ERROR] Failed to send Master Report: {e}")
                all_success = False

        # --- 2. Send Individual Staff Excel Reports ---
        if os.path.exists(config.REG_PATH):
            for person_name in os.listdir(config.REG_PATH):
                person_dir = os.path.join(config.REG_PATH, person_name)
                if os.path.isdir(person_dir):
                    excel_name = f"{person_name}_{target_date}.xlsx"
                    excel_file = os.path.join(person_dir, excel_name)

                    if os.path.exists(excel_file):
                        files_found = True
                        try:
                            clean_display_name = person_name.replace('_', ' ')
                            caption_text = f"👤 Daily Hygiene Report for {clean_display_name} ({target_date})"
                            with open(excel_file, 'rb') as doc:
                                files = {'document': doc}
                                payload = {
                                    'chat_id': config.BOT_CHAT_ID,
                                    'caption': caption_text
                                }
                                resp = requests.post(send_doc_url, data=payload, files=files, timeout=20)
                                if resp.status_code != 200:
                                    all_success = False
                        except Exception as e:
                            logging.error(f"[REPORTS ERROR] Failed to send {excel_name}: {e}")
                            all_success = False

        if files_found and not all_success:
            return False

        return True

    def stop(self):
        self.running = False
        self.wait()