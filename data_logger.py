import os
import time
import pandas as pd
import requests
import threading
import config
import logging

# --- THE MISSING GLOBAL LOCK ---
EXCEL_LOCK = threading.Lock()

class DataLogger:
    """Manages user session logging to Excel and sends notifications."""
    def __init__(self):
        """Initialize data logger."""
        self.ensure_directories()

    @staticmethod
    def ensure_directories():
        """Ensure database directories exist."""
        os.makedirs(config.REG_PATH, exist_ok=True)
        os.makedirs(config.INFO_PATH, exist_ok=True)

    def log_session(self, current_user, login_time, wash_status, mask_status, hat_status, all_steps):
        """Log a user session to a daily Excel file inside that specific person's registration folder."""
        if not current_user:
            return False
        try:
            date_str = time.strftime("%Y-%m-%d")
           
            folder_underscores = os.path.join(config.REG_PATH, current_user.replace(" ", "_"))
            folder_spaces = os.path.join(config.REG_PATH, current_user)

            if os.path.exists(folder_underscores):
                person_dir = folder_underscores
            elif os.path.exists(folder_spaces):
                person_dir = folder_spaces
            else:
                person_dir = folder_underscores
                os.makedirs(person_dir, exist_ok=True)

            # <-- NEW: Create Excel file as <person name>_date.xlsx
            clean_name = current_user.replace(" ", "_")
            excel_file = os.path.join(person_dir, f"{clean_name}_{date_str}.xlsx")
           
            parts = current_user.split(" ", 1)
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            with EXCEL_LOCK:
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file)
                    visit_count = len(df) + 1
                else:
                    df = pd.DataFrame()
                    visit_count = 1

                new_data = pd.DataFrame([{
                    "Visit #": visit_count,
                    "Date": date_str,
                    "Name": fname,
                    "Last name": lname,
                    "Time": login_time,
                    "Mask": mask_status,
                    "Hat": hat_status,
                    "Washing Complete": wash_status,
                    "All WHO Steps": all_steps  # <-- NEW: Data column added here
                }])

                if not df.empty:
                    df = pd.concat([df, new_data], ignore_index=True)
                else:
                    df = new_data

                df.to_excel(excel_file, index=False)
            logging.info(f"[LOG] Saved Visit #{visit_count} for {current_user} in {excel_file}")
            return True
        except Exception as e:
            logging.error(f"[ERROR] Could not save to personal Excel: {e}")
            return False

    @staticmethod
    def send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status, all_steps):
        try:
            bot_message = (
                f"  *Smart PPE Alert*\n"
                f"  User: {current_user}\n"
                f"  Time: {login_time}\n"
                f"  Mask: {mask_status}\n"
                f"  Hat: {hat_status}\n"
                f"  Washing Complete: {wash_status}\n"
                f"  All WHO Steps: {all_steps}"  # <-- NEW: Added to the bot message
            )
            payload = {"chat_id": config.BOT_CHAT_ID, "text": bot_message}
            response = requests.post(config.BOT_API_URL, json=payload, timeout=config.BOT_TIMEOUT)
            return response.status_code == 200
        except:
            return False

    def log_and_notify(self, current_user, login_time, wash_status, mask_status, hat_status, all_steps):
        self.log_session(current_user, login_time, wash_status, mask_status, hat_status, all_steps)
        self.send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status, all_steps)


class UserSessionManager:
    """Manages user authentication and session state."""
    def __init__(self):
        self.current_user = None
        self.login_time = None
        self.is_authenticating = False
        self.last_person_seen_time = time.time()
        self.last_auth_attempt_time = 0

    def set_user(self, user_name):
        self.current_user = user_name.replace("_", " ")
        self.login_time = time.strftime("%H:%M:%S")
        self.last_person_seen_time = time.time()

    def clear_user(self):
        self.current_user = None
        self.login_time = None

    def is_authenticated(self):
        return self.current_user is not None

    def check_presence_timeout(self, timeout_seconds=config.PRESENCE_TIMEOUT):
        if not self.is_authenticated():
            return False
        return (time.time() - self.last_person_seen_time) > timeout_seconds

    def can_attempt_auth(self):
        return (time.time() - self.last_auth_attempt_time) >= config.AUTH_COOLDOWN

    def set_auth_attempt(self):
        self.last_auth_attempt_time = time.time()

    def update_presence(self):
        self.last_person_seen_time = time.time()