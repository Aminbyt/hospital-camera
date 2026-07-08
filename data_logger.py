import os
import time
import pandas as pd
import requests
import threading
import config

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

    def log_session(self, current_user, login_time, wash_status, mask_status, hat_status):
        """Log a user session to a daily Excel file inside that specific person's registration folder."""
        if not current_user:
            return False

        try:
            date_str = time.strftime("%Y-%m-%d")

            # 1. Locate the person's exact folder inside REG_PATH
            # We check both underscore and space formats to guarantee we find their photo folder
            folder_underscores = os.path.join(config.REG_PATH, current_user.replace(" ", "_"))
            folder_spaces = os.path.join(config.REG_PATH, current_user)

            if os.path.exists(folder_underscores):
                person_dir = folder_underscores
            elif os.path.exists(folder_spaces):
                person_dir = folder_spaces
            else:
                # Fallback: if folder is missing for some reason, create it cleanly
                person_dir = folder_underscores
                os.makedirs(person_dir, exist_ok=True)

            # 2. Set the Excel file path directly inside their personal folder
            excel_file = os.path.join(person_dir, f"{date_str}.xlsx")

            parts = current_user.split(" ")
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            # 3. Thread-safe read/write using your global lock
            with EXCEL_LOCK:
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file)
                    visit_count = len(df) + 1  # Increment visit count for today
                else:
                    df = pd.DataFrame()
                    visit_count = 1            # First visit of the day!

                new_data = pd.DataFrame([{
                    "Visit #": visit_count,
                    "Date": date_str,
                    "Name": fname,
                    "Last name": lname,
                    "Time": login_time,
                    "Mask": mask_status,
                    "Hat": hat_status,
                    "Washing Complete": wash_status
                }])

                if not df.empty:
                    df = pd.concat([df, new_data], ignore_index=True)
                else:
                    df = new_data

                df.to_excel(excel_file, index=False)

            print(f"[LOG] Saved Visit #{visit_count} for {current_user} in {excel_file}")
            return True

        except Exception as e:
            print(f"[ERROR] Could not save to personal Excel: {e}")
            return False
    @staticmethod
    def send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status):
        try:
            parts = current_user.split(" ")
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            bot_message = (
                f"🏥 *Smart PPE Alert*\n"
                f"👤 User: {fname} {lname}\n"
                f"⏰ Time: {login_time}\n"
                f"😷 Mask: {mask_status}\n"
                f"👨‍⚕️ Hat: {hat_status}\n"
                f"🧼 Washing Complete: {wash_status}"
            )

            payload = {"chat_id": config.BOT_CHAT_ID, "text": bot_message}
            response = requests.post(config.BOT_API_URL, json=payload, timeout=config.BOT_TIMEOUT)
            return response.status_code == 200
        except:
            return False

    def log_and_notify(self, current_user, login_time, wash_status, mask_status, hat_status):
        self.log_session(current_user, login_time, wash_status, mask_status, hat_status)
        self.send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status)


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