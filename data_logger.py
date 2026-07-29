import os
import json
import time
import pandas as pd
import requests
import threading
import config
import logging

EXCEL_LOCK = threading.Lock()

class DataLogger:
    """Manages user session logging to personal and master Excel files, plus bot notifications."""
    def __init__(self):
        self.ensure_directories()

    @staticmethod
    def ensure_directories():
        os.makedirs(config.REG_PATH, exist_ok=True)
        os.makedirs(config.INFO_PATH, exist_ok=True)
        os.makedirs(os.path.join(config.DB_PATH, "LOGS"), exist_ok=True)

    @staticmethod
    def get_user_role(current_user):
        """Reads stored user role from their folder's user_info.json."""
        if not current_user:
            return "N/A"
        clean_name = current_user.replace(" ", "_")
        folder_underscores = os.path.join(config.REG_PATH, clean_name)
        folder_spaces = os.path.join(config.REG_PATH, current_user)

        person_dir = folder_underscores if os.path.exists(folder_underscores) else folder_spaces
        info_file = os.path.join(person_dir, "user_info.json")

        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    data = json.load(f)
                    return data.get("role", "N/A")
            except Exception:
                pass
        return "N/A"

    def log_session(self, current_user, login_time, wash_status, mask_status, hat_status, all_steps, wash_duration):
        if not current_user:
            return False
        try:
            date_str = time.strftime("%Y-%m-%d")
            role = self.get_user_role(current_user)

            # Locate individual user folder
            folder_underscores = os.path.join(config.REG_PATH, current_user.replace(" ", "_"))
            folder_spaces = os.path.join(config.REG_PATH, current_user)
            if os.path.exists(folder_underscores):
                person_dir = folder_underscores
            elif os.path.exists(folder_spaces):
                person_dir = folder_spaces
            else:
                person_dir = folder_underscores
                os.makedirs(person_dir, exist_ok=True)

            clean_name = current_user.replace(" ", "_")
            person_excel = os.path.join(person_dir, f"{clean_name}_{date_str}.xlsx")
            master_excel = os.path.join(config.DB_PATH, "LOGS", f"master_daily_report_{date_str}.xlsx")

            parts = current_user.split(" ", 1)
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            row_data = {
                "Date": date_str,
                "Name": fname,
                "Last name": lname,
                "Role": role,
                "Time": login_time,
                "Mask": mask_status,
                "Hat": hat_status,
                "Washing Complete": wash_status,
                "Wash Duration (s)": int(wash_duration),
                "All WHO Steps": all_steps
            }

            with EXCEL_LOCK:
                # 1. Log to Individual Person Excel
                if os.path.exists(person_excel):
                    df_p = pd.read_excel(person_excel)
                    p_visit = len(df_p) + 1
                else:
                    df_p = pd.DataFrame()
                    p_visit = 1

                p_row = {"Visit #": p_visit}
                p_row.update(row_data)
                df_p_new = pd.DataFrame([p_row])
                df_p = pd.concat([df_p, df_p_new], ignore_index=True) if not df_p.empty else df_p_new
                df_p.to_excel(person_excel, index=False)

                # 2. Log to Master Daily Excel
                if os.path.exists(master_excel):
                    df_m = pd.read_excel(master_excel)
                    m_visit = len(df_m) + 1
                else:
                    df_m = pd.DataFrame()
                    m_visit = 1

                m_row = {"Visit #": m_visit}
                m_row.update(row_data)
                df_m_new = pd.DataFrame([m_row])
                df_m = pd.concat([df_m, df_m_new], ignore_index=True) if not df_m.empty else df_m_new
                df_m.to_excel(master_excel, index=False)

            logging.info(f"[LOG] Saved visit for {current_user} ({role}) - Duration: {int(wash_duration)}s")
            return True
        except Exception as e:
            logging.error(f"[ERROR] Could not save Excel log: {e}")
            return False

    def send_bot_notification(self, current_user, login_time, wash_status, mask_status, hat_status, all_steps, wash_duration):
        try:
            role = self.get_user_role(current_user)
            bot_message = (
                f"🏥 *Smart PPE Alert*\n"
                f"👤 User: {current_user}\n"
                f"💼 Role: {role}\n"
                f"⏰ Time: {login_time}\n"
                f"😷 Mask: {mask_status}\n"
                f"🧑‍⚕️ Hat: {hat_status}\n"
                f"🧼 Washing Complete: {wash_status}\n"
                f"⏱️ Wash Duration: {int(wash_duration)}s\n"
                f"✅ All WHO Steps: {all_steps}"
            )
            payload = {"chat_id": config.BOT_CHAT_ID, "text": bot_message}
            response = requests.post(config.BOT_API_URL, json=payload, timeout=config.BOT_TIMEOUT)
            return response.status_code == 200
        except:
            return False

    def log_and_notify(self, current_user, login_time, wash_status, mask_status, hat_status, all_steps, wash_duration):
        self.log_session(current_user, login_time, wash_status, mask_status, hat_status, all_steps, wash_duration)
        self.send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status, all_steps, wash_duration)


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