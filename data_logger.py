"""Data Logging Module - Handles Excel export and bot notifications."""

import os
import time
import pandas as pd
import requests
import config


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
        """Log a user session to Excel file.
        
        Args:
            current_user: User's display name
            login_time: Login time string (HH:MM:SS)
            wash_status: "YES" or "NO"
            mask_status: "YES" or "NO"
            hat_status: "YES" or "NO"
            
        Returns:
            bool: True if successful
        """
        if not current_user:
            return False

        try:
            date_str = time.strftime("%Y-%m-%d")
            excel_file = os.path.join(config.INFO_PATH, f"{date_str}.xlsx")

            # Parse name
            parts = current_user.split(" ")
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            # Create new record
            new_data = pd.DataFrame([{
                "Date": date_str,
                "Name": fname,
                "Last name": lname,
                "Time": login_time,
                "Mask": mask_status,
                "Hat": hat_status,
                "Washing Complete": wash_status
            }])

            # Append or create Excel file
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                df = pd.concat([df, new_data], ignore_index=True)
                df.to_excel(excel_file, index=False)
            else:
                new_data.to_excel(excel_file, index=False)

            print(f"[LOG] Session logged for {current_user}")
            return True

        except Exception as e:
            print(f"[ERROR] Could not save to Excel: {e}")
            return False

    @staticmethod
    def send_bot_notification(current_user, login_time, wash_status, mask_status, hat_status):
        """Send session summary to bot API.
        
        Args:
            current_user: User's display name
            login_time: Login time string (HH:MM:SS)
            wash_status: "YES" or "NO"
            mask_status: "YES" or "NO"
            hat_status: "YES" or "NO"
            
        Returns:
            bool: True if notification sent successfully
        """
        try:
            parts = current_user.split(" ")
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            # Format message
            bot_message = (
                f"🏥 *Smart PPE Alert*\n"
                f"👤 User: {fname} {lname}\n"
                f"⏰ Time: {login_time}\n"
                f"😷 Mask: {mask_status}\n"
                f"👨‍⚕️ Hat: {hat_status}\n"
                f"🧼 Washing Complete: {wash_status}"
            )

            payload = {
                "chat_id": config.BOT_CHAT_ID,
                "text": bot_message
            }

            # Send notification
            response = requests.post(
                config.BOT_API_URL, 
                json=payload, 
                timeout=config.BOT_TIMEOUT
            )

            if response.status_code == 200:
                print("[LOG] Bot notification sent successfully")
                return True
            else:
                print(f"[WARNING] Bot returned status {response.status_code}")
                return False

        except requests.Timeout:
            print("[WARNING] Bot notification timeout - continuing anyway")
            return False
        except Exception as e:
            print(f"[WARNING] Could not send bot notification: {e}")
            return False

    def log_and_notify(self, current_user, login_time, wash_status, 
                       mask_status, hat_status, send_notification=True):
        """Log session and optionally send notification.
        
        Args:
            current_user: User's display name
            login_time: Login time string
            wash_status: Washing completion status
            mask_status: Mask verification status
            hat_status: Hat verification status
            send_notification: bool - Whether to send bot notification
            
        Returns:
            dict: {'excel_logged': bool, 'notification_sent': bool}
        """
        results = {
            'excel_logged': self.log_session(
                current_user, login_time, wash_status, mask_status, hat_status
            ),
            'notification_sent': False
        }

        if send_notification:
            results['notification_sent'] = self.send_bot_notification(
                current_user, login_time, wash_status, mask_status, hat_status
            )

        return results

    @staticmethod
    def clear_deepface_cache():
        """Clear DeepFace model cache to force re-indexing.
        
        Returns:
            bool: True if cache cleared
        """
        try:
            cache_file = os.path.join(config.REG_PATH, "representations_vgg_face.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print("[LOG] DeepFace cache cleared")
                return True
            return False
        except Exception as e:
            print(f"[WARNING] Could not clear cache: {e}")
            return False


class UserSessionManager:
    """Manages user authentication and session state."""

    def __init__(self):
        """Initialize session manager."""
        self.current_user = None
        self.login_time = None
        self.is_authenticating = False
        self.last_person_seen_time = time.time()
        self.last_auth_attempt_time = 0

    def set_user(self, user_name):
        """Set authenticated user.
        
        Args:
            user_name: User's name
        """
        self.current_user = user_name.replace("_", " ")
        self.login_time = time.strftime("%H:%M:%S")
        self.last_person_seen_time = time.time()
        print(f"[LOG] User logged in: {self.current_user}")

    def clear_user(self):
        """Clear current user session."""
        if self.current_user:
            print(f"[LOG] User logged out: {self.current_user}")
        self.current_user = None
        self.login_time = None

    def is_authenticated(self):
        """Check if a user is authenticated.
        
        Returns:
            bool: True if user is logged in
        """
        return self.current_user is not None

    def check_presence_timeout(self, timeout_seconds=config.PRESENCE_TIMEOUT):
        """Check if user should be auto-logged out due to absence.
        
        Args:
            timeout_seconds: Timeout duration in seconds
            
        Returns:
            bool: True if timeout exceeded (user should logout)
        """
        if not self.is_authenticated():
            return False

        return (time.time() - self.last_person_seen_time) > timeout_seconds

    def can_attempt_auth(self):
        """Check if enough time has passed for another auth attempt.
        
        Returns:
            bool: True if auth attempt is allowed
        """
        return (time.time() - self.last_auth_attempt_time) >= config.AUTH_COOLDOWN

    def set_auth_attempt(self):
        """Record authentication attempt timestamp."""
        self.last_auth_attempt_time = time.time()

    def update_presence(self):
        """Update last presence timestamp."""
        self.last_person_seen_time = time.time()
