"""User Session Management Module - Pure state tracking, zero networking."""

import os
import json
import time
import config

class UserSessionManager:
    """Manages user authentication and session state for a camera worker."""
    def __init__(self):
        self.current_user = None
        self.login_time = None
        self.is_authenticating = False
        self.last_person_seen_time = time.time()
        self.last_auth_attempt_time = 0

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