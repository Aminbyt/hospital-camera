"""Heartbeat Worker - Sends hourly status updates to Bale group."""

import requests
import logging
from PyQt5.QtCore import QThread
import config


class HeartbeatThread(QThread):
    """Background thread that sends an hourly heartbeat message to the Bale group."""

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        logging.info("[HEARTBEAT] Hourly system heartbeat thread initialized.")
        # Send an immediate heartbeat message on application startup
        self.send_heartbeat()

        while self.running:
            # Sleep for 1 hour (3600 seconds) in interruptible 1-second steps
            for _ in range(3600):
                if not self.running:
                    return
                self.msleep(1000)

            if self.running:
                self.send_heartbeat()

    def send_heartbeat(self):
        try:
            payload = {
                "chat_id": config.BOT_CHAT_ID,
                "text": "🟢 ACTIVATE"
            }
            resp = requests.post(
                config.BOT_API_URL, 
                json=payload, 
                timeout=config.BOT_TIMEOUT
            )
            if resp.status_code == 200:
                logging.info("[HEARTBEAT] Sent '🟢 ACTIVATE' message to Bale group.")
            else:
                logging.warning(f"[HEARTBEAT FAILED] Bot returned status code {resp.status_code}")
        except Exception as e:
            logging.error(f"[HEARTBEAT ERROR] Could not send heartbeat: {e}")

    def stop(self):
        self.running = False
        self.wait()