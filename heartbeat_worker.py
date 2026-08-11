"""Heartbeat Worker - Sends robust, isolated status updates to Bale group."""

import time
import threading
import logging
import requests
import config
from logger_setup import LogCategory

class HeartbeatThread(threading.Thread):
    """Background daemon that sends resilient system status pings."""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.last_success = 0.0
        self.consecutive_failures = 0
        
        # Base interval for the heartbeat (e.g., every few hours depending on config)
        self.base_interval_sec = getattr(config, 'HEARTBEAT_INTERVAL_SEC', 14400)

    def run(self):
        logging.info(f"{LogCategory.NETWORK} System Heartbeat Thread started.")
        
        # Send an initial boot-up heartbeat to announce the kiosk is online
        self._attempt_heartbeat()

        while self.running:
            # 1. Determine wait time based on network health
            if self.consecutive_failures > 0:
                # Exponential backoff: 1m, 2m, 4m, 8m... capped at a 1-hour max wait
                wait_time = min(60 * (2 ** (self.consecutive_failures - 1)), 3600)
                logging.info(f"{LogCategory.NETWORK} Heartbeat backoff active. Retrying in {wait_time}s...")
            else:
                wait_time = self.base_interval_sec

            # 2. Sleep in interruptible 1-second chunks (allows fast shutdown)
            for _ in range(int(wait_time)):
                if not self.running:
                    return
                time.sleep(1)

            # 3. Trigger the isolated network call
            if self.running:
                self._attempt_heartbeat()

    def _attempt_heartbeat(self):
        """Attempts the network POST with strict timeouts and comprehensive exception catching."""
        if not getattr(config, 'BOT_API_URL', None):
            return

        try:
            # Format the last successful ping time for the diagnostic message
            if self.last_success > 0:
                success_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_success))
            else:
                success_str = 'N/A'

            payload = {
                "chat_id": getattr(config, 'BOT_CHAT_ID', ''),
                "text": f"🟢 HOSPITAL AI ACTIVE\nUptime Last Success: {success_str}"
            }
            
            # Explicit timeout prevents the thread from hanging on dead sockets
            timeout_sec = getattr(config, 'BOT_TIMEOUT', 10)
            resp = requests.post(config.BOT_API_URL, json=payload, timeout=timeout_sec)
            
            if resp.status_code == 200:
                self.last_success = time.time()
                self.consecutive_failures = 0
                logging.info(f"{LogCategory.NETWORK} Heartbeat sent successfully.")
            else:
                self.consecutive_failures += 1
                logging.warning(f"{LogCategory.NETWORK} Heartbeat rejected (HTTP {resp.status_code}). Failures: {self.consecutive_failures}")
                
        except requests.exceptions.RequestException as e:
            # Catches all network volatility (DNS failures, connection drops, read timeouts)
            self.consecutive_failures += 1
            logging.warning(f"{LogCategory.NETWORK} Heartbeat network timeout/error. Failures: {self.consecutive_failures}")
            
        except Exception as e:
            # Absolute fallback to prevent thread death from malformed payloads/JSON errors
            self.consecutive_failures += 1
            logging.error(f"{LogCategory.NETWORK} Unexpected heartbeat exception: {e}")

    def stop(self):
        """Signals the thread to stop safely during a system shutdown."""
        self.running = False
        self.join(timeout=2.0)