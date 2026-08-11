"""Background Event Worker - Unified event-driven background processing."""

import time
import queue
import threading
import logging
import requests
import config
from session_database import SessionDatabase

class EventType:
    """Enumeration of system-wide background events."""
    SESSION_COMPLETED = "SESSION_COMPLETED"
    SEND_NOTIFICATION = "SEND_NOTIFICATION"
    EXPORT_REPORT = "EXPORT_REPORT"


class BackgroundEventWorker(threading.Thread):
    """Singleton: Processes all non-real-time events from a centralized queue."""
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(BackgroundEventWorker, cls).__new__(cls)
                
                # --- FIX: Initialize the parent thread class ---
                threading.Thread.__init__(cls._instance)
                # -----------------------------------------------
                
                cls._instance.daemon = True
                cls._instance.event_queue = queue.Queue()
                cls._instance.running = True
                cls._instance.start()
            return cls._instance
        
    def __init__(self):
        # Initialization handled in __new__
        pass

    def run(self):
        """Main loop that continuously pulls events from the queue."""
        logging.info("[EVENT WORKER] Unified Background Event Worker Started.")
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                event_type = event.get('type')
                payload = event.get('payload', {})

                if event_type == EventType.SESSION_COMPLETED:
                    self._handle_session_completed(payload)
                elif event_type == EventType.SEND_NOTIFICATION:
                    self._handle_send_notification(payload)
                elif event_type == EventType.EXPORT_REPORT:
                    self._handle_export_report(payload)
                else:
                    logging.warning(f"[EVENT WORKER] Unknown event type: {event_type}")

                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"[EVENT WORKER ERROR] Unhandled exception processing {event.get('type')}: {e}")

    def emit(self, event_type, payload):
        """Public method for other threads to drop an event into the background queue."""
        self.event_queue.put({'type': event_type, 'payload': payload})

    def _handle_session_completed(self, payload):
        """Inserts session into the SQLite database."""
        db = SessionDatabase()
        success = db.insert_session(
            date_str=payload.get('date_str'),
            user=payload.get('user'),
            role=payload.get('role'),
            login_time=payload.get('login_time'),
            mask=payload.get('mask'),
            hat=payload.get('hat'),
            wash_complete=payload.get('wash_complete'),
            wash_duration=payload.get('wash_duration'),
            who_steps=payload.get('who_steps'),
            sink=payload.get('sink')
        )
        if success:
            logging.info(f"[EVENT SUCCESS] Logged session for {payload.get('user')} on {payload.get('sink')}")

    def _handle_send_notification(self, payload):
        """Sends HTTP POST payload to the configured Bale Bot with retry backoff."""
        if not getattr(config, 'BOT_API_URL', None):
            return

        max_retries = getattr(config, 'NETWORK_MAX_RETRIES', 3)
        backoff = getattr(config, 'NETWORK_BACKOFF_BASE', 2.0)

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(config.BOT_API_URL, json=payload, timeout=config.BOT_TIMEOUT)
                if response.status_code == 200:
                    logging.info("[BOT NOTIFY SUCCESS] Alert sent successfully.")
                    return
                else:
                    logging.warning(f"[BOT NOTIFY WARNING] Attempt {attempt} failed: HTTP {response.status_code}")
            except requests.exceptions.RequestException as e:
                logging.warning(f"[BOT NOTIFY ERROR] Attempt {attempt} network error: {e}")

            if attempt < max_retries:
                logging.info(f"[BOT NOTIFY] Waiting {backoff} seconds before retry...")
                time.sleep(backoff)
                backoff *= 2  # Exponentially increase wait time (2s -> 4s -> 8s)

        logging.error(f"[BOT NOTIFY FATAL] All {max_retries} attempts failed. Logging failure and continuing.")

    def _handle_export_report(self, payload):
        """Placeholder for triggering on-demand Excel exports."""
        report_type = payload.get('report_type')
        target_date = payload.get('target_date')
        logging.info(f"[EVENT WORKER] Triggering {report_type} export for {target_date}...")
        # Future step: Wire this up to the report generators in send_daily_reports.py

    def stop(self):
        """Signals the worker to shut down safely and unblocks the queue."""
        self.running = False
        # Drop a dummy event to instantly wake up the queue.get() blocking call
        self.event_queue.put({'type': 'SHUTDOWN', 'payload': {}})
        self.join(timeout=3.0)