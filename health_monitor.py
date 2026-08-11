"""System Health Monitor & Watchdog - Active escalation and recovery."""

import os
import time
import logging
import psutil
from PyQt5.QtCore import QThread
from logger_setup import LogCategory
import logging



class HealthMonitor(QThread):
    """Logs system vitals and actively recovers hung components."""
    
    def __init__(self, workers_dict, worker_recovery_cb, ai_recovery_cb):
        super().__init__()
        self.workers = workers_dict
        self.worker_recovery_cb = worker_recovery_cb
        self.ai_recovery_cb = ai_recovery_cb
        self.running = True
        
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.process.cpu_percent(interval=None)
        
        # Track recovery escalations
        self.worker_restarts = {sink_id: 0 for sink_id in self.workers.keys()}
        self.last_ai_reset = 0.0

    def run(self):
        logging.info("[WATCHDOG] Active Health Monitor & Watchdog Started.")
        
        while self.running:
            for _ in range(15):  # Check every 15 seconds
                if not self.running:
                    return
                self.msleep(1000)
                
            self._check_watchdog()

    def _check_watchdog(self):
        try:
            # --- 1. SYSTEM METRICS ---
            uptime_sec = int(time.time() - self.start_time)
            hours, remainder = divmod(uptime_sec, 3600)
            minutes, _ = divmod(remainder, 60)
            uptime_str = f"{hours}h{minutes}m"
            
            cpu = self.process.cpu_percent(interval=None) / psutil.cpu_count()
            ram_gb = self.process.memory_info().rss / (1024 ** 3)
            threads = self.process.num_threads()
            
            try:
                handles = self.process.num_handles()
            except AttributeError:
                handles = 0
                
            disk_free_gb = psutil.disk_usage('/').free / (1024 ** 3)

            log_lines = [
                "\n" + "="*55,
                "WATCHDOG HEALTH CHECK",
                f"uptime={uptime_str} cpu={cpu:.1f}% ram={ram_gb:.2f}GB threads={threads} handles={handles} disk_free={disk_free_gb:.1f}GB",
                "-"*55
            ]

            # --- 2. EVALUATE WORKERS & ESCALATE ---
            now = time.monotonic()
            
            for sink_id, worker in self.workers.items():
                state = worker.state
                hb_age = now - getattr(worker, 'last_heartbeat', now)
                
                # ESCALATION TIER 2: Worker Recovery
                if 10.0 < hb_age <= 30.0:
                    hb_status = f"HUNG ({hb_age:.1f}s)"
                    logging.warning(f"[WATCHDOG] {sink_id} heartbeat missing. Triggering Worker Recovery.")
                    self.worker_restarts[sink_id] += 1
                    self.worker_recovery_cb(sink_id)
                    
                # ESCALATION TIER 3: AI Engine Recovery
                elif 30.0 < hb_age <= 60.0:
                    hb_status = f"DEADLOCKED ({hb_age:.1f}s)"
                    if time.time() - self.last_ai_reset > 120:  # Prevent spamming resets
                        logging.error(f"[WATCHDOG] {sink_id} failed to recover. Triggering AI Engine Reset.")
                        self.ai_recovery_cb()
                        self.last_ai_reset = time.time()
                
                # ESCALATION TIER 4: Process Suicide
                elif hb_age > 60.0:
                    logging.critical(f"[WATCHDOG FATAL] {sink_id} unrecoverable. Terminating process.")
                    os._exit(1) # Immediate hard kill. Relies on OS to restart application.
                else:
                    hb_status = f"OK ({hb_age:.1f}s)"
                    self.worker_restarts[sink_id] = 0 # Reset counter on healthy tick

                # Logging stats
                cam_ok = "OK" if state.connection_status == "CONNECTED" else state.connection_status
                time_since_frame = now - state.last_frame_time
                fps = (1.0 / time_since_frame) if time_since_frame > 0 else 0.0
                if fps > 60 or state.connection_status != "CONNECTED": 
                    fps = 0.0
                user = state.user if state.user != "EMPTY" else "IDLE"
                
                log_lines.append(f"{sink_id} thread={hb_status} camera={cam_ok} fps={fps:.1f} user={user}")

            log_lines.append("="*55 + "\n")
            logging.info("\n".join(log_lines))

        except Exception as e:
            logging.error(f"[WATCHDOG ERROR] Monitor failure: {e}")

    def stop(self):
        self.running = False
        self.wait()