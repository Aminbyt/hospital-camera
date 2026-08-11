"""Headless AI Engine - Runs background workers and broadcasts via ZeroMQ."""

import sys
import os

# --- CRITICAL BOOT SEQUENCE: FIX DLL LOADING ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2' 

try:
    import onnxruntime
    import insightface
except Exception as e:
    pass
# -----------------------------------------------
import time
import logging
import json
import cv2
import zmq
import config

from logger_setup import setup_logger, LogCategory
from camrea_worker import CameraWorker
from background_worker import BackgroundEventWorker
from health_monitor import HealthMonitor
from send_daily_reports import DailyReportThread, MonthlyReportThread
from heartbeat_worker import HeartbeatThread
from sink_calibration import SinkCalibrationDialog

class HospitalKioskService:
    """Master background service running AI, Cameras, and IPC."""
    
    def __init__(self):
        # 1. Setup ZMQ Publisher (IPC Layer)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # Bind to all interfaces on port 5555
        self.socket.bind("tcp://*:5555") 
        logging.info(f"{LogCategory.NETWORK} ZMQ Publisher bound to tcp://*:5555")

        from ai_models import FaceRecognitionService
        logging.info("[BOOT] Pre-loading Face Recognition AI into RAM...")
        FaceRecognitionService()
        # 2. Initialize Camera Workers
        self.workers = {}
        for sink_id, cam_index in config.SINK_CAMERAS.items():
            worker = CameraWorker(sink_name=sink_id, camera_index=cam_index)
            self.workers[sink_id] = worker
            worker.start()

        # 3. Initialize Auxiliary & Watchdog Threads
        self.health_monitor = HealthMonitor(
            self.workers,
            worker_recovery_cb=self.recover_camera_worker,
            ai_recovery_cb=self.recover_ai_engine
        )
        self.health_monitor.start()

        self.report_thread = DailyReportThread()
        self.report_thread.start()

        self.monthly_thread = MonthlyReportThread()
        self.monthly_thread.start()

        self.heartbeat_thread = HeartbeatThread()
        self.heartbeat_thread.start()

        self.running = True

    def recover_camera_worker(self, sink_id):
        """Tier 2 Recovery logic."""
        logging.info(f"{LogCategory.WATCHDOG} Recovering worker {sink_id}...")
        old_worker = self.workers.get(sink_id)
        if old_worker:
            old_worker.running = False
            cam_index = old_worker.camera_index
        else:
            cam_index = config.SINK_CAMERAS.get(sink_id)

        new_worker = CameraWorker(sink_name=sink_id, camera_index=cam_index)
        self.workers[sink_id] = new_worker
        new_worker.start()

    def recover_ai_engine(self):
        """Tier 3 Recovery logic."""
        logging.info(f"{LogCategory.WATCHDOG} Resetting AI Engine...")
        try:
            from ai_models import AIModels
            AIModels().manager._initialize_models()
        except Exception as e:
            logging.critical(f"{LogCategory.WATCHDOG} AI reset failed: {e}")

    def run_publisher_loop(self):
        """Continuously pulls data from workers and broadcasts it over ZMQ."""
        logging.info(f"{LogCategory.NETWORK} Starting ZMQ broadcast loop...")
        
        while self.running:
            for sink_id, worker in self.workers.items():
                state = worker.state
                frame = worker.ui_frame
                
                # 1. Serialize State to Dictionary
                state_dict = {
                    "connection_status": state.connection_status,
                    "user": state.user,
                    "is_authenticated": state.is_authenticated, # For Home Tab
                    "is_auth": state.is_authenticated,          # For Dashboard Tab
                    "auth_message": state.auth_message,         # For Home Tab
                    "auth_msg": state.auth_message,             # For Dashboard Tab
                    "auth_color": state.auth_color,
                    "check_mask": state.check_mask,
                    "check_hat": state.check_hat,
                    "check_wash": state.check_wash,
                    "has_mask": state.has_mask,                 # For Home Tab
                    "mask": state.has_mask,                     # For Dashboard Tab
                    "has_hat": state.has_hat,                   # For Home Tab
                    "hat": state.has_hat,                       # For Dashboard Tab
                    "wash_time": state.wash_time,
                    "wash_status_text": state.wash_status_text, # For Home Tab
                    "wash_status": state.wash_status_text,      # For Dashboard Tab
                    "master_ready": state.master_ready
                }
                # 2. Compress Frame to JPEG Bytes
# --- FIX 3A: Send raw bytes directly (Zero CPU overhead) ---
                if frame is not None:
                    state_dict["frame_shape"] = frame.shape # Send dimensions so UI knows how to rebuild it
                    frame_bytes = frame.tobytes()
                else:
                    state_dict["frame_shape"] = None
                    frame_bytes = b""

                # 3. Send Multipart Message [Topic, JSON, Video Bytes]
                try:
                    self.socket.send_multipart([
                        sink_id.encode('utf-8'),
                        json.dumps(state_dict).encode('utf-8'),
                        frame_bytes
                    ])
                except Exception as e:
                    logging.error(f"{LogCategory.NETWORK} ZMQ Send Error: {e}")

            # Cap the broadcast speed to the target UI FPS (e.g., ~66ms for 15 FPS)
            time.sleep(1.0 / getattr(config, 'UI_FPS', 15))

    def stop(self):
        """Clean teardown of the engine and all its sub-processes."""
        logging.info("[SHUTDOWN] Stopping Headless Engine...")
        self.running = False
        
        if hasattr(self, "health_monitor"):
            self.health_monitor.stop()
            
        for worker in self.workers.values():
            worker.stop()
            
        # --- FIX: Only stop Singletons if they were actually initialized ---
        from ai_models import FaceRecognitionService
        if FaceRecognitionService._instance is not None:
            FaceRecognitionService().stop()
            
        from background_worker import BackgroundEventWorker
        if BackgroundEventWorker._instance is not None:
            BackgroundEventWorker().stop()
        # -------------------------------------------------------------------
        
        if hasattr(self, "report_thread"):
            self.report_thread.stop()
        if hasattr(self, "monthly_thread"):
            self.monthly_thread.stop()
        if hasattr(self, "heartbeat_thread"):
            self.heartbeat_thread.stop()
        
        # Close sockets
        try:
            self.socket.close()
            self.context.term()
        except Exception:
            pass
            
        logging.info("[SHUTDOWN] Engine terminated cleanly.")


if __name__ == "__main__":
    setup_logger()
    service = HospitalKioskService()
    try:
        service.run_publisher_loop()
    except KeyboardInterrupt:

        service.stop()