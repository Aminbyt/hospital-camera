"""Camera Worker Module - Background thread for AI processing."""
import cv2
import time
import threading
from PyQt5.QtCore import QThread, pyqtSignal
import config
from ai_models import AIModels, FaceRecognitionThread
from hand_wash_detector import HandWashDetector
from sink_calibration import SinkCalibration
from data_logger import DataLogger, UserSessionManager

class RTSPGrabber:
    """A dedicated high-speed thread that constantly clears the network buffer."""
    def __init__(self, src):
        self.src = src  
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        stream = cv2.VideoCapture(self.src)
        stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            ret, frame = stream.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
        stream.release()

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.stopped = True


class CameraWorker(QThread):
    # Signals to send data back to the UI safely
    frame_ready = pyqtSignal(object)   
    data_ready = pyqtSignal(str, dict) 
    dashboard_data = pyqtSignal(dict)  

    def __init__(self, sink_name, camera_index):
        super().__init__()
        self.sink_name = sink_name
        self.camera_index = camera_index
        self.running = True

        self.ai_models = AIModels()
        self.wash_detector = HandWashDetector()
        self.session_manager = UserSessionManager()
        self.data_logger = DataLogger()

        self.sink_y_start = None
        self.scrub_roi = None
        self.check_mask = True
        self.check_hat = True
        self.check_wash = True

        self.auth_check_counter = 0
        self.auth_message = "WAITING FOR FACE..."
        self.auth_color = "normal"
        self.video_stream = None 

    def run(self):
        """This runs continuously in the background!"""
        
        if isinstance(self.camera_index, int):
            self.video_stream = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            is_rtsp = False
        else:
            print(f"[INFO] Connecting to IP Camera: {self.sink_name} with Zero-Latency Grabber...")
            self.video_stream = RTSPGrabber(self.camera_index).start()
            is_rtsp = True

        while self.running:
            if is_rtsp:
                ret, frame = self.video_stream.read()
            else:
                ret, frame = self.video_stream.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_h, frame_w = frame.shape[:2]
            clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- THE FIX: PRE-CALCULATE HANDS TO PROVE PRESENCE ---
            hand_results = self.ai_models.detect_hands(clean_rgb)
            has_any_face = self.ai_models.detect_face(clean_rgb)

            # THE MAGIC "STAY ALIVE" RULE: If we see a face OR hands, reset the logout timer!
            if self.session_manager.is_authenticated() and (has_any_face or hand_results['detected']):
                self.session_manager.update_presence()

            # 1. HEARTBEAT & LOGIN CHECK
            if has_any_face:
                self.auth_check_counter += 1
                if self.auth_check_counter % 30 == 0:
                    if not self.session_manager.is_authenticating and self.session_manager.can_attempt_auth():
                        self.session_manager.is_authenticating = True
                        self.auth_message = "SCANNING FACE..."
                        self.auth_color = "warning"
                        self.face_thread = FaceRecognitionThread(frame.copy(), config.REG_PATH)
                        self.face_thread.result_signal.connect(self.handle_auth_result)
                        self.face_thread.start()
            else:
                self.auth_check_counter = 0
                if self.session_manager.check_presence_timeout():
                    self.logout_user()
                    self.auth_message = "WAITING FOR FACE..."
                    self.auth_color = "normal"

            # 2. SINK CALIBRATION
            if self.scrub_roi:
                self.sink_y_start = int(self.scrub_roi[1] * frame_h)
                frame = SinkCalibration.draw_manual_roi(frame, self.scrub_roi)
            else:
                if self.sink_y_start is None:
                    detected_y = SinkCalibration.auto_detect_sink_line(frame)
                    self.sink_y_start = SinkCalibration.calculate_sink_y_start(detected_y, frame_h)

            # 3. PPE DETECTION
            has_mask, has_hat = False, False
            if self.check_mask or self.check_hat:
                frame, has_mask, has_hat = self.ai_models.detect_ppe(frame)

            # 4. HAND WASHING
            if self.check_wash and hand_results['detected']:
                frame = self.ai_models.draw_hand_landmarks(frame, hand_results['hand_results'])
                wash_info = self.wash_detector.detect_washing(
                    hand_results, frame_w, frame_h, self.sink_y_start, self.ai_models
                )
                self.wash_detector.update_wash_time(wash_info['actively_washing'])
                frame = self.wash_detector.draw_bubble_zone(frame)

            # 5. DETERMINE MASTER STATUS
            master_ready = False
            if self.session_manager.is_authenticated():
                master_ready = True
                if self.check_mask and not has_mask: master_ready = False
                if self.check_hat and not has_hat: master_ready = False
                if self.check_wash and self.wash_detector.current_wash_time < config.MIN_WASH_TIME: master_ready = False

            # 6. SEND DATA BACK TO UI
            summary_data = {
                'user': self.session_manager.current_user if self.session_manager.current_user else "EMPTY",
                'is_auth': self.session_manager.is_authenticated(),
                'auth_msg': self.auth_message,
                'auth_color': self.auth_color,
                'mask': has_mask,
                'hat': has_hat,
                'check_mask': self.check_mask,
                'check_hat': self.check_hat,
                'check_wash': self.check_wash,
                'wash_time': self.wash_detector.current_wash_time,
                'wash_status': self.wash_detector.get_wash_status(hand_results.get('count', 0)) if hand_results else "STANDBY",
                'master_ready': master_ready
            }

            self.frame_ready.emit(frame)
            self.data_ready.emit(self.sink_name, summary_data)
            self.dashboard_data.emit(summary_data)

        # Cleanup when stopped
        if is_rtsp:
            self.video_stream.stop()
        else:
            self.video_stream.release()

    def handle_auth_result(self, result):
        self.session_manager.is_authenticating = False
        self.session_manager.set_auth_attempt()
        clean_result = result.replace("_", " ")

        if result == "NO_FACE" or result == "UNKNOWN":
            if not self.session_manager.is_authenticated():
                self.auth_message = "UNKNOWN USER"
                self.auth_color = "error"
            return

        if not self.session_manager.is_authenticated():
            self.session_manager.set_user(clean_result)
            self.wash_detector.reset_state()
            self.auth_message = f"{clean_result} LOGGED IN"
            self.auth_color = "success"
        elif self.session_manager.current_user != clean_result:
            print(f"[{self.sink_name} SWAP DETECTED] {self.session_manager.current_user} left, {clean_result} stepped in!")
            self.logout_user()
            self.session_manager.set_user(clean_result)
            self.wash_detector.reset_state()
            self.auth_message = f"SWAPPED TO {clean_result}"
            self.auth_color = "success"
        else:
            self.session_manager.update_presence()

    def logout_user(self):
        if self.session_manager.is_authenticated():
            wash_status = "YES" if self.wash_detector.current_wash_time >= config.MIN_WASH_TIME else "NO"
            mask_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_mask_seen_time) <= 3.0 else "NO"
            hat_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_hat_seen_time) <= 3.0 else "NO"

            self.data_logger.log_and_notify(
                self.session_manager.current_user,
                self.session_manager.login_time,
                wash_status, mask_status, hat_status
            )
        self.session_manager.clear_user()
        self.wash_detector.reset_state()

    def update_toggles(self, mask, hat, wash):
        self.check_mask = mask
        self.check_hat = hat
        self.check_wash = wash
    
    def set_manual_roi(self, roi):
        self.scrub_roi = roi
        self.sink_y_start = None

    def trigger_calibration(self):
        self.sink_y_start = None
        self.scrub_roi = None

    def stop(self):
        self.running = False
        self.wait()