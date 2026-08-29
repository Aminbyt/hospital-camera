"""Camera Worker Module - Background thread for AI processing."""
import cv2
import time
import numpy as np
import threading
from PyQt5.QtCore import QThread, pyqtSignal
import logging
import config
from ai_models import AIModels, FaceRecognitionThread, recognize_face_sync
from hand_wash_detector import HandWashDetector
from sink_calibration import SinkCalibration
from data_logger import DataLogger, UserSessionManager

class ZeroLatencyGrabber:
    """A dedicated high-speed thread that constantly clears the camera buffer for BOTH IP and USB."""
    def __init__(self, src):
        self.src = src  
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.last_frame_time = time.time()

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        if isinstance(self.src, int):
            # OBS Virtual Camera REQUIRES DirectShow to prevent the 1-second freeze!
            stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
        else:
            stream = cv2.VideoCapture(self.src)
            stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while not self.stopped:
            ret, frame = stream.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
                    self.last_frame_time = time.time()
        stream.release()
    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.stopped = True


class CameraWorker(QThread):
    # Signals to send data back to the UI safely
    frame_ready = pyqtSignal(object)   
    raw_frame_ready = pyqtSignal(str,object)
    data_ready = pyqtSignal(str, dict) 
    dashboard_data = pyqtSignal(dict)  

    def __init__(self, sink_name, camera_index):
        super().__init__()
        self.sink_name = sink_name
        self.camera_index = camera_index
        self.running = True
        self.auth_color = "normal"
        self.video_stream = None

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
        
        logging.info(f"[INFO] Connecting to Camera: {self.sink_name} with Zero-Latency Grabber...")
        
        # We start the universal grabber here for BOTH USB and IP cameras!
        self.video_stream = ZeroLatencyGrabber(self.camera_index).start()       

        last_freeze_check = time.time()

        while self.running:
            now = time.time()
            if now - last_freeze_check >=30.0:
                logging.info(f"[FREEZE_TRACKER] {self.sink_name} Thread is ALIVE and looping.")
                last_freeze_check = now
            if now - self.video_stream.last_frame_time > 5.0:
                logging.warning(f"[WATCHDOG] {self.sink_name} OpenCV stream hung! Force restarting hardware connection...")
                
                # Flag the old thread to stop (even if it's stuck, we abandon it)
                self.video_stream.stop() 
                
                # Spawn a brand new connection
                self.video_stream = ZeroLatencyGrabber(self.camera_index).start()
                
                # Reset the timer so it doesn't trigger again immediately
                self.video_stream.last_frame_time = time.time() 
                
                # Give the hardware a second to warm up before looping
                time.sleep(1) 
                continue
            ret, frame = self.video_stream.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue


            frame_h, frame_w = frame.shape[:2]
            clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- PRE-CALCULATE HANDS TO PROVE PRESENCE ---
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

                        # CALL SYNCHRONOUSLY WITH THREAD LOCK
                        auth_result = recognize_face_sync(frame.copy(), config.REG_PATH)
                        self.handle_auth_result(auth_result)
            else:
                self.auth_check_counter = 0
                if self.session_manager.check_presence_timeout():
                    self.logout_user()
                    self.auth_message = "WAITING FOR FACE..."
                    self.auth_color = "normal"

            # 2. ALCOHOL SCRUB ZONE (SPLIT SCREEN 50/50)
            # ---------------------------------------------------------
            self.sink_y_start = int(frame_h * 0.5)
            cv2.line(frame, (0, self.sink_y_start), (frame_w, self.sink_y_start), (0, 0, 255), 2)
            cv2.putText(frame, "ALCOHOL SCRUB ZONE", (10, self.sink_y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # ---------------------------------------------------------

            # 3. & 4. HEAVY AI (PPE & WASH) - THE "WAKE UP" CHECK
            # ---------------------------------------------------------
            has_mask, has_hat = False, False
           
            # THE MAGIC GATEKEEPER: Only run heavy YOLO math if someone looked at the camera and logged in!
            if self.session_manager.is_authenticated():
              
                # 3. PPE DETECTION
                if self.check_mask or self.check_hat:
                    frame, has_mask, has_hat = self.ai_models.detect_ppe(frame)

                # 4. HAND WASHING
                if self.check_wash and hand_results['detected']:
                    frame = self.ai_models.draw_hand_landmarks(frame, hand_results['hand_results'])
                    wash_info = self.wash_detector.detect_washing(
                        hand_results, frame_w, frame_h, self.sink_y_start, self.ai_models
                    )
                
                    # ---> PREDICT LIVE WHO GESTURE FIRST <---
                    if wash_info['actively_washing']:
                        current_who_step = self.ai_models.predict_who_step(hand_results['hand_results'])
                    
                        is_valid_who_step = (current_who_step >= 1 and current_who_step <= 6)
                        
                        # ---> THE FIX: Timer now ticks up for ANY scrubbing, ignoring WHO validity <---
                        self.wash_detector.update_wash_time(True)

                        # But we still quietly track the WHO steps for the final Bot report!
                        if is_valid_who_step:
                            self.wash_detector.completed_steps.add(current_who_step)
                    
                        step_labels = {
                            0: "PAUSED: Incorrect Gesture / Transition",
                            1: "Step 1: Palm to Palm",
                            2: "Step 2: Right over Left Dorsum",
                            3: "Step 3: Palm to Palm Interlaced",
                            4: "Step 4: Backs of Fingers",
                            5: "Step 5: Thumb Rotation",
                            6: "Step 6: Fingertips"
                        }
                        label_text = step_labels.get(current_who_step, "Detecting...")

                        # Draw banner (Red/Orange if step 0, Bright Green if steps 1-6)
                        bg_color = (27, 67, 50) if is_valid_who_step else (0, 0, 150)
                        border_color = (0, 255, 0) if is_valid_who_step else (0, 165, 255)
                    
                        cv2.rectangle(frame, (20, 30), (460, 80), bg_color, -1)
                        cv2.rectangle(frame, (20, 30), (460, 80), border_color, 2)
                        cv2.putText(frame, label_text, (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
                    else:
                        self.wash_detector.update_wash_time(False)
                        self.ai_models.clear_buffer()
                    
                    frame = self.wash_detector.draw_bubble_zone(frame)
                else:
                    self.wash_detector.update_wash_time(False)
                    self.ai_models.clear_buffer()
            
            else:
                self.wash_detector.reset_state()
                self.ai_models.clear_buffer()
            # ---------------------------------------------------------

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

            self.raw_frame_ready.emit(self.sink_name,frame.copy())
            self.frame_ready.emit(frame.copy())
            self.data_ready.emit(self.sink_name, summary_data)
            self.dashboard_data.emit(summary_data)

        # Cleanup when stopped
        self.video_stream.stop()


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
            logging.info(f"[{self.sink_name} SWAP DETECTED] {self.session_manager.current_user} left, {clean_result} stepped in!")
            self.logout_user()
            self.session_manager.set_user(clean_result)
            self.wash_detector.reset_state()
            self.auth_message = f"SWAPPED TO {clean_result}"
            self.auth_color = "success"
        else:
            self.session_manager.update_presence()

    def logout_user(self):


        if self.session_manager.is_authenticated():
            wash_duration = int(self.wash_detector.current_wash_time)
            wash_status = "YES" if self.wash_detector.current_wash_time >= config.MIN_WASH_TIME else "NO"
            mask_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_mask_seen_time) <= 3.0 else "NO"
            hat_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_hat_seen_time) <= 3.0 else "NO"
            all_steps = "YES" if len(self.wash_detector.completed_steps) >= 4 else "NO"

            self.data_logger.log_and_notify(
                self.session_manager.current_user,
                self.session_manager.login_time,
                wash_status, mask_status, hat_status, all_steps, wash_duration
            )

        # 1. Clear session
        self.session_manager.clear_user()

        # 2. Force reset wash detector values explicitly
        self.wash_detector.reset_state()
        self.wash_detector.current_wash_time = 0.0

        # 3. Clear AI prediction buffers
        self.ai_models.clear_buffer()


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
        # DELETE: if self.recorder.is_recording: self.recorder.stop_recording()
        self.video_stream.stop()
        self.wait()
