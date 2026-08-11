"""Camera Worker Module - Background thread for AI processing with robust auto-reconnect."""

import cv2
import time
import logging
import threading
import numpy as np
from PyQt5.QtCore import QThread

import config
from ai_models import AIModels, FaceRecognitionService
from hand_wash_detector import HandWashDetector
from background_worker import BackgroundEventWorker, EventType
from session_manager import UserSessionManager
from sink_state import SinkState
from logger_setup import LogCategory
import logging



class CameraState:
    """Explicit camera connectivity states."""
    CONNECTING = "CAMERA_CONNECTING"
    CONNECTED = "CAMERA_CONNECTED"
    DISCONNECTED = "CAMERA_DISCONNECTED"
    RECONNECTING = "CAMERA_RECONNECTING"
    ERROR = "CAMERA_ERROR"


# Explicit backoff retry schedule in seconds (1s, 2s, 4s, 8s, 15s, 30s max)
BACKOFF_SCHEDULE = [1, 2, 4, 8, 15, 30]


def get_backoff_sec(reconnect_attempt):
    """Calculates backoff delay based on the attempt count."""
    idx = min(max(0, reconnect_attempt - 1), len(BACKOFF_SCHEDULE) - 1)
    return BACKOFF_SCHEDULE[idx]


class ZeroLatencyGrabber:
    """Dedicated high-speed thread tracking latest frame, connection state, and auto-reconnect."""

    def __init__(self, src, failure_threshold=5):
        self.src = src
        self.failure_threshold = failure_threshold
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Frame State (Always latest frame only)
        self.latest_frame = None
        self.frame_timestamp = 0.0
        self.frame_id = 0

        # Connection & State Machine
        self.state = CameraState.CONNECTING
        self.last_success_time = time.time()
        self.consecutive_failures = 0
        self.reconnect_attempts = 0

        self.thread = None

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        return self

    def _open_stream(self):
        """Attempts to open the OpenCV VideoCapture stream."""
        try:
            if isinstance(self.src, int):
                stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)
                stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            else:
                stream = cv2.VideoCapture(self.src)

            stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if stream.isOpened():
                return stream
        except Exception as e:
            logging.error(f"[CAMERA ERROR] Exception opening source ({self.src}): {e}")
            with self.lock:
                self.state = CameraState.ERROR

        return None

    def _update_loop(self):
        """Continuous background loop clearing hardware buffers and managing recovery."""
        with self.lock:
            self.state = CameraState.CONNECTING

        stream = self._open_stream()

        while not self.stop_event.is_set():
            if stream is None or not stream.isOpened():
                with self.lock:
                    self.reconnect_attempts += 1
                    self.state = (
                        CameraState.RECONNECTING
                        if self.reconnect_attempts > 1
                        else CameraState.DISCONNECTED
                    )

                backoff = get_backoff_sec(self.reconnect_attempts)
                logging.warning(
                    f"[{self.state}] Camera source ({self.src}) offline. "
                    f"Attempt #{self.reconnect_attempts}. Retrying in {backoff}s..."
                )

                if self.stop_event.wait(timeout=backoff):
                    break

                stream = self._open_stream()
                continue

            # Read frame off hardware buffer
            ret, frame = stream.read()

            if ret and frame is not None:
                now = time.time()
                with self.lock:
                    self.latest_frame = frame
                    self.frame_timestamp = now
                    self.frame_id += 1
                    self.state = CameraState.CONNECTED
                    self.last_success_time = now
                    self.consecutive_failures = 0
                    self.reconnect_attempts = 0
            else:
                with self.lock:
                    self.consecutive_failures += 1

                if self.consecutive_failures >= self.failure_threshold:
                    with self.lock:
                        self.state = CameraState.DISCONNECTED
                    logging.warning(
                        f"[CAMERA DISCONNECTED] {self.consecutive_failures} consecutive frame drops on ({self.src}). "
                        f"Releasing handle to trigger reconnect schedule."
                    )
                    if stream:
                        stream.release()
                    stream = None
                else:
                    time.sleep(0.01)

        if stream:
            stream.release()
        with self.lock:
            self.state = CameraState.DISCONNECTED
        logging.info(f"[CAMERA CLEANUP] Released video capture handle for ({self.src}).")

    def read(self):
        """Returns thread-safe latest frame copy along with frame_id and timestamp."""
        with self.lock:
            if self.state == CameraState.CONNECTED and self.latest_frame is not None:
                return True, self.latest_frame.copy(), self.frame_id, self.frame_timestamp
            return False, None, self.frame_id, self.frame_timestamp

    def get_status(self):
        """Returns diagnostic metrics for watchdog monitoring."""
        with self.lock:
            return {
                'state': self.state,
                'connected': (self.state == CameraState.CONNECTED),
                'frame_id': self.frame_id,
                'frame_timestamp': self.frame_timestamp,
                'last_success_time': self.last_success_time,
                'consecutive_failures': self.consecutive_failures,
                'reconnect_attempts': self.reconnect_attempts
            }

    def stop(self):
        """Triggers thread shutdown and releases OpenCV handles."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class CameraWorker(QThread):

    def __init__(self, sink_name, camera_index):
        super().__init__()
        self.sink_name = sink_name
        self.camera_index = camera_index
        self.running = True
        self.video_stream = None
        self.last_heartbeat = time.monotonic()
        self.state = SinkState()

        self.ai_models = AIModels()
        self.wash_detector = HandWashDetector()
        self.session_manager = UserSessionManager()

        self.sink_y_start = None
        self.scrub_roi = None
        self.check_mask = True
        self.check_hat = True
        self.check_wash = True

        self.auth_check_counter = 0
        self.auth_message = "WAITING FOR FACE..."
        self.auth_color = "normal"

        self.state = SinkState()
        self.ui_frame = None
        self.raw_ui_frame = None

    def run(self):
        """This runs continuously in the background QThread."""
        logging.info(f"[INFO] Initializing Camera Worker for {self.sink_name}...")
        self.video_stream = ZeroLatencyGrabber(
            self.camera_index,
            failure_threshold=getattr(config, 'CAMERA_TIMEOUT_SEC', 5)
        ).start()

        last_processed_id = -1 

        while self.running:
            ret, frame, frame_id, frame_ts = self.video_stream.read()
            cam_status = self.video_stream.get_status()

            if not ret or frame is None:
                # Update state for disconnected camera
                self.state.connection_status = cam_status['state']
                self.state.user = self.session_manager.current_user if self.session_manager.current_user else "EMPTY"
                self.state.is_authenticated = self.session_manager.is_authenticated()
                self.state.auth_message = f"[{cam_status['state']}]"
                self.state.auth_color = "error"
                self.state.has_mask = False
                self.state.has_hat = False
                self.state.wash_time = self.wash_detector.current_wash_time
                self.state.wash_status_text = "CAMERA OFFLINE"
                self.state.master_ready = False
                
                time.sleep(0.05)
                continue
            
            if frame_id == last_processed_id:
                time.sleep(0.01)
            last_processed_id = frame_id

            frame_h, frame_w = frame.shape[:2]
            clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            # --- 1. RATE-LIMITED FACE DETECTION (GATEKEEPER) ---
            if (now - getattr(self, 'last_face_check', 0.0)) >= (1.0 / getattr(config, 'FACE_FPS', 5)):
                self.cached_has_face = self.ai_models.detect_face(clean_rgb)
                self.last_face_check = now
            
            has_any_face = getattr(self, 'cached_has_face', False)

            # --- 2. RATE-LIMITED HAND DETECTION (ONLY IF AUTHENTICATED) ---
            new_hand_data = False
            if self.session_manager.is_authenticated():
                if (now - getattr(self, 'last_hand_check', 0.0)) >= (1.0 / getattr(config, 'HAND_FPS', 12)):
                    self.cached_hand_results = self.ai_models.detect_hands(clean_rgb)
                    self.last_hand_check = now
                    new_hand_data = True
            else:
                self.cached_hand_results = {'detected': False, 'hand_results': None, 'count': 0}
                
            hand_results = getattr(self, 'cached_hand_results', {'detected': False, 'hand_results': None, 'count': 0})

            # --- 3. PRESENCE TRACKING ---
            # Reset logout timer if user is actively present
            if self.session_manager.is_authenticated() and (has_any_face or hand_results['detected']):
                self.session_manager.update_presence()
            if self.session_manager.check_presence_timeout():
                logging.info(f"[{self.sink_name}] Auto-logging out due to inactivity.")
                self.logout_user()

# 1. HEARTBEAT & LOGIN CHECK
            if has_any_face:
                self.auth_check_counter += 1
                if self.auth_check_counter % 30 == 0:
                    if not self.session_manager.is_authenticating and self.session_manager.can_attempt_auth():
                        self.session_manager.is_authenticating = True
                        self.auth_message = "SCANNING FACE..."
                        self.auth_color = "warning"

                        # Push to the central queue and supply the callback to handle the result
                        FaceRecognitionService().request_recognition(
                            self.sink_name, 
                            frame.copy(), 
                            self.handle_auth_result
                        )

            # 2. ALCOHOL SCRUB ZONE (SPLIT SCREEN 50/50)
            self.sink_y_start = int(frame_h * 0.5)
            cv2.line(frame, (0, self.sink_y_start), (frame_w, self.sink_y_start), (0, 0, 255), 2)
            cv2.putText(
                frame, "ALCOHOL SCRUB ZONE", (10, self.sink_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

            # 3. & 4. HEAVY AI (PPE & WASH)
            has_mask, has_hat = False, False

            if self.session_manager.is_authenticated():
                if self.check_mask or self.check_hat:
                    frame, has_mask, has_hat = self.ai_models.detect_ppe(frame)

# 4. HAND WASHING
                if self.check_wash and hand_results['detected']:
                    frame = self.ai_models.draw_hand_landmarks(frame, hand_results['hand_results'])
                    
                    # --- RESTORED EXACT OLD LOGIC FROM MAIN BRANCH ---
                    wash_info = self.wash_detector.detect_washing(
                        hand_results, frame_w, frame_h, self.sink_y_start, self.ai_models
                    )
                
                    # ---> PREDICT LIVE WHO GESTURE FIRST <---
                    if wash_info['actively_washing']:
                        
                        # --- FIX: Only push to the LSTM model if we have a FRESH moving hand frame! ---
                        if new_hand_data:
                            current_who_step = self.ai_models.predict_who_step(hand_results['hand_results'])
                            self.cached_who_step = current_who_step
                        else:
                            current_who_step = getattr(self, 'cached_who_step', 0)
                        # ------------------------------------------------------------------------------
                    
                        is_valid_who_step = (1 <= current_who_step <= 6)
                        
                        # Timer ticks up for ANY scrubbing, ignoring WHO validity
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
            # 5. DETERMINE MASTER STATUS
            master_ready = False
            if self.session_manager.is_authenticated():
                master_ready = True
                if self.check_mask and not has_mask:
                    master_ready = False
                if self.check_hat and not has_hat:
                    master_ready = False
                if self.check_wash and self.wash_detector.current_wash_time < config.MIN_WASH_TIME:
                    master_ready = False



            # 6. SEND DATA BACK TO UI
# --- UPDATE STATE OBJECT IN-PLACE (O(1) Memory) ---
            self.state.connection_status = "CONNECTED"
            self.state.last_frame_time = time.monotonic()
            
            self.state.user = self.session_manager.current_user if self.session_manager.current_user else "EMPTY"
            self.state.is_authenticated = self.session_manager.is_authenticated()
            self.state.auth_message = self.auth_message
            self.state.auth_color = self.auth_color
            
            self.state.check_mask = self.check_mask
            self.state.check_hat = self.check_hat
            self.state.check_wash = self.check_wash
            
            self.state.has_mask = has_mask
            self.state.has_hat = has_hat
            
            self.state.wash_time = self.wash_detector.current_wash_time
            self.state.wash_status_text = self.wash_detector.get_wash_status(hand_results.get('count', 0)) if hand_results else "STANDBY"
            self.state.current_who_step = getattr(self, 'cached_who_step', 0)
            self.state.completed_who_steps = self.wash_detector.completed_steps.copy()
            self.state.master_ready = master_ready

            # --- ATOMIC ASSIGNMENTS FOR UI PULL MODEL ---
            self.ui_frame = frame
            self.last_heartbeat = time.monotonic()

        # Cleanup when thread stops
        if self.video_stream:
            self.video_stream.stop()

    def handle_auth_result(self, result):
        self.session_manager.is_authenticating = False
        self.session_manager.set_auth_attempt()
        clean_result = result.replace("_", " ")

        if result in ("NO_FACE", "UNKNOWN"):
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
        """Concludes active user session and instantly drops an event into the background queue."""
        
        if self.session_manager.is_authenticated():
            # Calculate final session metrics
            wash_duration = int(self.wash_detector.current_wash_time)
            wash_status = "YES" if self.wash_detector.current_wash_time >= config.MIN_WASH_TIME else "NO"
            mask_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_mask_seen_time) <= 3.0 else "NO"
            hat_status = "YES" if (self.session_manager.last_person_seen_time - self.wash_detector.last_hat_seen_time) <= 3.0 else "NO"
            all_steps = "YES" if len(self.wash_detector.completed_steps) >= 4 else "NO"
            
            user_role = UserSessionManager.get_user_role(self.session_manager.current_user)
            # --- 1. FIRE SESSION COMPLETED EVENT ---
            session_payload = {
                'date_str': time.strftime("%Y-%m-%d"),
                'user': self.session_manager.current_user,
                'role': user_role,
                'login_time': self.session_manager.login_time,
                'mask': mask_status,
                'hat': hat_status,
                'wash_complete': wash_status,
                'wash_duration': wash_duration,
                'who_steps': all_steps,
                'sink': self.sink_name
            }
            BackgroundEventWorker().emit(EventType.SESSION_COMPLETED, session_payload)

            # --- 2. FIRE NOTIFICATION EVENT ---
            bot_message = (
                f"🛡️ *Smart PPE Alert*\n"
                f"👤 User: {self.session_manager.current_user}\n"
                f"💼 Role: {user_role}\n"
                f"🕒 Time: {self.session_manager.login_time}\n"
                f"😷 Mask: {mask_status}\n"
                f"🧢 Hat: {hat_status}\n"
                f"🧼 Washing Complete: {wash_status}\n"
                f"⏱️ Wash Duration: {wash_duration}s\n"
                f"✅ All WHO Steps: {all_steps}"
            )
            bot_payload = {"chat_id": config.BOT_CHAT_ID, "text": bot_message}
            BackgroundEventWorker().emit(EventType.SEND_NOTIFICATION, bot_payload)

        # --- 3. INSTANT STATE RESET ---
        self.session_manager.clear_user()
        self.wash_detector.reset_state()
        self.wash_detector.current_wash_time = 0.0
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

        if self.video_stream:
            self.video_stream.stop()
        self.wait()