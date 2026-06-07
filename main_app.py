"""Main Application - Hospital AI Smart Scrub Sink Kiosk."""

import sys
import os
import cv2
import time
import json

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QMessageBox
)
from PyQt5.QtCore import QTimer, QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QIcon

import config
from ai_models import AIModels, FaceRecognitionThread
from hand_wash_detector import HandWashDetector
from sink_calibration import SinkCalibration, create_roi_dialog
from data_logger import DataLogger, UserSessionManager
from ui_dashboard_tab import DashboardTab
from ui_registration_tab import RegistrationTab
from ui_settings_tab import SettingsTab


class ScrubSinkKiosk(QMainWindow):
    """Main application window for Hospital AI Smart Scrub Sink."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hospital AI - Smart Scrub Sink")
        self.setGeometry(100, 100, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.setStyleSheet(config.STYLESHEET)

        # --- INITIALIZE COMPONENTS ---
        print("[DEBUG] Initializing AI Models...")
        self.ai_models = AIModels()

        print("[DEBUG] Initializing Hand Wash Detector...")
        self.wash_detector = HandWashDetector()

        print("[DEBUG] Initializing Data Logger...")
        self.data_logger = DataLogger()

        print("[DEBUG] Initializing Session Manager...")
        self.session_manager = UserSessionManager()

        # --- SETTINGS ---
        self.settings = QSettings("HospitalAI", "KioskConfig")
        roi_data = self.settings.value("scrub_roi", None)
        self.scrub_roi = json.loads(roi_data) if roi_data else None
        self.sink_y_start = None

        # Detection toggles
        self.check_mask = True
        self.check_hat = True
        self.check_wash = True

        # --- UI TABS ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.dashboard = DashboardTab()
        self.registration = RegistrationTab()
        self.settings_tab = SettingsTab()

        self.tabs.addTab(self.dashboard, "SYSTEM DASHBOARD")
        self.tabs.addTab(self.registration, "REGISTRATION")
        self.tabs.addTab(self.settings_tab, "SETTINGS")

        # Connect signals
        self.dashboard.auth_requested.connect(self.start_authentication)
        self.settings_tab.camera_source_changed.connect(self.apply_camera_source)
        self.settings_tab.roi_requested.connect(self.enter_roi_mode)
        self.settings_tab.calibration_requested.connect(self.trigger_calibration)
        self.settings_tab.toggles_changed.connect(self.update_toggles)

        # --- CAMERA SETUP ---
        print("[DEBUG] Initializing Camera...")
        self.current_source = 0
        self.cap = cv2.VideoCapture(self.current_source, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        self.timer.start(delay)

        print("[DEBUG] Application initialized successfully!")

    def start_authentication(self):
        """Initiate face recognition authentication."""
        if self.registration.last_clean_frame is None or self.session_manager.is_authenticating:
            return

        if not self.session_manager.can_attempt_auth():
            return

        self.session_manager.is_authenticating = True
        self.dashboard.set_auto_status("SCANNING FACE... PLEASE HOLD STILL", "warning")

        # Start background thread
        self.face_thread = FaceRecognitionThread(self.registration.last_clean_frame, config.REG_PATH)
        self.face_thread.result_signal.connect(self.handle_auth_result)
        self.face_thread.start()

    def handle_auth_result(self, result):
        """Handle face recognition result.
        
        Args:
            result: "UNKNOWN", "NO_FACE", or matched user name
        """
        self.session_manager.is_authenticating = False
        self.session_manager.set_auth_attempt()

        if result == "NO_FACE":
            self.dashboard.set_auto_status("AUTO-SCAN: NO FACE DETECTED", "normal")
        elif result == "UNKNOWN":
            self.dashboard.set_auto_status("AUTO-SCAN: UNKNOWN USER", "error")
        else:
            # Success
            self.session_manager.set_user(result)
            self.dashboard.set_identity(self.session_manager.current_user, True)
            self.dashboard.set_auto_status("AUTO-SCAN: LOGGED IN", "success")
            self.dashboard.unlock_dashboard()
            self.wash_detector.reset_state()

    def logout_user(self):
        """Log out current user and save session data."""
        if self.session_manager.is_authenticated():
            # Calculate status
            wash_status = "YES" if self.wash_detector.current_wash_time >= config.MIN_WASH_TIME else "NO"
            mask_status = "YES" if (self.session_manager.last_person_seen_time - 
                                   self.wash_detector.last_mask_seen_time) <= 3.0 else "NO"
            hat_status = "YES" if (self.session_manager.last_person_seen_time - 
                                  self.wash_detector.last_hat_seen_time) <= 3.0 else "NO"

            # Log and notify
            self.data_logger.log_and_notify(
                self.session_manager.current_user,
                self.session_manager.login_time,
                wash_status,
                mask_status,
                hat_status
            )

        # Reset UI
        self.session_manager.clear_user()
        self.dashboard.set_identity("", False)
        self.dashboard.set_auto_status("AUTO-SCAN: WAITING FOR FACE...", "normal")
        self.dashboard.lock_dashboard()
        self.wash_detector.reset_state()

    def apply_camera_source(self, source):
        """Apply new camera source.
        
        Args:
            source: int (webcam index) or str (video file path)
        """
        self.timer.stop()
        if self.cap:
            self.cap.release()

        self.dashboard.video_label.setText("STARTING VIDEO SOURCE...")
        self.dashboard.progress_bar.setValue(0)

        self.current_source = source
        if isinstance(source, int):
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source)

        self.wash_detector.reset_state()
        self.sink_y_start = None
        self.tabs.setCurrentIndex(0)

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        self.timer.start(delay)

    def enter_roi_mode(self):
        """Enter ROI drawing mode."""
        ret, frame = self.cap.read()
        if not ret:
            return

        self.timer.stop()
        accepted, roi = create_roi_dialog(self, frame, self.scrub_roi)
        
        if accepted and roi:
            self.scrub_roi = roi
            self.settings.setValue("scrub_roi", json.dumps(self.scrub_roi))

        self.timer.start()

    def trigger_calibration(self):
        """Reset sink calibration."""
        self.sink_y_start = None
        self.scrub_roi = None
        self.settings.setValue("scrub_roi", None)
        self.tabs.setCurrentIndex(0)

    def update_toggles(self):
        """Update detection toggles from settings."""
        toggles = self.settings_tab.get_detection_toggles()
        self.check_mask = toggles['mask']
        self.check_hat = toggles['hat']
        self.check_wash = toggles['wash']

        if not self.check_wash:
            self.wash_detector.reset_state()

    def update_frame(self):
        """Update frame processing and UI."""
        ret, frame = self.cap.read()

        if not ret:
            if isinstance(self.current_source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.dashboard.video_label.setText("VIDEO SIGNAL LOST")
            return

        if isinstance(self.current_source, str):
            self.cap.grab()

        frame_h, frame_w = frame.shape[:2]
        clean_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- PRESENCE CHECK ---
        if self.ai_models.detect_face(clean_rgb):
            self.session_manager.update_presence()
            if not self.session_manager.is_authenticated() and not self.session_manager.is_authenticating:
                self.start_authentication()

        # --- CALIBRATION ---
        if self.scrub_roi:
            self.sink_y_start = int(self.scrub_roi[1] * frame_h)
            frame = SinkCalibration.draw_manual_roi(frame, self.scrub_roi)
        else:
            if self.sink_y_start is None:
                detected_y = SinkCalibration.auto_detect_sink_line(frame)
                self.sink_y_start = SinkCalibration.calculate_sink_y_start(detected_y, frame_h)
            frame = SinkCalibration.draw_sink_zone(frame, self.sink_y_start, manual=bool(self.scrub_roi))

        # --- PPE DETECTION ---
        has_mask, has_hat = False, False
        if self.check_mask or self.check_hat:
            frame, has_mask, has_hat = self.ai_models.detect_ppe(frame)
            if has_mask:
                self.session_manager.update_presence()
                self.wash_detector.last_mask_seen_time = time.time()
            if has_hat:
                self.session_manager.update_presence()
                self.wash_detector.last_hat_seen_time = time.time()

        # Update PPE display
        if not self.check_mask and not self.check_hat:
            self.dashboard.set_ppe_status("", "", disabled=True)
        else:
            m_text = "VERIFIED ✅" if has_mask else "MISSING ❌"
            h_text = "VERIFIED ✅" if has_hat else "MISSING ❌"
            self.dashboard.set_ppe_status(m_text, h_text)

        # --- HAND WASHING DETECTION ---
        hand_results = self.ai_models.detect_hands(clean_rgb)
        if self.check_wash and hand_results['detected']:
            frame = self.ai_models.draw_hand_landmarks(frame, hand_results['hand_results'])
            wash_info = self.wash_detector.detect_washing(
                hand_results, frame_w, frame_h, self.sink_y_start, self.ai_models
            )
            self.wash_detector.update_wash_time(wash_info['actively_washing'])
            frame = self.wash_detector.draw_bubble_zone(frame)

            status_text = self.wash_detector.get_wash_status(wash_info['hands_count'])
            self.dashboard.set_wash_status(status_text, int(self.wash_detector.current_wash_time))

        # Update wash display when check is disabled
        if not self.check_wash:
            self.dashboard.set_wash_status("WASH TIMER:⏭️ DISABLED", config.MAX_WASH_TIME)

        # --- MASTER STATUS ---
        if self.session_manager.is_authenticated():
            # Check timeout
            if self.session_manager.check_presence_timeout():
                self.logout_user()
            else:
                # Evaluate ready state
                master_ready = True
                if self.check_mask and not has_mask:
                    master_ready = False
                if self.check_hat and not has_hat:
                    master_ready = False
                if self.check_wash and self.wash_detector.current_wash_time < config.MIN_WASH_TIME:
                    master_ready = False

                self.dashboard.set_master_status(master_ready)
        else:
            self.dashboard.lock_dashboard()

        # --- UPDATE UI ---
        self.dashboard.update_video(frame)
        self.registration.set_frame(frame)

    def closeEvent(self, event):
        """Handle window close event."""
        self.cap.release()
        self.ai_models.cleanup()
        event.accept()


def main():
    """Entry point for the application."""
    print("[DEBUG 1] Setting up environment...")
    print("[DEBUG 2] Starting PyQt Application...")
    
    app = QApplication(sys.argv)
    
    print("[DEBUG 3] Creating main window...")
    window = ScrubSinkKiosk()
    
    print("[DEBUG 4] Showing window...")
    window.show()
    
    print("[DEBUG 5] Entering event loop...")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
