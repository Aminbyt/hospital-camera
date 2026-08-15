"""Main Application - Hospital AI Smart Scrub Sink Kiosk."""
import sys
import os

# --- 1. SET CRITICAL ENVIRONMENT VARIABLES BEFORE ANY IMPORTS ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'  # Prevents CPU thread starvation!

# --- 2. REGISTER DLL PATHS FOR PYINSTALLER ---
if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
    torch_lib_dir = os.path.join(base_dir, 'torch', 'lib')
    mediapipe_dir = os.path.join(base_dir, 'mediapipe')
    onnx_dir = os.path.join(base_dir, 'onnxruntime', 'capi')
   
    # A. Register DLL directories with Windows
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(base_dir)
        if os.path.exists(torch_lib_dir):
            os.add_dll_directory(torch_lib_dir)
        if os.path.exists(mediapipe_dir):
            os.add_dll_directory(mediapipe_dir)
        if os.path.exists(onnx_dir):
            os.add_dll_directory(onnx_dir)
           
    # B. Force-inject into Windows PATH for legacy C++ sub-dependencies
    os.environ['PATH'] = f"{base_dir};{torch_lib_dir};{mediapipe_dir};{onnx_dir};" + os.environ.get('PATH', '')
   
    # C. UPGRADED DLL LOADER: Pre-loads both PyTorch AND ONNX Runtime DLLs!
    import ctypes
    import glob
    target_dlls = glob.glob(os.path.join(torch_lib_dir, "*.dll")) + glob.glob(os.path.join(onnx_dir, "*.dll"))
    loaded_dlls = set()
    for pass_num in range(1, 4):
        for dll_path in target_dlls:
            if dll_path not in loaded_dlls:
                try:
                    ctypes.CDLL(dll_path)
                    loaded_dlls.add(dll_path)
                except Exception:
                    pass
    print(f"[BOOT SUCCESS] Pre-loaded {len(loaded_dlls)} PyTorch & ONNX Runtime DLLs into memory!")
# ---------------------------------------------

# --- 3. CRITICAL IMPORT ORDER: MEDIAPIPE, ONNXRUNTIME & INSIGHTFACE FIRST ---
try:
    import mediapipe as mp
    print("[BOOT SUCCESS] MediaPipe C++ framework bindings initialized cleanly!")
except Exception as e:
    print(f"[BOOT WARNING] MediaPipe early import note: {e}")

try:
    import onnxruntime
    import insightface
    print("[BOOT SUCCESS] ONNX Runtime & InsightFace C++ engines initialized cleanly!")
except Exception as e:
    print(f"[BOOT WARNING] ONNX/InsightFace early import note: {e}")

import torch
# ----------------------------------------------------------------------------

# --- 4. INITIALIZE MASTER LOGGER ---
try:
    from logger_setup import setup_system_logger
    logger = setup_system_logger()
except ImportError:
    pass
# -----------------------------------

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel
)
from PyQt5.QtCore import Qt

import config
from ui_home_tab import HomeSummaryTab
from ui_dashboard_tab import DashboardTab
from ui_registration_tab import RegistrationTab
from ui_settings_tab import SettingsTab
from camrea_worker import CameraWorker
from sink_calibration import SinkCalibration, create_roi_dialog
from send_daily_reports import DailyReportThread , MonthlyReportThread
from heartbeat_worker import HeartbeatThread

class ScrubSinkKiosk(QMainWindow):
    """Master Control Center."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hospital AI - Master Control Center")
        self.setGeometry(50, 50, 1400, 800)
        self.setStyleSheet(config.STYLESHEET)

        # --- MASTER LAYOUT ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- LEFT SIDEBAR ---
        sidebar_widget = QWidget()
        sidebar_widget.setStyleSheet("background-color: #1b4332; color: white;")
        sidebar_widget.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(10, 30, 10, 30)
       
        # Sidebar Logo/Title
        title_lbl = QLabel("SMART SCRUB\nCONTROL CENTER")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; text-align: center;")
        title_lbl.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_lbl)
        sidebar_layout.addSpacing(30)

        # Create Navigation Buttons
        self.buttons = {}
        nav_items = ["HOME", "SINK 1", "SINK 2", "SINK 3", "SINK 4", "SINK 5", "REGISTRATION", "SETTINGS"]
       
        for item in nav_items:
            btn = QPushButton(item)
            btn.setStyleSheet("""
                QPushButton { padding: 15px; text-align: left; font-weight: bold; font-size: 14px;
                              border: none; background: transparent; color: white; }
                QPushButton:hover { background-color: #2d6a4f; border-radius: 5px; }
            """)
            self.buttons[item] = btn
            sidebar_layout.addWidget(btn)
       
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar_widget)

        # --- RIGHT CONTENT STACK (The "Pages") ---
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, stretch=1)

        # 1. Initialize Pages
        self.page_home = HomeSummaryTab()
        self.page_cam1 = DashboardTab(sink_name="SINK 1")  # We will pass specific sink names to these later
        self.page_cam2 = DashboardTab(sink_name="SINK 2")
        self.page_cam3 = DashboardTab(sink_name="SINK 3")
        self.page_cam4 = DashboardTab(sink_name="SINK 4")
        self.page_cam5 = DashboardTab(sink_name="SINK 5")
        self.page_reg = RegistrationTab()
        self.page_set = SettingsTab()

        # 2. Add Pages to Stack
        self.content_stack.addWidget(self.page_home) # Index 0
        self.content_stack.addWidget(self.page_cam1) # Index 1
        self.content_stack.addWidget(self.page_cam2) # Index 2
        self.content_stack.addWidget(self.page_cam3) # Index 3
        self.content_stack.addWidget(self.page_cam4) # Index 4
        self.content_stack.addWidget(self.page_cam5) # Index 5
        self.content_stack.addWidget(self.page_reg)  # Index 6
        self.content_stack.addWidget(self.page_set)  # Index 7

        # 3. Connect Sidebar Buttons to change pages
        self.buttons["HOME"].clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.buttons["SINK 1"].clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        self.buttons["SINK 2"].clicked.connect(lambda: self.content_stack.setCurrentIndex(2))
        self.buttons["SINK 3"].clicked.connect(lambda: self.content_stack.setCurrentIndex(3))
        self.buttons["SINK 4"].clicked.connect(lambda: self.content_stack.setCurrentIndex(4))
        self.buttons["SINK 5"].clicked.connect(lambda: self.content_stack.setCurrentIndex(5))
        self.buttons["REGISTRATION"].clicked.connect(lambda: self.content_stack.setCurrentIndex(6))
        self.buttons["SETTINGS"].clicked.connect(lambda: self.content_stack.setCurrentIndex(7))

                # --- INITIALIZE THE 5 BACKGROUND AI THREADS ---
        self.workers = {}
       
        # Pull camera mappings from config (e.g., SINK_1: 0, SINK_2: 1)
        for sink_id, cam_index in config.SINK_CAMERAS.items():
            worker = CameraWorker(sink_name=sink_id, camera_index=cam_index)
           

            worker.raw_frame_ready.connect(self.page_reg.set_frame)
            # 2. Route the video frame AND UI data to the correct Dashboard Tabs
            if sink_id == "SINK_1":
                worker.frame_ready.connect(self.page_cam1.update_video)
                worker.dashboard_data.connect(self.page_cam1.update_from_worker)
                self.page_cam1.roi_requested.connect(lambda w=worker, p=self.page_cam1: self.open_roi_dialog(w, p))
               
            elif sink_id == "SINK_2":
                worker.frame_ready.connect(self.page_cam2.update_video)
                worker.dashboard_data.connect(self.page_cam2.update_from_worker)
                self.page_cam2.roi_requested.connect(lambda w=worker, p=self.page_cam2: self.open_roi_dialog(w, p))
               
            elif sink_id == "SINK_3":
                worker.frame_ready.connect(self.page_cam3.update_video)
                worker.dashboard_data.connect(self.page_cam3.update_from_worker)
                self.page_cam3.roi_requested.connect(lambda w=worker, p=self.page_cam3: self.open_roi_dialog(w, p))
               
            elif sink_id == "SINK_4":
                worker.frame_ready.connect(self.page_cam4.update_video)
                worker.dashboard_data.connect(self.page_cam4.update_from_worker)
                self.page_cam4.roi_requested.connect(lambda w=worker, p=self.page_cam4: self.open_roi_dialog(w, p))
               
            elif sink_id == "SINK_5":
                worker.frame_ready.connect(self.page_cam5.update_video)
                worker.dashboard_data.connect(self.page_cam5.update_from_worker)
                self.page_cam5.roi_requested.connect(lambda w=worker, p=self.page_cam5: self.open_roi_dialog(w, p))

            # 3. Route text data to the Home Overview Tab
            worker.data_ready.connect(self.page_home.update_sink_data)
           
            # Save worker to memory and start it!
            self.workers[sink_id] = worker
            worker.start()

        # --- CONNECT SETTINGS BUTTONS TO WORKERS ---
        # When you change settings, we loop through all 5 workers and update them!
        self.page_set.toggles_changed.connect(self.master_update_toggles)
        self.page_set.calibration_requested.connect(self.master_trigger_calibration)

        self.master_update_toggles()

        self.report_thread =DailyReportThread()
        self.report_thread.start()

        self.monthly_thread = MonthlyReportThread()
        self.monthly_thread.start()

        self.heartbeat_thread = HeartbeatThread()
        self.heartbeat_thread.start()

    def open_roi_dialog(self, worker, page_widget):
        """Pauses, opens the drawing window, and saves the new red line to the specific camera."""
        if not hasattr(page_widget, 'last_frame') or page_widget.last_frame is None:
            return
           
        # Open the drawing popup
        accepted, new_roi = create_roi_dialog(self, page_widget.last_frame, worker.scrub_roi)
       
        # If the user clicked "SAVE", send it to the background AI thread!
        if accepted:
            worker.set_manual_roi(new_roi)
            
    def master_update_toggles(self):
        toggles = self.page_set.get_detection_toggles()
        for worker in self.workers.values():
            worker.update_toggles(toggles['mask'], toggles['hat'], toggles['wash'],toggles.get('record',True))

    def master_trigger_calibration(self):
        for worker in self.workers.values():
            worker.trigger_calibration()

    def closeEvent(self, event):
        """Safely shut down all 5 cameras when closing the app."""
        if hasattr(self , "report_thread"):
            self.report_thread.stop()

        if hasattr(self , "heartbeat_thread"):
            self.heartbeat_thread.stop()
            
        if hasattr(self, "monthly_thread"):
            self.monthly_thread.stop()
            
        for worker in self.workers.values():
            worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ScrubSinkKiosk()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()