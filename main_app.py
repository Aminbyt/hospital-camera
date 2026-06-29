"""Main Application - Hospital AI Smart Scrub Sink Kiosk."""
import sys
import os
if getattr(sys, 'frozen', False):
    os.add_dll_directory(sys._MEIPASS)

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
from sink_calibration import SinkCalibration , create_roi_dialog

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
           
            # 1. Route the video frame AND UI data to the correct Dashboard Tabs
            if sink_id == "SINK_1":
                worker.frame_ready.connect(self.page_cam1.update_video)
                worker.dashboard_data.connect(self.page_cam1.update_from_worker)
                worker.raw_frame_ready.connect(self.page_reg.set_frame)
                # Connect the drawing button!
                self.page_cam1.roi_requested.connect(lambda w=worker, p=self.page_cam1: self.open_roi_dialog(w, p))
               
            elif sink_id == "SINK_2":
                worker.frame_ready.connect(self.page_cam2.update_video)
                worker.dashboard_data.connect(self.page_cam2.update_from_worker)
                # Connect the drawing button!
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

            # 2. Route the text data to the Home Overview Tab
            worker.data_ready.connect(self.page_home.update_sink_data)
           
            # Save worker to memory and start it!
            self.workers[sink_id] = worker
            worker.start()

        # --- CONNECT SETTINGS BUTTONS TO WORKERS ---
        # When you change settings, we loop through all 5 workers and update them!
        self.page_set.toggles_changed.connect(self.master_update_toggles)
        self.page_set.calibration_requested.connect(self.master_trigger_calibration)

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
            worker.update_toggles(toggles['mask'], toggles['hat'], toggles['wash'])

    def master_trigger_calibration(self):
        for worker in self.workers.values():
            worker.trigger_calibration()

    def closeEvent(self, event):
        """Safely shut down all 5 cameras when closing the app."""
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