"""Main Application (Dumb Client) - Hospital AI Smart Scrub Sink Kiosk."""
import sys
import json
import cv2
import numpy as np
import zmq

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QStackedWidget, QLabel)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

import config
from ui_home_tab import HomeSummaryTab
from ui_dashboard_tab import DashboardTab
from ui_registration_tab import RegistrationTab
from ui_settings_tab import SettingsTab

class StateWrapper:
    """Utility to convert the incoming JSON dictionary back into an object and dictionary format for the UI."""
    def __init__(self, dictionary):
        self._dictionary = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        """Allows bracket lookup (e.g. state_obj['user']) for legacy UI tabs."""
        return self._dictionary[key]

    def get(self, key, default=None):
        """Allows dictionary .get() safety fallback."""
        return self._dictionary.get(key, default)

class ZmqSubscriberThread(QThread):
    """Listens for ZMQ broadcasts from the headless engine."""
    state_received = pyqtSignal(str, object)
    frame_received = pyqtSignal(str, np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Listen to all sink topics

    def run(self):
        while self.running:
            try:
                # NOBLOCK prevents the UI from freezing if the engine stops sending
                topic, state_json, frame_bytes = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                
                sink_id = topic.decode('utf-8')
                state_dict = json.loads(state_json.decode('utf-8'))
                state_obj = StateWrapper(state_dict)
                
                self.state_received.emit(sink_id, state_obj)

# --- FIX 3B: Reconstruct raw bytes instantly ---
                if frame_bytes and state_dict.get("frame_shape"):
                    np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = np_arr.reshape(state_dict["frame_shape"])
                    self.frame_received.emit(sink_id, frame)
                        
            except zmq.Again:
                self.msleep(10)  # Sleep briefly if no message is waiting
            except Exception as e:
                print(f"[UI ERROR] ZMQ Rx Error: {e}")

    def stop(self):
        self.running = False
        self.wait(2000)
        self.socket.close()
        self.context.term()


class ScrubSinkKiosk(QMainWindow):
    """Master Control Center - UI Only."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hospital AI - Master Control Center (Client)")
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
       
        title_lbl = QLabel("SMART SCRUB\nCONTROL CENTER")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: bold; text-align: center;")
        title_lbl.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_lbl)
        sidebar_layout.addSpacing(30)

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

        self.page_home = HomeSummaryTab()
        self.page_cam1 = DashboardTab(sink_name="SINK_1")
        self.page_cam2 = DashboardTab(sink_name="SINK_2")
        self.page_cam3 = DashboardTab(sink_name="SINK_3")
        self.page_cam4 = DashboardTab(sink_name="SINK_4")
        self.page_cam5 = DashboardTab(sink_name="SINK_5")
        self.page_reg = RegistrationTab()
        self.page_set = SettingsTab()

        self.content_stack.addWidget(self.page_home) 
        self.content_stack.addWidget(self.page_cam1) 
        self.content_stack.addWidget(self.page_cam2) 
        self.content_stack.addWidget(self.page_cam3) 
        self.content_stack.addWidget(self.page_cam4) 
        self.content_stack.addWidget(self.page_cam5) 
        self.content_stack.addWidget(self.page_reg)  
        self.content_stack.addWidget(self.page_set)  

        self.buttons["HOME"].clicked.connect(lambda: self.content_stack.setCurrentIndex(0))
        self.buttons["SINK 1"].clicked.connect(lambda: self.content_stack.setCurrentIndex(1))
        self.buttons["SINK 2"].clicked.connect(lambda: self.content_stack.setCurrentIndex(2))
        self.buttons["SINK 3"].clicked.connect(lambda: self.content_stack.setCurrentIndex(3))
        self.buttons["SINK 4"].clicked.connect(lambda: self.content_stack.setCurrentIndex(4))
        self.buttons["SINK 5"].clicked.connect(lambda: self.content_stack.setCurrentIndex(5))
        self.buttons["REGISTRATION"].clicked.connect(lambda: self.content_stack.setCurrentIndex(6))
        self.buttons["SETTINGS"].clicked.connect(lambda: self.content_stack.setCurrentIndex(7))

        # --- START ZMQ SUBSCRIBER ---
        self.zmq_listener = ZmqSubscriberThread()
        self.zmq_listener.state_received.connect(self.route_state_update)
        self.zmq_listener.frame_received.connect(self.route_frame_update)
        self.zmq_listener.start()

    def route_state_update(self, sink_id, state_obj):
        """Routes incoming IPC state to the correct UI tabs."""
        # Update Home Overview
        self.page_home.update_sink_data(sink_id, state_obj)
        
        # Update Specific Dashboard
        dashboard_map = {
            "SINK_1": self.page_cam1,
            "SINK_2": self.page_cam2,
            "SINK_3": self.page_cam3,
            "SINK_4": self.page_cam4,
            "SINK_5": self.page_cam5,
        }
        if sink_id in dashboard_map:
            dashboard_map[sink_id].update_from_worker(state_obj)

    def route_frame_update(self, sink_id, frame):
        """Routes incoming video frames to the currently visible tab."""
        current_idx = self.content_stack.currentIndex()
        
        # Only process the frame if we are actually looking at that specific dashboard
        dashboard_indices = {"SINK_1": 1, "SINK_2": 2, "SINK_3": 3, "SINK_4": 4, "SINK_5": 5}
        if sink_id in dashboard_indices and current_idx == dashboard_indices[sink_id]:
            dashboard = getattr(self, f"page_cam{sink_id.split('_')[1]}")
            dashboard.update_video(frame)
            
        # Route to registration tab if active
        if current_idx == 6:
            self.page_reg.set_frame(sink_id, frame)

    def closeEvent(self, event):
        """Clean UI teardown."""
        self.zmq_listener.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = ScrubSinkKiosk()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()