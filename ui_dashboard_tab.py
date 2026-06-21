"""Dashboard Tab UI Module - Main system dashboard with protocol status."""

import cv2
from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
import config

class DashboardTab(QWidget):
    """Main dashboard showing protocol status and hand washing timer."""
    
    # Signals for authentication
    auth_requested = pyqtSignal()
    roi_requested = pyqtSignal()

    def __init__(self,sink_name = "CAMERA 1", parent=None):
        super().__init__(parent)
        self.sink_name = sink_name
        self.parent_window = parent
        self.last_frame =None
        self.build_ui()
    

    def build_ui(self):
        """Build the dashboard UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)

        # --- LEFT SIDE: VIDEO FEED ---
        self.video_label = QLabel("INITIALIZING CAMERA...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #000000; background: #000000; color: #ffffff;")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, stretch=2)

        # --- RIGHT SIDE: STATUS PANEL ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(right_panel, stretch=1)

        # Title
        title = QLabel(f"{self.sink_name} - PROTOCOL STATUS")
        title.setFont(QFont("Arial" , 16, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(title)

        # Divider line
        line = QWidget()
        line.setFixedHeight(2)
        line.setStyleSheet("background-color: #000000;")
        right_layout.addWidget(line)
        right_layout.addSpacing(20)

        # Identity label
        self.identity_label = QLabel("USER: NOT AUTHENTICATED")
        self.identity_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.identity_label.setStyleSheet("color: #ff0000;")
        right_layout.addWidget(self.identity_label)

        # Auto-scan status
        self.auto_status_label = QLabel("AUTO-SCAN: WAITING FOR FACE...")
        self.auto_status_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.auto_status_label.setStyleSheet("padding: 10px; background: #eeeeee; border: 1px solid #000000;")
        self.auto_status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.auto_status_label)
        right_layout.addSpacing(20)

        # Authentication button
        
        self.btn_roi = QPushButton("SET MANUAL SINL ZONE")
        self.btn_roi.setStyleSheet ("padding: 10px; font-weight: bold; background: #e9ecef")
        self.btn_roi.clicked.connect(self.roi_requested.emit)
        right_layout.addWidget(self.btn_roi)

        right_layout.addSpacing(20)


        # PPE label
        self.ppe_label = QLabel("[ PPE VERIFICATION PENDING ]")
        self.ppe_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.ppe_label.setStyleSheet("padding: 15px; border: 2px solid #000000; background: #ffffff;")
        right_layout.addWidget(self.ppe_label)
        right_layout.addSpacing(10)

        # Wash label
        self.wash_label = QLabel("WASH TIMER: STANDBY")
        self.wash_label.setFont(QFont("Arial", 11))

        self.wash_label.setMinimumHeight(50)
        self.wash_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        right_layout.addWidget(self.wash_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(config.MAX_WASH_TIME)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True) 
        self.progress_bar.setFormat("%v SECONDS") 
        self.progress_bar.setAlignment(Qt.AlignCenter) 
        self.progress_bar.setStyleSheet(""" 
                QProgressBar { border: 2px solid #000000;
                background: #ffffff; height: 30px; text-align: center;
                font-weight: bold; color: #000000; font-size: 14px; } 
                QProgressBar::chunk { background-color: #1b4332; }""")
        right_layout.addWidget(self.progress_bar)

        right_layout.addStretch()

        # Master status
        self.master_status = QLabel("STATUS: ACTION REQUIRED")
        self.master_status.setFont(QFont("Arial", 14, QFont.Bold))
        self.master_status.setAlignment(Qt.AlignCenter)
        self.master_status.setStyleSheet(
            "color: #000000; border: 2px solid #000000; padding: 20px; background: #ffffff;")
        right_layout.addWidget(self.master_status)

 

    def on_auth_clicked(self):
        """Handle authentication button click."""
        self.auth_requested.emit()

    def update_video(self, frame):
        """Update video display."""
        self.last_frame = frame.copy()
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def set_identity(self, user_name, authenticated=True):
        """Set identity display."""
        if authenticated:
            self.identity_label.setText(f"USER: {user_name.upper()} ✅")
            self.identity_label.setStyleSheet("color: #1b4332;")
        else:
            self.identity_label.setText("USER: NOT AUTHENTICATED")
            self.identity_label.setStyleSheet("color: #ff0000;")

    def set_auto_status(self, status_text, status_type="normal"):
        """Set auto-scan status display."""
        colors = {
            "normal": "padding: 10px; background: #eeeeee; color: #000000; border: 1px solid #000000;",
            "warning": "padding: 10px; background: #ffff00; color: #000000; border: 1px solid #000000;",
            "success": "padding: 10px; background: #aaffaa; color: #000000; border: 1px solid #000000;",
            "error": "padding: 10px; background: #ffaaaa; color: #000000; border: 1px solid #000000;"
        }
        self.auto_status_label.setText(status_text)
        self.auto_status_label.setStyleSheet(colors.get(status_type, colors["normal"]))

    def set_ppe_status(self, mask_status, hat_status, disabled=False):
        """Set PPE verification display."""
        if disabled:
            self.ppe_label.setText("[ PPE CHECK:⏭️ DISABLED ]")
        else:
            self.ppe_label.setText(f"[ MASK: {mask_status} ]    [ HAT: {hat_status} ]")

    def set_wash_status(self, status_text, progress_value):
        """Set hand washing status."""
        self.wash_label.setText(status_text)
        self.progress_bar.setValue(progress_value)

    def set_master_status(self, ready=False):
        """Set master status display."""
        if ready:
            self.master_status.setText("STATUS: PROCEED TO Operating Room ✅")
            self.master_status.setStyleSheet(
                "color: #ffffff; background-color: #000000; border: 2px solid #000000; padding: 20px;")
        else:
            self.master_status.setText("STATUS: ACTION REQUIRED ⚠️")
            self.master_status.setStyleSheet(
                "color: #000000; background-color: #ffffff; border: 2px solid #000000; padding: 20px;")

    def lock_dashboard(self):
        """Lock dashboard (unauthenticated state)."""
        self.master_status.setText("SYSTEM LOCKED: PLEASE AUTHENTICATE")
        self.master_status.setStyleSheet("color: #ffffff; background-color: #000000; border: 2px solid #000000; padding: 20px;")
        self.wash_label.setText("WASH TIMER: LOCKED")
        self.ppe_label.setText("[ PPE CHECK: LOCKED ]")

    def unlock_dashboard(self):
        """Unlock dashboard (authenticated state)."""
        self.wash_label.setText("WASH TIMER: STANDBY")
        self.ppe_label.setText("[ PPE VERIFICATION PENDING ]")

    def update_from_worker(self, data):
        """Update all UI elements from the worker thread data."""
        self.set_identity(data['user'], data['is_auth'])
        self.set_auto_status(data['auth_msg'], data['auth_color'])

        if not data['is_auth']:
            self.lock_dashboard()
            return
           
        self.unlock_dashboard()

        # Update PPE Visuals
        if not data['check_mask'] and not data['check_hat']:
            self.set_ppe_status("", "", disabled=True)
        else:
            m_text = "VERIFIED ✅" if data['mask'] else "MISSING ❌"
            h_text = "VERIFIED ✅" if data['hat'] else "MISSING ❌"
            self.set_ppe_status(m_text, h_text)

        # Update Wash Visuals
        if not data['check_wash']:
            self.set_wash_status("WASH TIMER:⏭️ DISABLED", config.MAX_WASH_TIME)
        else:
            self.set_wash_status(data['wash_status'], int(data['wash_time']))

        # Update Final Ready State
        self.set_master_status(data['master_ready'])