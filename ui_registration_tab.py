"""Registration Tab UI Module - Handles staff registration interface."""

import os
import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QRadioButton, QGroupBox, QMessageBox,QComboBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
import config
from data_logger import DataLogger
from ai_models import reset_face_cache ,add_single_face_to_cache
class RegistrationTab(QWidget):
    """Registration tab for adding new staff members."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.last_clean_frame = None
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_val = 0
        self.data_logger = DataLogger()
        self.build_ui()

    def build_ui(self):
        """Build the registration tab UI."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(40)

        # --- LEFT SIDE: THE FORM ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        title = QLabel("NEW STAFF REGISTRATION")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        left_layout.addWidget(title)

        cam_group = QGroupBox("SELECT CAMERA SOURCE")
        cam_layout = QVBoxLayout(cam_group)

        self.cam_selector = QComboBox()
        self.cam_selector.addItems(["SINK 1","SINK 2","SINK 3","SINK 4","SINK 5"])
        self.cam_selector.setStyleSheet("padding: 10px; font-size: 12xp; font-weight: bold; border: 1xp solid #000000;")

        cam_layout.addWidget(self.cam_selector)
        left_layout.addWidget(cam_group)

        form_group = QGroupBox("STAFF DETAILS")
        form_layout = QVBoxLayout(form_group)

        self.reg_fname = QLineEdit()
        self.reg_fname.setPlaceholderText("FIRST NAME")
        self.reg_lname = QLineEdit()
        self.reg_lname.setPlaceholderText("LAST NAME")
        self.reg_role = QLineEdit()
        self.reg_role.setPlaceholderText("ROLE (e.g., Surgeon, Nurse)")

        for field in [self.reg_fname, self.reg_lname, self.reg_role]:
            field.setStyleSheet("padding: 10px; font-size: 14px; border: 1px solid #000000;")
            form_layout.addWidget(field)

        # Gender selection
        gender_widget = QWidget()
        gender_layout = QHBoxLayout(gender_widget)
        gender_layout.setContentsMargins(0, 5, 0, 5)

        gender_label = QLabel("GENDER:")
        gender_label.setFont(QFont("Arial", 12, QFont.Bold))

        self.radio_male = QRadioButton("MALE")
        self.radio_female = QRadioButton("FEMALE")
        self.radio_male.setChecked(True)
        self.radio_male.setFont(QFont("Arial", 11))
        self.radio_female.setFont(QFont("Arial", 11))

        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(self.radio_male)
        gender_layout.addWidget(self.radio_female)
        gender_layout.addStretch()

        form_layout.addWidget(gender_widget)
        left_layout.addWidget(form_group)

        self.capture_btn = QPushButton("LOOK AT CAMERA & START TIMER")
        self.capture_btn.setStyleSheet("background: #000000; color: #ffffff; padding: 15px; font-size: 14px;")
        self.capture_btn.clicked.connect(self.start_countdown)
        left_layout.addWidget(self.capture_btn)
        left_layout.addStretch()

        main_layout.addWidget(left_panel, stretch=1)

        # --- RIGHT SIDE: THE LIVE CAMERA ---
        self.reg_video_label = QLabel("CAMERA STANDBY")
        self.reg_video_label.setAlignment(Qt.AlignCenter)
        self.reg_video_label.setStyleSheet("border: 2px solid #000000; background: #000000; color: #ffffff;")
        self.reg_video_label.setMinimumSize(480, 360)
        main_layout.addWidget(self.reg_video_label, stretch=2)

    def set_frame(self, sink_name, frame):
        """Update the current camera frame only if it matches the selected dropdown sink."""
        clean_name = sink_name.replace("_", " ")
        if clean_name != self.cam_selector.currentText():
            return  

        self.last_clean_frame = frame.copy()
       
        if self.countdown_val > 0 and not self.capture_btn.isEnabled():
            self.display_frame_with_countdown(frame)
        else:
            self.display_frame(frame)

    def display_frame(self, frame):

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.reg_video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.reg_video_label.width(), self.reg_video_label.height(), Qt.KeepAspectRatio))

    def display_frame_with_countdown(self, frame):

        reg_frame = frame.copy()
        h, w = reg_frame.shape[:2]
        
        # Draw countdown text
        text = str(self.countdown_val)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 7, 15)[0]
        text_x = int((w - text_size[0]) / 2)
        text_y = int((h + text_size[1]) / 2)
        cv2.putText(reg_frame, text, (text_x, text_y), font, 7, (0, 0, 255), 15, cv2.LINE_AA)
        
        self.display_frame(reg_frame)

    def start_countdown(self):
        """Start the registration countdown."""
        fname = self.reg_fname.text().strip()
        lname = self.reg_lname.text().strip()
        
        if not fname or not lname:
            QMessageBox.warning(self, "ERROR", "First and Last name are required.")
            return
        
        if self.last_clean_frame is None:
            QMessageBox.warning(self, "ERROR", "Camera not ready.")
            return

        self.capture_btn.setEnabled(False)
        self.countdown_val = 5
        self.capture_btn.setText(f"TAKING PICTURE IN {self.countdown_val}...")
        self.countdown_timer.start(1000)

    def update_countdown(self):
        """Update countdown display."""
        self.countdown_val -= 1
       
        if self.countdown_val > 0:
            self.capture_btn.setText(f"TAKING PICTURE IN {self.countdown_val}...")
        else:
            self.countdown_timer.stop()
            self.capture_btn.setText("PROCESSING...")
            # This now correctly calls our updated multi-angle saving method below!
            self.register_new_user()

    def register_new_user(self):
        """Captures the current clean frame, saves role metadata, and appends reference photos."""
        import json
        fname = self.reg_fname.text().strip().upper()
        lname = self.reg_lname.text().strip().upper()
        role = self.reg_role.text().strip().title() or "N/A"

        if not fname or not lname or self.last_clean_frame is None:
            QMessageBox.warning(self, "ERROR", "Missing name or camera frame!")
            self.capture_btn.setText("LOOK AT CAMERA & START TIMER")
            self.capture_btn.setEnabled(True)
            return

        # 1. Format folder path
        full_name = f"{fname} {lname}"
        clean_folder_name = f"{fname}_{lname}".replace(" ", "_")
        user_dir = os.path.join(config.REG_PATH, clean_folder_name)
        os.makedirs(user_dir, exist_ok=True)

        # 2. Save/Update Role in user_info.json
        info_path = os.path.join(user_dir, "user_info.json")
        with open(info_path, "w") as f:
            json.dump({"role": role}, f, indent=4)

        # 3. Count existing photos for sequential naming
        existing_photos = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        next_angle_num = len(existing_photos) + 1

        # 4. Save image
        file_name = f"angle_{next_angle_num}.jpg"
        save_path = os.path.join(user_dir, file_name)
        cv2.imwrite(save_path, self.last_clean_frame)

        # 5. Inject into live RAM cache
        add_single_face_to_cache(clean_folder_name, save_path)

        # 6. Reset UI fields & show confirmation
        self.reg_fname.clear()
        self.reg_lname.clear()
        self.reg_role.clear()
        self.capture_btn.setText("LOOK AT CAMERA & START TIMER")
        self.capture_btn.setEnabled(True)

        QMessageBox.information(
            self,
            "SUCCESS",
            f"Saved {file_name} for {full_name}\nRole: {role}\nTotal reference angles stored: {next_angle_num}"
        )
        print(f"[REGISTRATION] Saved angle #{next_angle_num} for {full_name} ({role}) to {save_path}")


