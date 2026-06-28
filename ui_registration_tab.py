"""Registration Tab UI Module - Handles staff registration interface."""

import os
import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QRadioButton, QGroupBox, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
import config
from data_logger import DataLogger
from ai_models import reset_face_cache


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

    def set_frame(self, frame):
        """Update the current camera frame.
        
        Args:
            frame: Current camera frame
        """
        self.last_clean_frame = frame.copy()
        
        # Display frame if countdown is active
        if self.countdown_val > 0 and not self.capture_btn.isEnabled():
            self.display_frame_with_countdown(frame)
        else:
            self.display_frame(frame)

    def display_frame(self, frame):
        """Display frame in the video label.
        
        Args:
            frame: Frame to display
        """
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.reg_video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.reg_video_label.width(), self.reg_video_label.height(), Qt.KeepAspectRatio))

    def display_frame_with_countdown(self, frame):
        """Display frame with countdown overlay.
        
        Args:
            frame: Frame to display
        """
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
            self.register_new_user()

    def register_new_user(self):
        """Register new user with captured face."""
        fname = self.reg_fname.text().strip()
        lname = self.reg_lname.text().strip()
        gender = "MALE" if self.radio_male.isChecked() else "FEMALE"
        full_name = f"{fname}_{lname}"

        # Save face image
        person_folder = os.path.join(config.REG_PATH, full_name)
        os.makedirs(person_folder, exist_ok=True)

        img_path = os.path.join(person_folder, f"{full_name}.jpg")
        cv2.imwrite(img_path, self.last_clean_frame)

        # Clear cache
        reset_face_cache()

        QMessageBox.information(self, "SUCCESS", 
                              f"Profile for {full_name} ({gender}) saved successfully!")

        # Reset form
        self.reg_fname.clear()
        self.reg_lname.clear()
        self.reg_role.clear()
        self.radio_male.setChecked(True)
        self.capture_btn.setText("LOOK AT CAMERA & START TIMER")
        self.capture_btn.setEnabled(True)
