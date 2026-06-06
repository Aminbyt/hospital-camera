import os
print("[DEBUG 1] Setting environment variables...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("[DEBUG 2] Loading YOLO (PyTorch) FIRST...")
from ultralytics import YOLO

print("[DEBUG 3] Loading DeepFace (TensorFlow)...")
from deepface import DeepFace

print("[DEBUG 4] Loading OpenCV and standard libraries...")
import sys, cv2, time, math, json
import numpy as np
import pandas as pd
import mediapipe as mp
import requests


print("[DEBUG 5] Loading PyQt5 UI LAST...")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QProgressBar, QTabWidget,
                             QRadioButton, QButtonGroup, QLineEdit, QPushButton,
                             QFileDialog, QSpinBox, QGroupBox, QCheckBox, QDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QPoint, QSettings, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QPen

print("[DEBUG 6] ALL LIBRARIES LOADED SUCCESSFULLY!")
# --- BACKGROUND WORKER FOR DEEPFACE ---
class FaceRecognitionThread(QThread):
    result_signal = pyqtSignal(str) #

    def __init__(self, frame_to_check, db_path):
        super().__init__()
        self.frame = frame_to_check
        self.db_path = db_path

    def run(self):
        temp_img_path = "temp_check.jpg"
        cv2.imwrite(temp_img_path, self.frame)
        try:
            dfs = DeepFace.find(img_path=temp_img_path, db_path=self.db_path, enforce_detection=True, silent=True)
            if len(dfs) > 0 and len(dfs[0]) > 0:
                matched_image_path = dfs[0].iloc[0]['identity']
                matched_name = os.path.basename(matched_image_path).split('.')[0]
                self.result_signal.emit(matched_name)
            else:
                self.result_signal.emit("UNKNOWN")
        except ValueError:
            self.result_signal.emit("NO_FACE")
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

class ROIDrawer(QWidget):
    def __init__(self, original_frame, current_roi, parent=None):
        super().__init__(parent)
        self.original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        self.original_h, self.original_w = self.original_frame.shape[:2]
        self.roi = current_roi if current_roi else None
        self.drawing = False
        self.start_pos = QPoint()
        self.end_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            scale = min(self.width() / self.original_w, self.height() / self.original_h)
            scaled_w = int(self.original_w * scale)
            scaled_h = int(self.original_h * scale)
            offset_x = (self.width() - scaled_w) // 2
            offset_y = (self.height() - scaled_h) // 2

            x1 = (self.start_pos.x() - offset_x) / scaled_w
            y1 = (self.start_pos.y() - offset_y) / scaled_h
            x2 = (event.x() - offset_x) / scaled_w
            y2 = (event.y() - offset_y) / scaled_h

            self.roi = [max(0, min(x1, x2)), max(0, min(y1, y2)), min(1, max(x1, x2)), min(1, max(y1, y2))]
            self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.fillRect(self.rect(), Qt.black)
        scale = min(self.width() / self.original_w, self.height() / self.original_h)
        scaled_w = int(self.original_w * scale)
        scaled_h = int(self.original_h * scale)
        offset_x = (self.width() - scaled_w) // 2
        offset_y = (self.height() - scaled_h) // 2

        scaled_frame = cv2.resize(self.original_frame, (scaled_w, scaled_h))
        qimg = QImage(scaled_frame.data, scaled_w, scaled_h, scaled_w * 3, QImage.Format_RGB888)
        qp.drawImage(offset_x, offset_y, qimg)

        qp.setPen(QPen(Qt.white, 3))
        if self.roi:
            rx = offset_x + int(self.roi[0] * scaled_w)
            ry = offset_y + int(self.roi[1] * scaled_h)
            rw = int((self.roi[2] - self.roi[0]) * scaled_w)
            rh = int((self.roi[3] - self.roi[1]) * scaled_h)
            qp.drawRect(rx, ry, rw, rh)

        if self.drawing:
            qp.setPen(QPen(Qt.gray, 2, Qt.DashLine))
            x, y = min(self.start_pos.x(), self.end_pos.x()), min(self.start_pos.y(), self.end_pos.y())
            w, h = abs(self.end_pos.x() - self.start_pos.x()), abs(self.end_pos.y() - self.start_pos.y())
            qp.drawRect(x, y, w, h)


class ScrubSinkKiosk(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hospital AI - Smart Scrub Sink")
        self.setGeometry(100, 100, 1200, 700)

        # --- STRICT BLACK & WHITE MINIMALIST THEME ---
        self.setStyleSheet("""
            QMainWindow { background-color: #ffffff; }
            QWidget { color: #000000; font-family: 'Segoe UI', Arial, sans-serif; background-color: #ffffff; }
            QTabWidget::pane { border: 1px solid #cccccc; }
            QTabBar::tab { background: #f5f5f5; border: 1px solid #cccccc; padding: 10px 15px; margin-right: 2px; font-weight: bold; font-size: 12px; min-width: 180px; }
            QTabBar::tab:selected { background: #ffffff; border-bottom: 2px solid #000000; }
            QGroupBox { border: 1px solid #000000; margin-top: 20px; font-weight: bold; font-size: 12px; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background: #ffffff; border: 2px solid #000000; padding: 10px 20px; font-weight: bold; font-size: 12px; }
            QPushButton:hover { background: #eeeeee; }
            QPushButton:pressed { background: #000000; color: #ffffff; }
            QLineEdit, QSpinBox { border: 1px solid #cccccc; padding: 8px; background: #ffffff; font-size: 12px; }
            QProgressBar { border: 2px solid #000000; background: #ffffff; height: 30px; text-align: center; }
            QProgressBar::chunk { background-color: #1b4332; }
            QCheckBox, QRadioButton { font-size: 12px; font-weight: 500; }
        """)

        # --- AI SETUP ---
        print("Loading YOLOv8 Model...")
        self.model = YOLO('runs/detect/train/weights/best.pt')

        print("Loading MediaPipe Hands...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(min_detection_confidence=0.5)

        # --- VARIABLES ---
        self.MIN_WASH_TIME = 35
        self.MAX_WASH_TIME = 60
        self.sink_y_start = None
        self.reset_wash_state()

        self.DB_PATH = "database"
        self.REG_PATH = os.path.join(self.DB_PATH, "REGISTER_PERSONS")
        self.INFO_PATH = os.path.join(self.DB_PATH, "INFORMATION")
        os.makedirs(self.REG_PATH, exist_ok=True)
        os.makedirs(self.INFO_PATH, exist_ok=True)

        self.current_user = None
        self.login_time = None
        self.is_authenticating = False
        self.last_clean_frame = None

        self.last_person_seen_time = time.time()
        self.last_auth_attempt_time = 0
        self.auth_cooldown = 4.0

        #Load saved manual Zone
        self.settings = QSettings("HospitalAI" , "KioskConfig")
        roi_data = self.settings.value("scrub_roi" , None)
        self.scrub_roi = json.loads(roi_data) if roi_data else None

        # Detection Toggles
        self.check_mask = True
        self.check_hat = True
        self.check_wash = True

        # --- UI LAYOUT: TABS ---
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.kiosk_tab = QWidget()
        self.registration_tab = QWidget()
        self.settings_tab = QWidget()

        self.tabs.addTab(self.kiosk_tab, "SYSTEM DASHBOARD")
        self.tabs.addTab(self.registration_tab, "REGISTRATION")
        self.tabs.addTab(self.settings_tab, "SETTINGS")

        self.build_kiosk_tab()
        self.build_registration_tab()
        self.build_settings_tab()

        # --- CAMERA SETUP ---
        self.current_source = 0
        self.cap = cv2.VideoCapture(self.current_source, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps>0 :
            delay = int(1000 / fps)
        else:
            delay = 30
        self.timer.start(delay)

    def build_registration_tab(self):
        # Use an HBox to put Form on Left, Camera on Right
        main_layout = QHBoxLayout(self.registration_tab)
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

        # --- NEW GENDER RADIO BUTTONS ---
        gender_widget = QWidget()
        gender_layout = QHBoxLayout(gender_widget)
        gender_layout.setContentsMargins(0, 5, 0, 5)

        gender_label = QLabel("GENDER:")
        gender_label.setFont(QFont("Arial", 12, QFont.Bold))

        self.radio_male = QRadioButton("MALE")
        self.radio_female = QRadioButton("FEMALE")
        self.radio_male.setChecked(True)  # Default to male
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

        # --- TIMER LOGIC ---
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        self.countdown_val = 0


    def start_countdown(self):
        fname = self.reg_fname.text().strip()
        lname = self.reg_lname.text().strip()
        if not fname or not lname:
            QMessageBox.warning(self, "ERROR", "First and Last name are required.")
            return
        if self.last_clean_frame is None:
            QMessageBox.warning(self, "ERROR", "Camera not ready.")
            return

        # Lock the button and start the clock!
        self.capture_btn.setEnabled(False)
        self.countdown_val = 5
        self.capture_btn.setText(f"TAKING PICTURE IN {self.countdown_val}...")
        self.countdown_timer.start(1000)

    def update_countdown(self):
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.capture_btn.setText(f"TAKING PICTURE IN {self.countdown_val}...")
        else:
            self.countdown_timer.stop()
            self.capture_btn.setText("PROCESSING...")
            self.register_new_user()

    def register_new_user(self):
        fname = self.reg_fname.text().strip()
        lname = self.reg_lname.text().strip()

        gender = "MALE" if self.radio_male.isChecked() else "FEMALE"

        full_name = f"{fname}_{lname}"

        # --- SAVE IN THEIR PERSONAL FOLDER ---
        person_folder = os.path.join(self.REG_PATH, full_name)
        os.makedirs(person_folder, exist_ok=True)

        img_path = os.path.join(person_folder, f"{full_name}.jpg")
        cv2.imwrite(img_path, self.last_clean_frame)

        # Clear the DeepFace cache in the new REG_PATH
        cache_file = os.path.join(self.REG_PATH, "representations_vgg_face.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)

        QMessageBox.information(self, "SUCCESS", f"Profile for {full_name} ({gender}) saved successfully!")

        # Reset the form
        self.reg_fname.clear()
        self.reg_lname.clear()
        self.reg_role.clear()
        self.radio_male.setChecked(True)
        self.capture_btn.setText("LOOK AT CAMERA & START TIMER")
        self.capture_btn.setEnabled(True)

    def reset_wash_state(self):
        self.current_wash_time = 0
        self.last_hand_seen_time = 0
        self.is_washing = False
        self.last_valid_wash_time = 0
        self.prev_hand_pts = None
        self.scrub_anchor_pos = None
        self.last_touch_time = 0
        self.scrub_bubble_radius = 200

        self.last_mask_seen_time = 0
        self.last_hat_seen_time = 0

    def build_kiosk_tab(self):
        main_layout = QHBoxLayout(self.kiosk_tab)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(30)

        self.video_label = QLabel("INITIALIZING CAMERA...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #000000; background: #000000; color: #ffffff;")
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, stretch=2)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(right_panel, stretch=1)

        title = QLabel("PROTOCOL STATUS")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(title)

        # Divider line
        line = QWidget()
        line.setFixedHeight(2)
        line.setStyleSheet("background-color: #000000;")
        right_layout.addWidget(line)
        right_layout.addSpacing(20)

        self.identity_label = QLabel("USER: NOT AUTHENTICATED")
        self.identity_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.identity_label.setStyleSheet("color: #ff0000;")
        right_layout.addWidget(self.identity_label)
        self.auto_status_label = QLabel("AUTO-SCAN: WAITING FOR FACE...")
        self.auto_status_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.auto_status_label.setStyleSheet("padding: 10px; background: #eeeeee; border: 1px solid #000000;")
        self.auto_status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.auto_status_label)
        right_layout.addSpacing(20)

        self.btn_auth = QPushButton("STEP 1: SCAN FACE (REMOVE MASK)")
        self.btn_auth.clicked.connect(self.start_authentication)
        right_layout.addWidget(self.btn_auth)
        right_layout.addSpacing(20)

        self.ppe_label = QLabel("[ PPE VERIFICATION PENDING ]")
        self.ppe_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.ppe_label.setStyleSheet("padding: 15px; border: 2px solid #000000; background: #ffffff;")
        right_layout.addWidget(self.ppe_label)
        right_layout.addSpacing(10)

        self.wash_label = QLabel("WASH TIMER: STANDBY")
        self.wash_label.setFont(QFont("Arial", 11))
        right_layout.addWidget(self.wash_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.MAX_WASH_TIME)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        right_layout.addWidget(self.progress_bar)

        right_layout.addStretch()

        self.master_status = QLabel("STATUS: ACTION REQUIRED")
        self.master_status.setFont(QFont("Arial", 14, QFont.Bold))
        self.master_status.setAlignment(Qt.AlignCenter)
        self.master_status.setStyleSheet(
            "color: #000000; border: 2px solid #000000; padding: 20px; background: #ffffff;")
        right_layout.addWidget(self.master_status)

    def build_settings_tab(self):
        settings_layout = QVBoxLayout(self.settings_tab)
        settings_layout.setAlignment(Qt.AlignTop)
        settings_layout.setContentsMargins(30, 30, 30, 30)

        detect_group = QGroupBox("MODULE CONFIGURATION")
        detect_layout = QVBoxLayout(detect_group)

        self.cb_mask = QCheckBox("VERIFY MEDICAL MASK ")
        self.cb_mask.setChecked(True)
        self.cb_hat = QCheckBox("VERIFY SURGICAL HAT ")
        self.cb_hat.setChecked(True)
        self.cb_wash = QCheckBox("VERIFY HAND WASHING")
        self.cb_wash.setChecked(True)

        self.cb_mask.stateChanged.connect(self.update_toggles)
        self.cb_hat.stateChanged.connect(self.update_toggles)
        self.cb_wash.stateChanged.connect(self.update_toggles)

        detect_layout.addWidget(self.cb_mask)
        detect_layout.addWidget(self.cb_hat)
        detect_layout.addWidget(self.cb_wash)
        settings_layout.addWidget(detect_group)

        source_group = QGroupBox("VIDEO SOURCE")
        source_layout = QVBoxLayout(source_group)

        self.radio_webcam = QRadioButton("LIVE WEBCAM STREAM")
        self.radio_webcam.setChecked(True)
        self.radio_video = QRadioButton("OFFLINE VIDEO FILE")

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_webcam)
        self.btn_group.addButton(self.radio_video)

        source_layout.addWidget(self.radio_webcam)
        source_layout.addWidget(self.radio_video)

        options_layout = QVBoxLayout()

        self.webcam_widget = QWidget()
        webcam_layout = QHBoxLayout(self.webcam_widget)
        webcam_label = QLabel("CAMERA INDEX:")
        webcam_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.cam_spinbox = QSpinBox()
        self.cam_spinbox.setRange(0, 5)
        webcam_layout.addWidget(webcam_label)
        webcam_layout.addWidget(self.cam_spinbox)
        webcam_layout.addStretch()
        options_layout.addWidget(self.webcam_widget)

        self.video_widget = QWidget()
        video_layout = QHBoxLayout(self.video_widget)
        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText("SELECT FILE PATH...")
        self.video_path_input.setReadOnly(True)

        browse_btn = QPushButton("BROWSE")
        browse_btn.clicked.connect(self.browse_file)

        video_layout.addWidget(self.video_path_input)
        video_layout.addWidget(browse_btn)
        self.video_widget.hide()
        options_layout.addWidget(self.video_widget)

        source_layout.addLayout(options_layout)
        settings_layout.addWidget(source_group)

        self.radio_webcam.toggled.connect(lambda: self.toggle_source_options(True))
        self.radio_video.toggled.connect(lambda: self.toggle_source_options(False))

        # --- CAMERA CONTROLS ---
        controls_layout = QHBoxLayout()
        apply_btn = QPushButton("RESTART SYSTEM FEED")
        apply_btn.clicked.connect(self.apply_camera_source)
        controls_layout.addWidget(apply_btn)

        manual_btn = QPushButton("DRAW MANUAL ZONE")
        manual_btn.clicked.connect(self.enter_roi_mode)
        controls_layout.addWidget(manual_btn)

        calib_btn = QPushButton("AUTO-CALIBRATE SINK (RESET)")
        calib_btn.clicked.connect(self.trigger_calibration)
        controls_layout.addWidget(calib_btn)

        settings_layout.addSpacing(20)
        settings_layout.addLayout(controls_layout)
        settings_layout.addStretch()

    def trigger_calibration(self):

        self.sink_y_start = None
        self.scrub_roi = None
        self.settings.setValue("scrub_roi", None)
        self.tabs.setCurrentIndex(0)

    def enter_roi_mode(self):
        ret , frame = self.cap.read()
        if not ret: return
        self.timer.stop()
        dlg = QDialog(self)
        dlg.setWindowTitle("Draw Manual Scrub Zone")
        dlg.resize(900,700)
        d_layout = QVBoxLayout(dlg)
        self.drawer = ROIDrawer(frame, self.scrub_roi)
        d_layout.addWidget(self.drawer)
        btn = QPushButton("SAVE MANUAL ZONE")
        btn.clicked.connect(lambda: self.save_roi(dlg))
        d_layout.addWidget(btn)
        dlg.exec_()
        self.timer.start(30)

    def save_roi(self, dlg):
        self.scrub_roi = self.drawer.roi
        self.settings.setValue("scrub_roi", json.dumps(self.scrub_roi))
        dlg.accept()

    def auto_detect_sink_line(self, frame):

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        crop_start = int(h * 0.4)
        crop_end = int(h * 0.95)
        roi = gray[crop_start:crop_end, :]

        # Blur and find edges
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Look for a continuous straight line
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=int(w * 0.3), maxLineGap=50)

        best_y = None
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle to ensure it is a horizontal line
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                if angle < 15 or angle > 165:
                    actual_y = y1 + crop_start
                    # Grab the highest prominent horizontal line
                    if best_y is None or actual_y < best_y:
                        best_y = actual_y

        return best_y

    def update_toggles(self):
        self.check_mask = self.cb_mask.isChecked()
        self.check_hat = self.cb_hat.isChecked()
        self.check_wash = self.cb_wash.isChecked()
        if not self.check_wash:
            self.reset_wash_state()

    def toggle_source_options(self, is_webcam):
        if is_webcam:
            self.webcam_widget.show()
            self.video_widget.hide()
        else:
            self.webcam_widget.hide()
            self.video_widget.show()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                   "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.video_path_input.setText(file_path)

    def apply_camera_source(self):
        if self.radio_webcam.isChecked():
            new_source = self.cam_spinbox.value()
        else:
            new_source = self.video_path_input.text()
            if not new_source:
                return

        self.timer.stop()
        if self.cap:
            self.cap.release()

        self.video_label.setText("STARTING VIDEO SOURCE...")
        self.progress_bar.setValue(0)

        self.current_source = new_source
        if isinstance(self.current_source,int):
            self.cap = cv2.VideoCapture(self.current_source , cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.current_source)
        self.reset_wash_state()
        self.sink_y_start = None

        self.tabs.setCurrentIndex(0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps>0:
            delay = int(1000 / fps)
        else:
            delay = 30
        self.timer.start(delay)

    def get_hand_bbox(self, hand_landmarks, frame_w, frame_h):
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame_w
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame_w
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame_h
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame_h
        return [x_min, y_min, x_max, y_max]

    def bboxes_intersect(self, box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

    def start_authentication(self):
        if self.last_clean_frame is None or self.is_authenticating:
            return

        # Enforce a 4-second cooldown so it doesn't spam DeepFace infinitely
        if time.time() - self.last_auth_attempt_time < self.auth_cooldown:
            return

        self.is_authenticating = True
        self.auto_status_label.setText("SCANNING FACE... PLEASE HOLD STILL")
        self.auto_status_label.setStyleSheet(
            "padding: 10px; background: #ffff00; color: #000000; border: 1px solid #000000;")

        # Start the background thread
        self.face_thread = FaceRecognitionThread(self.last_clean_frame, self.REG_PATH)
        self.face_thread.result_signal.connect(self.handle_auth_result)
        self.face_thread.start()

    def handle_auth_result(self, result):
        self.is_authenticating = False
        self.last_auth_attempt_time = time.time()  # Start cooldown timer

        if result == "NO_FACE":
            self.auto_status_label.setText("AUTO-SCAN: NO FACE DETECTED")
            self.auto_status_label.setStyleSheet(
                "padding: 10px; background: #eeeeee; color: #000000; border: 1px solid #000000;")
        elif result == "UNKNOWN":
            self.auto_status_label.setText("AUTO-SCAN: UNKNOWN USER")
            self.auto_status_label.setStyleSheet(
                "padding: 10px; background: #ffaaaa; color: #000000; border: 1px solid #000000;")
        else:
            # Success! Lock in the user and record the time.
            name_display = result.replace("_", " ")
            self.current_user = name_display
            self.login_time = time.strftime("%H:%M:%S")

            self.identity_label.setText(f"USER: {name_display.upper()} ✅")
            self.identity_label.setStyleSheet("color: #1b4332;")
            self.auto_status_label.setText("AUTO-SCAN: LOGGED IN")
            self.auto_status_label.setStyleSheet(
                "padding: 10px; background: #aaffaa; color: #000000; border: 1px solid #000000;")
            self.reset_wash_state()
            self.last_person_seen_time = time.time()  # Reset presence timer

    def logout_user(self):
        # --- 1. WRITE RECORD TO EXCEL ---
        if self.current_user:
            date_str = time.strftime("%Y-%m-%d")
            excel_file = os.path.join(self.INFO_PATH, f"{date_str}.xlsx")

            parts = self.current_user.split(" ")
            fname = parts[0] if len(parts) > 0 else "UNKNOWN"
            lname = parts[1] if len(parts) > 1 else ""

            wash_status = "YES" if self.current_wash_time >= self.MIN_WASH_TIME else "NO"

            mask_status = "YES" if (self.last_person_seen_time - self.last_mask_seen_time) <= 3.0 else "NO"
            hat_status = "YES" if (self.last_person_seen_time - self.last_hat_seen_time) <= 3.0 else "NO"

            new_data = pd.DataFrame([{
                "Date": date_str,
                "Name": fname,
                "Last name": lname,
                "Time": self.login_time,
                "Mask": mask_status,
                "Hat": hat_status,
                "Washing Complete": wash_status
            }])

            try:
                if os.path.exists(excel_file):
                    df = pd.read_excel(excel_file)
                    df = pd.concat([df, new_data], ignore_index=True)
                    df.to_excel(excel_file, index=False)
                else:
                    new_data.to_excel(excel_file, index=False)
            except Exception as e:
                print(f"[ERROR] Could not save to Excel: {e}")

            # --- 2. SEND NOTIFICATION TO HOSPITALBOT ---
            try:
                # Format a nice looking message
                bot_message = (
                    f"🏥 *Smart PPE Alert*\n"
                    f"👤 User: {fname} {lname}\n"
                    f"⏰ Time: {self.login_time}\n"
                    f"😷 Mask: {mask_status}\n"
                    f"‍🧑‍⚕️ Hat: {hat_status}\n"
                    f"🧼 Washing Complete: {wash_status}"
                )

                # TODO: Replace with your actual Bot API URL and target Chat/Channel ID
                bot_api_url = "https://tapi.bale.ai/1291761237:o-9xVmgV_Vw4iS-5XA9Yc4TWQ182YqNM5v8/sendMessage"
                payload = {
                    "chat_id": "6277616651",
                    "text": bot_message
                }

                # Send the message (timeout=3 prevents the UI from freezing if internet is slow!)
                # REMOVE THE '#' ON THE LINE BELOW TO ACTIVATE THE BOT:
                requests.post(bot_api_url, json=payload, timeout=3)

                print(f"[DEBUG] Bot payload prepared: {bot_message}")

            except Exception as e:
                print(f"[ERROR] Could not connect to bot server: {e}")

        # --- 3. RESET THE DASHBOARD ---
        self.current_user = None
        self.login_time = None
        self.identity_label.setText("USER: NOT AUTHENTICATED")
        self.identity_label.setStyleSheet("color: #ff0000;")
        self.auto_status_label.setText("AUTO-SCAN: WAITING FOR FACE...")
        self.auto_status_label.setStyleSheet("padding: 10px; background: #eeeeee; border: 1px solid #000000;")
        self.reset_wash_state()

    def update_frame(self):
        ret, frame = self.cap.read()

        if not ret:
            if isinstance(self.current_source, str):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return
            else:
                self.video_label.setText("VIDEO SIGNAL LOST")
                return

        if isinstance(self.current_source, str):
            self.cap.grab()

        self.last_clean_frame = frame.copy()
        clean_rgb = cv2.cvtColor(self.last_clean_frame, cv2.COLOR_BGR2RGB)

        frame_h, frame_w, _ = frame.shape
        has_mask, has_hat = False, False

        # --- 0. AUTO LOGIN PRESENCE CHECK ---
        face_results = self.face_detector.process(clean_rgb)
        if face_results.detections:
            self.last_person_seen_time = time.time()
            if self.current_user is None and not self.is_authenticating:
                self.start_authentication()

        if self.scrub_roi:
            self.sink_y_start = int(self.scrub_roi[1] * frame_h)
            rx1 , ry1 = int(self.scrub_roi[0] * frame_w), int(self.scrub_roi[1] * frame_h)
            rx2 , ry2 = int(self.scrub_roi[2] * frame_w), int(self.scrub_roi[3] * frame_h)
            cv2.rectangle(frame , (rx1, ry1), (rx2, ry2), (0, 0, 255), 2 )
            cv2.putText(frame , "VALID ZONE (MANUAL)" , (rx1 , ry1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            if self.sink_y_start is None:
                detected_y = self.auto_detect_sink_line(frame)
                if detected_y is not None:

                    elbow_offset = int(frame_h * 0.15)
                    self.sink_y_start = max(0, detected_y - elbow_offset)
                else:
                    # Fallback safely
                    self.sink_y_start = int(frame_h * 0.65)

        # Draw the dynamic Valid Anchor Zone Line (White for minimal theme)
            cv2.line(frame, (0, self.sink_y_start), (frame_w, self.sink_y_start), (0, 0, 255), 2)
            cv2.putText(frame, "VALID ZONE (AUTO)", (10, self.sink_y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

        # ---------------------------------------------------------
        # 1. Check PPE (YOLO)
        # ---------------------------------------------------------
        if self.check_mask or self.check_hat:
            results = self.model(frame, stream=True, conf=0.6, verbose=False)
            for r in results:
                frame = r.plot()
                for box in r.boxes:
                    class_name = self.model.names[int(box.cls[0])]
                    if class_name == 'mask':
                        has_mask = True
                        self.last_person_seen_time = time.time()
                        self.last_mask_seen_time = time.time()
                    elif class_name == 'hat':
                        has_hat = True
                        self.last_person_seen_time = time.time()
                        self.last_hat_seen_time = time.time()

        if not self.check_mask and not self.check_hat:
            self.ppe_label.setText("[ PPE CHECK:⏭️ DISABLED ]")
        else:
            m_text = "VERIFIED ✅" if has_mask else "MISSING ❌"
            h_text = "VERIFIED ✅" if has_hat else "MISSING ❌"
            self.ppe_label.setText(f"[ MASK: {m_text} ]    [ HAT: {h_text} ]")

        # ---------------------------------------------------------
        # 2. Check Hands (Strict Scrub Logic)
        # ---------------------------------------------------------
        actively_washing = False
        hands_count = 0

        if self.check_wash:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            movement_speed = 0

            if hand_results.multi_hand_landmarks:
                hands_count = len(hand_results.multi_hand_landmarks)

                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                hand0 = hand_results.multi_hand_landmarks[0]
                current_hand_pts = [
                    (hand0.landmark[0].x * frame_w, hand0.landmark[0].y * frame_h),
                    (hand0.landmark[4].x * frame_w, hand0.landmark[4].y * frame_h),
                    (hand0.landmark[8].x * frame_w, hand0.landmark[8].y * frame_h)
                ]

                if self.prev_hand_pts:
                    speeds = [math.hypot(c[0] - p[0], c[1] - p[1]) for c, p in
                              zip(current_hand_pts, self.prev_hand_pts)]
                    movement_speed = max(speeds)
                self.prev_hand_pts = current_hand_pts

                is_touching = False

                if hands_count == 2:
                    w1 = hand_results.multi_hand_landmarks[0].landmark[0]
                    w2 = hand_results.multi_hand_landmarks[1].landmark[0]
                    wrist_distance = math.hypot((w1.x - w2.x) * frame_w, (w1.y - w2.y) * frame_h)

                    if wrist_distance > 65:
                        box1 = self.get_hand_bbox(hand_results.multi_hand_landmarks[0], frame_w, frame_h)
                        box2 = self.get_hand_bbox(hand_results.multi_hand_landmarks[1], frame_w, frame_h)

                        if self.bboxes_intersect(box1, box2):
                            anchor_x = (box1[0] + box1[2] + box2[0] + box2[2]) / 4
                            anchor_y = (box1[1] + box1[3] + box2[1] + box2[3]) / 4

                            if anchor_y > self.sink_y_start:
                                is_touching = True
                                self.scrub_anchor_pos = (anchor_x, anchor_y)

                                h1_wrist = hand_results.multi_hand_landmarks[0].landmark[0]
                                h1_mid = hand_results.multi_hand_landmarks[0].landmark[12]
                                hand_size = math.hypot((h1_wrist.x - h1_mid.x) * frame_w,
                                                       (h1_wrist.y - h1_mid.y) * frame_h)

                                self.scrub_bubble_radius = max(hand_size * 2.5, 250)

                time_since_last_touch = time.time() - self.last_touch_time
                is_moving = movement_speed > 2.0

                if is_touching and is_moving:
                    actively_washing = True
                    self.last_touch_time = time.time()
                elif hands_count >= 1 and is_moving and time_since_last_touch < 2.5:
                    if self.scrub_anchor_pos:
                        curr_x = hand0.landmark[9].x * frame_w
                        curr_y = hand0.landmark[9].y * frame_h
                        dist_to_anchor = math.hypot(curr_x - self.scrub_anchor_pos[0],
                                                    curr_y - self.scrub_anchor_pos[1])

                        radius = int(self.scrub_bubble_radius)
                        cv2.circle(frame, (int(self.scrub_anchor_pos[0]), int(self.scrub_anchor_pos[1])), radius,
                                   (0, 255, 255), 2)

                        if dist_to_anchor < self.scrub_bubble_radius:
                            actively_washing = True
            else:
                self.prev_hand_pts = None

            current_time = time.time()
            if actively_washing:
                self.last_valid_wash_time = current_time
            else:
                if hands_count < 2 and (current_time - self.last_valid_wash_time) < 3.0:
                    actively_washing = True
        if self.current_user is not None:
            if time.time() - self.last_person_seen_time > 5.0:
                self.logout_user()
        # ---------------------------------------------------------
        # UPDATE UI
        # ---------------------------------------------------------
        if not self.check_wash:
            self.wash_label.setText("WASH TIMER:⏭️ DISABLED")
            self.progress_bar.setValue(self.MAX_WASH_TIME)
        else:
            if self.current_wash_time < self.MAX_WASH_TIME:

                # 1. Calculate Time
                if actively_washing:
                    if self.is_washing:
                        time_spent = time.time() - self.last_hand_seen_time
                        self.current_wash_time += time_spent
                    self.is_washing = True
                    self.last_hand_seen_time = time.time()

                # 2. Build the Base Status Sentence
                if self.current_wash_time >= self.MIN_WASH_TIME:
                    base_text = "WASHING (MINIMUM REACHED ✅ - YOU CAN CONTINUE)"
                else:
                    base_text = "WASHING IN PROGRESS ...⏳"

                # 3. Apply Text & Stack Warnings Underneath
                if actively_washing:
                    self.wash_label.setText(base_text)
                else:
                    self.is_washing = False

                    if hands_count == 0:
                        if self.current_wash_time >= self.MIN_WASH_TIME:
                            self.wash_label.setText("WASH COMPLETE: ✅")
                        elif self.current_wash_time > 0:
                            self.wash_label.setText(f"{base_text}\n[ PAUSED: RETURN HANDS TO ZONE ]")
                        else:
                            self.wash_label.setText("WASH TIMER:⏯️ Pause")
                    elif hands_count == 1:
                        self.wash_label.setText(f"{base_text}\n[ ⚠️WARNING: USE BOTH HANDS ]")
                    elif hands_count == 2:
                        self.wash_label.setText(f"{base_text}\n[⚠️ WARNING: INTERLOCK HANDS ]")

                self.progress_bar.setValue(int(self.current_wash_time))
            else:
                self.wash_label.setText("MAXIMUM WASH REACHED: ✅")
                self.progress_bar.setValue(self.MAX_WASH_TIME)
# ---------------------------------------------------------
        # 3. MASTER LOGIC CHECK
        # ---------------------------------------------------------
        if self.current_user is None:
            self.master_status.setText("SYSTEM LOCKED: PLEASE AUTHENTICATE")
            self.master_status.setStyleSheet("color: #ffffff; background-color: #000000; border: 2px solid #000000; padding: 20px;")
            self.wash_label.setText("WASH TIMER: LOCKED")
            self.ppe_label.setText("[ PPE CHECK: LOCKED ]")
        else:
            master_ready = True
            if self.check_mask and not has_mask: master_ready = False
            if self.check_hat and not has_hat: master_ready = False
            if self.check_wash and self.current_wash_time < self.MIN_WASH_TIME: master_ready = False

            if master_ready:
                self.master_status.setText("STATUS: PROCEED TO Operating Room ✅")
                self.master_status.setStyleSheet(
                    "color: #ffffff; background-color: #000000; border: 2px solid #000000; padding: 20px;")
            else:
                self.master_status.setText("STATUS: ACTION REQUIRED ⚠️")
                self.master_status.setStyleSheet(
                    "color: #000000; background-color: #ffffff; border: 2px solid #000000; padding: 20px;")

        # ---------------------------------------------------------
        # Render Camera to UI (Main Dashboard)
        # ---------------------------------------------------------
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

        # ---------------------------------------------------------
        # Render Camera to Registration Tab (Photo Booth)
        # ---------------------------------------------------------
        if self.tabs.currentIndex() == 1 and self.last_clean_frame is not None:
            reg_frame = self.last_clean_frame.copy()

            # Draw big red countdown on the video!
            if self.countdown_val > 0 and not self.capture_btn.isEnabled():
                text = str(self.countdown_val)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Center the text perfectly
                text_size = cv2.getTextSize(text, font, 7, 15)[0]
                text_x = int((w - text_size[0]) / 2)
                text_y = int((h + text_size[1]) / 2)
                cv2.putText(reg_frame, text, (text_x, text_y), font, 7, (0, 0, 255), 15, cv2.LINE_AA)

            reg_rgb = cv2.cvtColor(reg_frame, cv2.COLOR_BGR2RGB)
            reg_q_img = QImage(reg_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.reg_video_label.setPixmap(QPixmap.fromImage(reg_q_img).scaled(
                self.reg_video_label.width(), self.reg_video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    print("[DEBUG 8] Starting PyQt Application...")
    app = QApplication(sys.argv)
    print("[DEBUG 9] Booting up Kiosk Class...")
    window = ScrubSinkKiosk()
    print("[DEBUG 10] Drawing Window to Screen...")
    window.show()
    print("[DEBUG 11] Entering Main Loop!")
    sys.exit(app.exec_())