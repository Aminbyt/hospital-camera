"""Settings Tab UI Module - System configuration interface."""

import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QRadioButton,
    QSpinBox, QLineEdit, QGroupBox, QFileDialog, QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
import config


class SettingsTab(QWidget):
    """Settings tab for system configuration."""
    
    # Signals
    camera_source_changed = pyqtSignal(object)  # int or str (path)
    roi_requested = pyqtSignal()
    calibration_requested = pyqtSignal()
    toggles_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.build_ui()

    def build_ui(self):
        """Build the settings UI."""
        settings_layout = QVBoxLayout(self)
        settings_layout.setAlignment(Qt.AlignTop)
        settings_layout.setContentsMargins(30, 30, 30, 30)

        # --- MODULE CONFIGURATION ---
        detect_group = QGroupBox("MODULE CONFIGURATION")
        detect_layout = QVBoxLayout(detect_group)

        self.cb_mask = QCheckBox("VERIFY MEDICAL MASK ")
        self.cb_mask.setChecked(True)
        self.cb_hat = QCheckBox("VERIFY SURGICAL HAT ")
        self.cb_hat.setChecked(True)
        self.cb_wash = QCheckBox("VERIFY HAND WASHING")
        self.cb_wash.setChecked(True)

        self.cb_mask.stateChanged.connect(self.on_toggles_changed)
        self.cb_hat.stateChanged.connect(self.on_toggles_changed)
        self.cb_wash.stateChanged.connect(self.on_toggles_changed)

        detect_layout.addWidget(self.cb_mask)
        detect_layout.addWidget(self.cb_hat)
        detect_layout.addWidget(self.cb_wash)
        settings_layout.addWidget(detect_group)

        # --- VIDEO SOURCE ---
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

        # Webcam options
        self.webcam_widget = QWidget()
        webcam_layout = QHBoxLayout(self.webcam_widget)
        webcam_label = QLabel("CAMERA INDEX:")
        webcam_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.cam_spinbox = QSpinBox()
        self.cam_spinbox.setRange(0, 5)
        self.cam_spinbox.setValue(0)
        webcam_layout.addWidget(webcam_label)
        webcam_layout.addWidget(self.cam_spinbox)
        webcam_layout.addStretch()
        options_layout.addWidget(self.webcam_widget)

        # Video file options
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
        apply_btn.clicked.connect(self.on_camera_apply)
        controls_layout.addWidget(apply_btn)

        manual_btn = QPushButton("DRAW MANUAL ZONE")
        manual_btn.clicked.connect(lambda: self.roi_requested.emit())
        controls_layout.addWidget(manual_btn)

        calib_btn = QPushButton("AUTO-CALIBRATE SINK (RESET)")
        calib_btn.clicked.connect(lambda: self.calibration_requested.emit())
        controls_layout.addWidget(calib_btn)

        settings_layout.addSpacing(20)
        settings_layout.addLayout(controls_layout)
        settings_layout.addStretch()

    def toggle_source_options(self, is_webcam):
        """Toggle between webcam and video file options.
        
        Args:
            is_webcam: bool - True for webcam, False for video file
        """
        if is_webcam:
            self.webcam_widget.show()
            self.video_widget.hide()
        else:
            self.webcam_widget.hide()
            self.video_widget.show()

    def browse_file(self):
        """Open file browser for video selection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.video_path_input.setText(file_path)

    def get_camera_source(self):
        """Get the selected camera source.
        
        Returns:
            int or str: Camera index or video file path
        """
        if self.radio_webcam.isChecked():
            return self.cam_spinbox.value()
        else:
            return self.video_path_input.text()

    def on_camera_apply(self):
        """Handle camera source change."""
        source = self.get_camera_source()
        if self.radio_video.isChecked() and not source:
            return
        self.camera_source_changed.emit(source)

    def on_toggles_changed(self):
        """Handle detection toggle changes."""
        self.toggles_changed.emit()

    def get_detection_toggles(self):
        """Get current detection toggle states.
        
        Returns:
            dict: {'mask': bool, 'hat': bool, 'wash': bool}
        """
        return {
            'mask': self.cb_mask.isChecked(),
            'hat': self.cb_hat.isChecked(),
            'wash': self.cb_wash.isChecked()
        }
