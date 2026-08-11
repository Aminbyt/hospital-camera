"""Sink Calibration UI - Standalone dialog for defining the scrub zone without blocking."""

import cv2
import numpy as np
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

class SinkCalibrationDialog(QDialog):
    """UI Dialog that freezes a single frame for the user to set the Y-axis boundary."""
    
    def __init__(self, parent, frame, current_roi=None):
        super().__init__(parent)
        self.setWindowTitle("Calibrate Sink Scrub Zone")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        # Freeze a copy of the frame so we don't interfere with the live stream
        self.frame = frame.copy()
        
        # Default to the middle of the frame if no ROI exists
        self.roi_y = current_roi if current_roi is not None else int(self.frame.shape[0] * 0.5)

        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)

        self.info_label = QLabel("Click on the image to set the Alcohol Scrub Zone line (Y-axis).")
        self.info_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 5px;")
        self.layout.addWidget(self.info_label)

        # Image display area
        self.image_label = QLabel()
        self.image_label.setCursor(Qt.CrossCursor)
        self.image_label.mousePressEvent = self.on_image_click
        self.update_image()
        self.layout.addWidget(self.image_label)

        # Controls
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save Calibration")
        save_btn.setStyleSheet("background-color: #2d6a4f; color: white; padding: 10px; font-weight: bold;")
        save_btn.clicked.connect(self.accept)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: #dc3545; color: white; padding: 10px; font-weight: bold;")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        self.layout.addLayout(btn_layout)

    def on_image_click(self, event):
        """Updates the boundary line based on user click."""
        if event.button() == Qt.LeftButton:
            y = event.pos().y()
            # Clamp Y to ensure it stays within frame bounds
            self.roi_y = max(0, min(y, self.frame.shape[0]))
            self.update_image()

    def update_image(self):
        """Draws the calibration line on the frozen frame and updates the UI."""
        display_frame = self.frame.copy()
        h, w = display_frame.shape[:2]

        # Draw the visual indicator
        cv2.line(display_frame, (0, self.roi_y), (w, self.roi_y), (0, 0, 255), 2)
        cv2.putText(display_frame, "ALCOHOL SCRUB ZONE", (10, self.roi_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Convert to PyQt image format
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def get_roi(self):
        return self.roi_y


def create_roi_dialog(parent, frame, current_roi=None):
    """
    Spawns the calibration dialog.
    Returns: (accepted_boolean, new_roi_y)
    """
    if frame is None:
        return False, current_roi
        
    dialog = SinkCalibrationDialog(parent, frame, current_roi)
    if dialog.exec_() == QDialog.Accepted:
        return True, dialog.get_roi()
    return False, current_roi