"""Sink Calibration Module - Handles ROI drawing and sink line auto-detection."""

import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen


class ROIDrawer(QWidget):
    """Interactive widget for drawing Region of Interest (ROI) for scrubbing zone."""

    def __init__(self, original_frame, current_roi, parent=None):
        super().__init__(parent)
        self.original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        self.original_h, self.original_w = self.original_frame.shape[:2]
        self.roi = current_roi if current_roi else None
        self.drawing = False
        self.start_pos = QPoint()
        self.end_pos = QPoint()
        self.setFocusPolicy(Qt.StrongFocus)

    def mousePressEvent(self, event):
        """Handle mouse press - start drawing."""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move - update drawing rectangle."""
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release - finalize ROI."""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            
            # Calculate scaling
            scale = min(self.width() / self.original_w, self.height() / self.original_h)
            scaled_w = int(self.original_w * scale)
            scaled_h = int(self.original_h * scale)
            offset_x = (self.width() - scaled_w) // 2
            offset_y = (self.height() - scaled_h) // 2

            # Convert screen coordinates to normalized frame coordinates
            x1 = (self.start_pos.x() - offset_x) / scaled_w
            y1 = (self.start_pos.y() - offset_y) / scaled_h
            x2 = (event.x() - offset_x) / scaled_w
            y2 = (event.y() - offset_y) / scaled_h

            # Store normalized ROI coordinates [0, 1]
            self.roi = [
                max(0, min(x1, x2)), 
                max(0, min(y1, y2)), 
                min(1, max(x1, x2)), 
                min(1, max(y1, y2))
            ]
            self.update()

    def paintEvent(self, event):
        """Paint the frame and ROI rectangle."""
        qp = QPainter(self)
        qp.fillRect(self.rect(), Qt.black)
        
        # Calculate scaling
        scale = min(self.width() / self.original_w, self.height() / self.original_h)
        scaled_w = int(self.original_w * scale)
        scaled_h = int(self.original_h * scale)
        offset_x = (self.width() - scaled_w) // 2
        offset_y = (self.height() - scaled_h) // 2

        # Draw scaled frame
        scaled_frame = cv2.resize(self.original_frame, (scaled_w, scaled_h))
        qimg = QImage(scaled_frame.data, scaled_w, scaled_h, scaled_w * 3, QImage.Format_RGB888)
        qp.drawImage(offset_x, offset_y, qimg)

        # Draw existing ROI
        qp.setPen(QPen(Qt.white, 3))
        if self.roi:
            rx = offset_x + int(self.roi[0] * scaled_w)
            ry = offset_y + int(self.roi[1] * scaled_h)
            rw = int((self.roi[2] - self.roi[0]) * scaled_w)
            rh = int((self.roi[3] - self.roi[1]) * scaled_h)
            qp.drawRect(rx, ry, rw, rh)

        # Draw current drawing rectangle
        if self.drawing:
            qp.setPen(QPen(Qt.gray, 2, Qt.DashLine))
            x = min(self.start_pos.x(), self.end_pos.x())
            y = min(self.start_pos.y(), self.end_pos.y())
            w = abs(self.end_pos.x() - self.start_pos.x())
            h = abs(self.end_pos.y() - self.start_pos.y())
            qp.drawRect(x, y, w, h)


class SinkCalibration:
    """Handles automatic and manual sink line calibration."""

    @staticmethod
    def auto_detect_sink_line(frame):
        """Auto-detect the sink line using edge detection and Hough transform.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            int or None: Y coordinate of detected sink line, or None if not found
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop to region of interest (40% to 95% of frame height)
        crop_start = int(h * 0.4)
        crop_end = int(h * 0.95)
        roi = gray[crop_start:crop_end, :]

        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)

        # Detect lines using Hough Line Probability transform
        lines = cv2.HoughLinesP(
            edges, 
            1, 
            np.pi / 180, 
            threshold=100, 
            minLineLength=int(w * 0.3), 
            maxLineGap=50
        )

        best_y = None
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle to ensure it's a horizontal line
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                
                # Accept lines close to horizontal (< 15 degrees or > 165 degrees)
                if angle < 15 or angle > 165:
                    actual_y = y1 + crop_start
                    
                    # Keep the highest (topmost) prominent horizontal line
                    if best_y is None or actual_y < best_y:
                        best_y = actual_y

        return best_y

    @staticmethod
    def calculate_sink_y_start(detected_y, frame_h):
        """Calculate the valid scrubbing zone top boundary.
        
        Args:
            detected_y: Auto-detected sink line Y coordinate
            frame_h: Frame height
            
        Returns:
            int: Adjusted Y coordinate for valid zone
        """
        if detected_y is not None:
            elbow_offset = int(frame_h * 0.15)
            return max(0, detected_y - elbow_offset)
        else:
            # Fallback to center-lower part of frame
            return int(frame_h * 0.65)

    @staticmethod
    def draw_sink_zone(frame, sink_y_start, manual=False):
        """Draw the valid scrubbing zone on the frame.
        
        Args:
            frame: Input frame
            sink_y_start: Y coordinate of zone boundary
            manual: bool - Whether this is a manual ROI
            
        Returns:
            frame: Frame with drawn zone
        """
        frame_h, frame_w = frame.shape[:2]
        
        cv2.line(frame, (0, sink_y_start), (frame_w, sink_y_start), (0, 0, 255), 2)
        
        zone_type = "MANUAL" if manual else "AUTO"
        cv2.putText(frame, f"VALID ZONE ({zone_type})", (10, sink_y_start - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame

    @staticmethod
    def draw_manual_roi(frame, roi):
        """Draw manual ROI rectangle on frame.
        
        Args:
            frame: Input frame
            roi: [x1, y1, x2, y2] normalized coordinates (0-1)
            
        Returns:
            frame: Frame with drawn ROI
        """
        if roi:
            frame_h, frame_w = frame.shape[:2]
            rx1 = int(roi[0] * frame_w)
            ry1 = int(roi[1] * frame_h)
            rx2 = int(roi[2] * frame_w)
            ry2 = int(roi[3] * frame_h)
            
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            cv2.putText(frame, "VALID ZONE (MANUAL)", (rx1, ry1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame


def create_roi_dialog(parent, frame, current_roi):
    """Create a dialog for manual ROI drawing.
    
    Args:
        parent: Parent widget
        frame: Current camera frame
        current_roi: Current ROI if exists
        
    Returns:
        tuple: (accepted, roi) where roi is normalized coordinates or None
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Draw Manual Scrub Zone")
    dlg.resize(900, 700)
    
    d_layout = QVBoxLayout(dlg)
    
    drawer = ROIDrawer(frame, current_roi)
    d_layout.addWidget(drawer)
    
    def save_roi():
        dlg.accepted_roi = drawer.roi
        dlg.accept()
    
    btn = QPushButton("SAVE MANUAL ZONE")
    btn.clicked.connect(save_roi)
    d_layout.addWidget(btn)
    
    dlg.accepted_roi = None
    dlg.exec_()
    
    return dlg.result() == QDialog.Accepted, dlg.accepted_roi
