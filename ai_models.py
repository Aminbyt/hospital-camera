"""AI Detection Module - Handles YOLO, DeepFace, and MediaPipe models."""
import torch
import tensorflow
import config
import os

from ultralytics import YOLO
from deepface import DeepFace

import cv2
import math

import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal


class FaceRecognitionThread(QThread):
    """Background thread for face recognition using DeepFace."""
    result_signal = pyqtSignal(str)

    def __init__(self, frame_to_check, db_path):
        super().__init__()
        self.frame = frame_to_check
        self.db_path = db_path

    def run(self):
        """Execute face recognition in background."""
        temp_img_path = "temp_check.jpg"
        cv2.imwrite(temp_img_path, self.frame)
        try:
            dfs = DeepFace.find(
                img_path=temp_img_path, 
                db_path=self.db_path, 
                enforce_detection=True, 
                silent=True
            )
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


class AIModels:
    """Manages all AI models: YOLO, DeepFace, MediaPipe."""

    def __init__(self):
        """Initialize all AI models."""
        print("[DEBUG] Loading YOLOv8 Model...")
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)

        print("[DEBUG] Loading MediaPipe Hands...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        print("[DEBUG] Loading MediaPipe Face Detection...")
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )

    def detect_ppe(self, frame):
        """Detect PPE (mask and hat) using YOLO.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            tuple: (frame_with_annotations, has_mask, has_hat)
        """
        has_mask = False
        has_hat = False

        results = self.yolo_model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
        for r in results:
            frame = r.plot()
            for box in r.boxes:
                class_name = self.yolo_model.names[int(box.cls[0])]
                if class_name == 'mask':
                    has_mask = True
                elif class_name == 'hat':
                    has_hat = True

        return frame, has_mask, has_hat

    def detect_face(self, frame_rgb):
        """Detect face presence using MediaPipe.
        
        Args:
            frame_rgb: RGB color space frame
            
        Returns:
            bool: True if face detected
        """
        face_results = self.face_detector.process(frame_rgb)
        return bool(face_results.detections)

    def detect_hands(self, frame_rgb):
        """Detect hands using MediaPipe.
        
        Args:
            frame_rgb: RGB color space frame
            
        Returns:
            dict: Hand detection results with landmarks
        """
        hand_results = self.hands.process(frame_rgb)
        
        return {
            'detected': bool(hand_results.multi_hand_landmarks),
            'hand_results': hand_results,
            'count': len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        }

    def draw_hand_landmarks(self, frame, hand_results):
        """Draw hand landmarks on frame.
        
        Args:
            frame: Input frame
            hand_results: Hand detection results
            
        Returns:
            frame: Frame with drawn landmarks
        """
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame

    def get_hand_bbox(self, hand_landmarks, frame_w, frame_h):
        """Calculate bounding box for a hand.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_w: Frame width
            frame_h: Frame height
            
        Returns:
            list: [x_min, y_min, x_max, y_max]
        """
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame_w
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame_w
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame_h
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame_h
        
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def bboxes_intersect(box1, box2):
        """Check if two bounding boxes intersect.
        
        Args:
            box1: [x_min, y_min, x_max, y_max]
            box2: [x_min, y_min, x_max, y_max]
            
        Returns:
            bool: True if bounding boxes overlap
        """
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                   box1[3] < box2[1] or box1[1] > box2[3])

    @staticmethod
    def calculate_hand_movement(current_pts, prev_pts):
        """Calculate maximum movement speed of hand landmarks.
        
        Args:
            current_pts: Current landmark positions
            prev_pts: Previous landmark positions
            
        Returns:
            float: Maximum movement distance
        """
        if not prev_pts:
            return 0
        
        speeds = [math.hypot(c[0] - p[0], c[1] - p[1]) 
                  for c, p in zip(current_pts, prev_pts)]
        
        return max(speeds) if speeds else 0

    def cleanup(self):
        """Release resources."""
        self.hands.close()
        self.face_detector.close()
