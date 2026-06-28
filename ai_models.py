"""AI Detection Module - Handles YOLO, DeepFace, and MediaPipe models."""
import os
import sys

# --- PYINSTALLER DLL SECURITY FIX ---
if getattr(sys, 'frozen', False):
    os.add_dll_directory(sys._MEIPASS)
# ------------------------------------

import cv2
import torch
import config
import numpy as np
from ultralytics import YOLO
import math
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal

# ... (keep your imports) ...
from PyQt5.QtCore import QThread, pyqtSignal

# --- ADD THESE TWO GLOBAL VARIABLES ---
GLOBAL_RECOGNIZER = None
GLOBAL_LABEL_MAP = {}

def reset_face_cache():
    """Forces the AI to retrain its memory on the next scan."""
    global GLOBAL_RECOGNIZER
    GLOBAL_RECOGNIZER = None
    print("[INFO] Face cache cleared! Will retrain on next scan.")

class FaceRecognitionThread(QThread):
    """Background thread for pure OpenCV lightweight face recognition."""
    result_signal = pyqtSignal(str)

    def __init__(self, frame_to_check, db_path):
        super().__init__()
        self.frame = frame_to_check
        self.db_path = db_path

    def run(self):
        global GLOBAL_RECOGNIZER, GLOBAL_LABEL_MAP
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                self.result_signal.emit("NO_FACE")
                return

            (x, y, w, h) = faces[0]
            live_face_roi = gray_frame[y:y+h, x:x+w]

            # --- ONLY TRAIN THE DB IF WE HAVEN'T YET ---
            if GLOBAL_RECOGNIZER is None:
                print("[INFO] Training Face Database into memory...")
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                faces_db = []
                labels = []
                label_map = {}
                current_label = 0

                for person_name in os.listdir(self.db_path):
                    person_dir = os.path.join(self.db_path, person_name)
                    if os.path.isdir(person_dir):
                        label_map[current_label] = person_name
                        has_images = False
                        for img_name in os.listdir(person_dir):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(person_dir, img_name)
                                db_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if db_img is not None:
                                    db_faces = face_cascade.detectMultiScale(db_img, 1.1, 5)
                                    if len(db_faces) > 0:
                                        (dx, dy, dw, dh) = db_faces[0]
                                        faces_db.append(db_img[dy:dy+dh, dx:dx+dw])
                                        labels.append(current_label)
                                        has_images = True
                        if has_images:
                            current_label += 1

                if len(faces_db) == 0:
                    self.result_signal.emit("UNKNOWN")
                    return

                recognizer.train(faces_db, np.array(labels))
                GLOBAL_RECOGNIZER = recognizer
                GLOBAL_LABEL_MAP = label_map

            # Predict Identity Instantly from RAM
            label, confidence = GLOBAL_RECOGNIZER.predict(live_face_roi)

            if confidence < 115:
                self.result_signal.emit(GLOBAL_LABEL_MAP[label])
            else:
                self.result_signal.emit("UNKNOWN")

        except Exception as e:
            print(f"[ERROR] Face Auth Error: {e}")
            self.result_signal.emit("UNKNOWN")

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

        self.frame_counter = 0
        self.last_yolo_boxes = []

    def detect_ppe(self, frame):
        """Detect PPE using Frame Skipping for massive CPU speed boost."""
        has_mask = False
        has_hat = False
        self.frame_counter += 1

        # Only run the heavy AI inference every 3rd frame!
        if self.frame_counter % 3 == 0 or not self.last_yolo_boxes:
            results = self.yolo_model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
            self.last_yolo_boxes = []
            
            for r in results:
                for box in r.boxes:
                    self.last_yolo_boxes.append({
                        'xyxy': box.xyxy[0].cpu().numpy(),
                        'cls': int(box.cls[0])
                    })

        # Draw the saved boxes instantly without stalling the CPU
        for box_data in self.last_yolo_boxes:
            class_id = box_data['cls']
            class_name = self.yolo_model.names[class_id]
            x1, y1, x2, y2 = map(int, box_data['xyxy'])
            
            # Draw box and text
            color = (0, 255, 0) if class_name == 'mask' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if class_name == 'mask':
                has_mask = True
            elif class_name == 'hat':
                has_hat = True

        return frame, has_mask, has_hat
    def detect_face(self, frame_rgb):
        """Detect face presence using MediaPipe with strict 'Looking at Camera' filters."""
        face_results = self.face_detector.process(frame_rgb)
       
        if not face_results.detections:
            return False

        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
           
            # 1. THE PROXIMITY RULE: Ignore people far away in the background
            # Face width must take up at least 12% of the frame
            if bboxC.width < 0.12:
                continue

            # 2. THE CENTER RULE: Ignore people walking past the extreme edges
            face_center_x = bboxC.xmin + (bboxC.width / 2)
            if face_center_x < 0.20 or face_center_x > 0.80:
                continue
               
            # 3. THE "LOOK AT ME" RULE: Check if the head is turned sideways
            # MediaPipe Keypoints: 0=Right Eye, 1=Left Eye, 2=Nose
            keypoints = detection.location_data.relative_keypoints
            right_eye = keypoints[0]
            left_eye = keypoints[1]
            nose = keypoints[2]

            # Measure the horizontal distance from the nose to each eye
            dist_right = abs(nose.x - right_eye.x)
            dist_left = abs(left_eye.x - nose.x)

            # Avoid division by zero
            if dist_left == 0 or dist_right == 0:
                continue
               
            # If the head is turned sideways (profile), one eye is geometrically closer to the nose outline.
            # A perfect forward-facing head has a ratio of exactly 1.0.
            # We allow 0.5 to 2.0 to account for slight natural head tilts.
            ratio = dist_right / dist_left
            if 0.5 < ratio < 2.0:
                return True # The user is close, centered, and intentionally looking at the camera!

        return False


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
