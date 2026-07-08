"""AI Detection Module - Handles YOLO, InsightFace, and MediaPipe models."""
import os
import sys
import cv2
import torch
import config
import numpy as np
from ultralytics import YOLO
import math
import mediapipe as mp
from PyQt5.QtCore import QThread, pyqtSignal

# --- PYINSTALLER DLL SECURITY FIX ---
if getattr(sys, 'frozen', False):
    os.add_dll_directory(sys._MEIPASS)

# --- GLOBAL AI CACHE ---
GLOBAL_INSIGHT_APP = None
GLOBAL_DB_EMBEDDINGS = {}

def reset_face_cache():
    """Forces the AI to retrain its memory on the next scan."""
    global GLOBAL_DB_EMBEDDINGS
    GLOBAL_DB_EMBEDDINGS = {}
    print("[INFO] Face cache cleared! Will reload DB on next scan.")

class FaceRecognitionThread(QThread):
    """Background thread for InsightFace recognition with Multi-Angle support."""
    result_signal = pyqtSignal(str)

    def __init__(self, frame_to_check, db_path):
        super().__init__()
        self.frame = frame_to_check
        self.db_path = db_path

    def run(self):
        global GLOBAL_INSIGHT_APP, GLOBAL_DB_EMBEDDINGS
       
        try:
            if GLOBAL_INSIGHT_APP is None:
                print("[INFO] Safely loading InsightFace and ONNX Runtime in background thread...")
                from insightface.app import FaceAnalysis
               
                GLOBAL_INSIGHT_APP = FaceAnalysis(providers=['CPUExecutionProvider'])
                GLOBAL_INSIGHT_APP.prepare(ctx_id=0, det_size=(640, 640))
                print("[INFO] InsightFace Engine initialized successfully!")

            # 2. Build Database storing MULTIPLE embeddings per person
            if not GLOBAL_DB_EMBEDDINGS:
                print("[INFO] Building Multi-Angle Face Embeddings Database...")
                for person_name in os.listdir(self.db_path):
                    person_dir = os.path.join(self.db_path, person_name)
                    if os.path.isdir(person_dir):
                        GLOBAL_DB_EMBEDDINGS[person_name] = [] # Create a list for this person
                       
                        for img_name in os.listdir(person_dir):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(person_dir, img_name)
                                db_img = cv2.imread(img_path)
                                if db_img is not None:
                                    faces = GLOBAL_INSIGHT_APP.get(db_img)
                                    if faces:
                                        # Save every valid photo angle we find!
                                        GLOBAL_DB_EMBEDDINGS[person_name].append(faces[0].normed_embedding)
                       
                        print(f"[DB] Loaded {len(GLOBAL_DB_EMBEDDINGS[person_name])} angle(s) for: {person_name}")

            # 3. Detect Live Face
            faces = GLOBAL_INSIGHT_APP.get(self.frame)
            if not faces:
                self.result_signal.emit("NO_FACE")
                return

            detected_face = faces[0]
            best_match = "UNKNOWN"
            min_dist = 1.0  

            # 4. Compare live face against ALL saved angles for every person
            for name, embeddings_list in GLOBAL_DB_EMBEDDINGS.items():
                for saved_embedding in embeddings_list:
                    dist = np.sum(np.square(detected_face.normed_embedding - saved_embedding))
                   
                    # Relaxed threshold to 0.48 specifically for steep 180cm pitch angles
                    if dist < 0.48:
                        if dist < min_dist:
                            min_dist = dist
                            best_match = name

            self.result_signal.emit(best_match)

        except Exception as e:
            print(f"[ERROR] InsightFace Auth Error inside thread: {e}")
            self.result_signal.emit("UNKNOWN")
class AIModels:
    """Manages all AI models: YOLO, MediaPipe."""

    def __init__(self):
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

        print("[DEBUG] Loading MediaPipe Face Detection (Gatekeeper)...")
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )

        self.frame_counter = 0
        self.last_yolo_boxes = []

    def detect_ppe(self, frame):
        has_mask, has_hat = False, False
        self.frame_counter += 1

        if self.frame_counter % 3 == 0 or not self.last_yolo_boxes:
            results = self.yolo_model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
            self.last_yolo_boxes = []
            
            for r in results:
                for box in r.boxes:
                    self.last_yolo_boxes.append({
                        'xyxy': box.xyxy[0].cpu().numpy(),
                        'cls': int(box.cls[0])
                    })

        for box_data in self.last_yolo_boxes:
            class_id = box_data['cls']
            class_name = self.yolo_model.names[class_id]
            x1, y1, x2, y2 = map(int, box_data['xyxy'])
            
            color = (0, 255, 0) if class_name == 'mask' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if class_name == 'mask': has_mask = True
            elif class_name == 'hat': has_hat = True

        return frame, has_mask, has_hat

    def detect_face(self, frame_rgb):
        # This remains your "Gatekeeper" - lightweight and fast
        face_results = self.face_detector.process(frame_rgb)
        if not face_results.detections: return False

        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            if bboxC.width < 0.12: continue
            
            face_center_x = bboxC.xmin + (bboxC.width / 2)
            if face_center_x < 0.20 or face_center_x > 0.80: continue
               
            keypoints = detection.location_data.relative_keypoints
            right_eye, left_eye, nose = keypoints[0], keypoints[1], keypoints[2]
            
            dist_right = abs(nose.x - right_eye.x)
            dist_left = abs(left_eye.x - nose.x)
            if dist_left == 0 or dist_right == 0: continue
               
            ratio = dist_right / dist_left
            if 0.5 < ratio < 2.0: return True
        return False

    def detect_hands(self, frame_rgb):
        hand_results = self.hands.process(frame_rgb)
        return {
            'detected': bool(hand_results.multi_hand_landmarks),
            'hand_results': hand_results,
            'count': len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        }

    def draw_hand_landmarks(self, frame, hand_results):
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_hand_bbox(self, hand_landmarks, frame_w, frame_h):
        x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame_w
        x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame_w
        y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame_h
        y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame_h
        return [x_min, y_min, x_max, y_max]

    @staticmethod
    def bboxes_intersect(box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or 
                   box1[3] < box2[1] or box1[1] > box2[3])

    def cleanup(self):
        self.hands.close()
        self.face_detector.close()