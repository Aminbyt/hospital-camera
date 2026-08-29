"""AI Detection Module - Handles YOLO, InsightFace, MediaPipe, and WHO Handwashing models."""
import os
import sys
import cv2
import torch
import config
import numpy as np
from ultralytics import YOLO
import math
import mediapipe as mp
import threading
from PyQt5.QtCore import QThread, pyqtSignal

# --- NEW IMPORTS FOR WHO GESTURE CLASSIFIER ---
import joblib
import pandas as pd
from collections import Counter, deque
# ----------------------------------------------

# --- PYINSTALLER DLL SECURITY FIX ---
if getattr(sys, 'frozen', False):
    os.add_dll_directory(sys._MEIPASS)

# --- GLOBAL AI CACHE & MUTEX LOCK ---
GLOBAL_INSIGHT_APP = None
GLOBAL_DB_EMBEDDINGS = {}
FACE_LOCK = threading.RLock() # <-- RLock prevents the thread from locking itself out!

import pickle

def initialize_face_engine(db_path):
    """Initializes InsightFace and loads staff photo embeddings ONCE at system boot!"""
    global GLOBAL_INSIGHT_APP, GLOBAL_DB_EMBEDDINGS
    
    # 1. FIXED INDENTATION: Aligned perfectly with 4 spaces
    with FACE_LOCK: 
        try:
            if GLOBAL_INSIGHT_APP is None:
                print("\n[INFO] ==================================================")
                print("[INFO] Initializing InsightFace Engine at system boot...")
                from insightface.app import FaceAnalysis
               
                # 2. UPGRADE RESTORED: We put the smarter model back!
                GLOBAL_INSIGHT_APP = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
                
                # Keeping the lowered threshold from our earlier fix!
                GLOBAL_INSIGHT_APP.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))
                print("[INFO] InsightFace Engine initialized successfully!")
                print("[INFO] ==================================================\n")

            if not GLOBAL_DB_EMBEDDINGS and os.path.exists(db_path):
                cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
            
            # --- 1. TRY TO LOAD MEMORY CACHE FIRST ---
            if os.path.exists(cache_path):
                print("\n[INFO] Loading face embeddings from high-speed cache...")
                with open(cache_path, 'rb') as f:
                    GLOBAL_DB_EMBEDDINGS = pickle.load(f)
                print(f"[INFO] Loaded {len(GLOBAL_DB_EMBEDDINGS)} staff members instantly!\n")
                return
            
            # --- 2. IF NO CACHE, BUILD IT (The slow loop) ---
            print("\n" + "="*55)
            print("  🏥 HOSPITAL AI - BUILDING FACE DATABASE (First Time):")
            print("="*55)
            total_people = 0
            for person_name in sorted(os.listdir(db_path)):
                person_dir = os.path.join(db_path, person_name)
                if os.path.isdir(person_dir):
                    GLOBAL_DB_EMBEDDINGS[person_name] = []
                   
                    for img_name in os.listdir(person_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(person_dir, img_name)
                            db_img = cv2.imread(img_path)
                            if db_img is not None:
                                faces = GLOBAL_INSIGHT_APP.get(db_img)
                                if faces:
                                    GLOBAL_DB_EMBEDDINGS[person_name].append(faces[0].normed_embedding)
                                else:
                                    print(f"  [WARNING] Face completely missed in {img_name}! (Too blurry/dark)")
                   
                    angle_count = len(GLOBAL_DB_EMBEDDINGS[person_name])
                    if angle_count > 0:
                        total_people += 1
                        print(f"  👤 {person_name:<25} | Loaded {angle_count} Angle(s)")
            print("="*55)
            print(f"  TOTAL REGISTERED STAFF: {total_people}")
            print("="*55 + "\n")
            
            # --- 3. SAVE TO MEMORY CACHE FOR NEXT BOOT ---
            with open(cache_path, 'wb') as f:
                pickle.dump(GLOBAL_DB_EMBEDDINGS, f)
            print("[INFO] Face cache saved! Future start-ups will be instant.")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize InsightFace engine: {e}")


def reset_face_cache():
    """Forces the AI to retrain its memory on the next scan."""
    global GLOBAL_DB_EMBEDDINGS
    with FACE_LOCK:
        GLOBAL_DB_EMBEDDINGS = {}
        cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)  
            
        initialize_face_engine(config.REG_PATH)
    print("[INFO] Face cache reloaded and saved with newly registered staff photos!")

def add_single_face_to_cache(person_name, img_path):
    """Instantly adds a single new photo to RAM without rebuilding the whole database."""
    global GLOBAL_INSIGHT_APP, GLOBAL_DB_EMBEDDINGS
   
    with FACE_LOCK:
        if person_name not in GLOBAL_DB_EMBEDDINGS:
            GLOBAL_DB_EMBEDDINGS[person_name] = []
           
        db_img = cv2.imread(img_path)
        if db_img is not None:
            faces = GLOBAL_INSIGHT_APP.get(db_img)
            if faces:
                GLOBAL_DB_EMBEDDINGS[person_name].append(faces[0].normed_embedding)
                print(f"[INFO] Instantly injected new angle for {person_name} into RAM!")
                
                # Resave the updated memory dictionary
                cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
                with open(cache_path, 'wb') as f:
                    pickle.dump(GLOBAL_DB_EMBEDDINGS, f)

def recognize_face_sync(frame_to_check, db_path=config.REG_PATH):
    """Thread-safe synchronous face recognition using a global mutex lock."""
    global GLOBAL_INSIGHT_APP, GLOBAL_DB_EMBEDDINGS
   
    with FACE_LOCK:  # <-- Guarantees only ONE camera accesses ONNX Runtime at a time!
        try:
            if GLOBAL_INSIGHT_APP is None or not GLOBAL_DB_EMBEDDINGS:
                initialize_face_engine(db_path)
               
            if not GLOBAL_DB_EMBEDDINGS:
                return "UNKNOWN"

            # 1. Detect Live Face safely inside the lock
            faces = GLOBAL_INSIGHT_APP.get(frame_to_check)
            if not faces:
                return "NO_FACE"

            detected_face = faces[0]
            best_match = "UNKNOWN"
            min_dist = 1.0  

            # 2. Compare live face against ALL saved angles in RAM
            for name, embeddings_list in GLOBAL_DB_EMBEDDINGS.items():
                for saved_embedding in embeddings_list:
                    dist = np.sum(np.square(detected_face.normed_embedding - saved_embedding))
                   
                    if dist < 0.55:
                        if dist < min_dist:
                            min_dist = dist
                            best_match = name

            return best_match

        except Exception as e:
            print(f"[ERROR] InsightFace Auth Error: {e}")
            return "UNKNOWN"


class FaceRecognitionThread(QThread):
    """Background thread wrapper kept for backward compatibility."""
    result_signal = pyqtSignal(str)

    def __init__(self, frame_to_check, db_path):
        super().__init__()
        self.frame = frame_to_check
        self.db_path = db_path

    def run(self):
        result = recognize_face_sync(self.frame, self.db_path)
        self.result_signal.emit(result)


class AIModels:
    """Manages all AI models: YOLO, MediaPipe, InsightFace, and WHO Handwashing."""

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

        # --- LOAD WHO HANDWASHING GESTURE CLASSIFIER ---
        self.who_session = None
        model_path = "who_cnn_lstm_model.onnx"
        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.who_session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self.who_input_name = self.who_session.get_inputs()[0].name
                print("✅ WHO Handwashing LSTM Neural Network loaded successfully!")
            except Exception as e:
                print(f"❌ Could not load who_lstm_model.onnx: {e}")
        else:
            print("⚠️ who_lstm_model.onnx not found. WHO gesture classification disabled.")

        # --- TEMPORAL SMOOTHING BUFFERS ---
        # Stores the last 45 frame predictions (~1.5 seconds) for smooth UI voting
        self.prediction_buffer = deque(maxlen=45)
       
        # <-- NEW: Stores a rolling window of 15 frames of hand coordinates for the LSTM
        self.lstm_sequence_buffer = deque(maxlen=30)

        initialize_face_engine(config.REG_PATH)

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
            if bboxC.width >= 0.10:
                return True
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

    # --- NEW METHODS FOR WHO STEP PREDICTION & SMOOTHING ---
    def predict_who_step(self, hand_landmarks_data):
        if not self.who_session or not hand_landmarks_data or not hand_landmarks_data.multi_hand_landmarks:
            self.lstm_sequence_buffer.clear()
            return 0
           
        sorted_hands = sorted(
            hand_landmarks_data.multi_hand_landmarks, key=lambda h: h.landmark[0].x
        )
       
        current_features = []
        for hand in sorted_hands[:2]:
            wrist = hand.landmark[0]
            middle_base = hand.landmark[9]
           
            import math
            hand_size = math.hypot(wrist.x - middle_base.x, wrist.y - middle_base.y)
            if hand_size == 0: hand_size = 1.0
           
            for lm in hand.landmark:
                current_features.extend([
                    (lm.x - wrist.x) / hand_size,
                    (lm.y - wrist.y) / hand_size,
                    (lm.z - wrist.z) / hand_size
                ])
               
        while len(current_features) < 126:
            current_features.append(0.0)

        # --- NEW: ADD HAND-TO-HAND DISTANCE (128 Dims) ---
        if len(sorted_hands) == 2:
            h1_w, h2_w = sorted_hands[0].landmark[0], sorted_hands[1].landmark[0]
            current_features.append(math.hypot(h1_w.x - h2_w.x, h1_w.y - h2_w.y))
           
            h1_i, h2_i = sorted_hands[0].landmark[8], sorted_hands[1].landmark[8]
            current_features.append(math.hypot(h1_i.x - h2_i.x, h1_i.y - h2_i.y))
        else:
            current_features.extend([1.0, 1.0])
           
        # --- ADD TO LSTM 30-FRAME TIME SEQUENCE ---
        self.lstm_sequence_buffer.append(current_features)
       
        # Wait for 1 full second of motion before predicting
        if len(self.lstm_sequence_buffer) < 30:
            return 0
           
        # Convert deque to shape (1, 30, 128) float32 numpy array
        seq_array = np.array([list(self.lstm_sequence_buffer)], dtype=np.float32)
       
        logits = self.who_session.run(None, {self.who_input_name: seq_array})[0]
        pred = int(np.argmax(logits, axis=1)[0])
       
        self.prediction_buffer.append(pred)
        most_common = Counter(self.prediction_buffer).most_common(1)[0][0]
        return most_common


    def get_smoothed_step(self):
        """Returns the most common prediction from the last 15 frames."""
        if not self.prediction_buffer:
            return 0
        most_common_step, _ = Counter(self.prediction_buffer).most_common(1)[0]
        return most_common_step

    def clear_buffer(self):
        """Resets the rolling vote when hands leave the sink or stop washing."""
        self.prediction_buffer.clear()
    # -------------------------------------------------------

    def cleanup(self):
        self.hands.close()
        self.face_detector.close()