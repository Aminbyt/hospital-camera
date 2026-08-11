"""AI Detection Module - Handles Shared Models and Per-Sink State."""

import os
import sys
import cv2
import numpy as np
import threading
import queue
import pickle
import math
import time
from collections import Counter, deque
import logging
import mediapipe as mp
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal

import config

if getattr(sys, 'frozen', False):
    os.add_dll_directory(sys._MEIPASS)


# --- 1. FACE RECOGNITION SERVICE (ASYNC WORKER) ---

# --- 1. FACE RECOGNITION SERVICE (MESSAGE-DRIVEN WORKER) ---

class FaceRecognitionService(threading.Thread):
    """Dedicated background thread managing absolute ownership of the InsightFace engine."""
    _instance = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super(FaceRecognitionService, cls).__new__(cls)
                
                # --- ADD THIS MISSING LINE ---
                threading.Thread.__init__(cls._instance)
                
                cls._instance.daemon = True
                cls._instance.request_queue = queue.Queue()
                cls._instance.insight_app = None
                cls._instance.db_embeddings = {}
                cls._instance.running = True
                
                # Notice: No state_lock needed anymore! Thread-confinement guarantees safety.
                cls._instance._load_face_engine()
                cls._instance.start()
            return cls._instance
    def _load_face_engine(self):
        """Initializes the engine during thread bootup."""
        logging.info("\n[INFO] ==================================================")
        logging.info("[INFO] Initializing Central Face Recognition Service...")
        from insightface.app import FaceAnalysis
        self.insight_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.insight_app.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))

        cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
        if not self.db_embeddings and os.path.exists(config.REG_PATH):
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    self.db_embeddings = pickle.load(f)
                logging.info(f"     [SUCCESS] Loaded {len(self.db_embeddings)} staff members from cache.")
            else:
                logging.info("     [INFO] Building Face Database (First Time)...")
                total_people = 0
                for person_name in sorted(os.listdir(config.REG_PATH)):
                    person_dir = os.path.join(config.REG_PATH, person_name)
                    if os.path.isdir(person_dir):
                        self.db_embeddings[person_name] = []
                        for img_name in os.listdir(person_dir):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(person_dir, img_name)
                                db_img = cv2.imread(img_path)
                                if db_img is not None:
                                    faces = self.insight_app.get(db_img)
                                    if faces:
                                        self.db_embeddings[person_name].append(faces[0].normed_embedding)
                        
                        angle_count = len(self.db_embeddings[person_name])
                        if angle_count > 0:
                            total_people += 1
                            logging.info(f"       {person_name:<25} | Loaded {angle_count} Angle(s)")
                
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.db_embeddings, f)
                logging.info("     [SUCCESS] Face cache saved!")
        logging.info("[INFO] ==================================================\n")

    def run(self):
        """Main loop: Sequentially processes commands to ensure thread safety."""
        logging.info("[INFO] Face Recognition Worker Thread Started.")
        while self.running:
            try:
                request = self.request_queue.get(timeout=1.0)
                command = request[0]
                
                if command == "RECOGNIZE":
                    _, sink_name, frame, callback = request
                    result = self._internal_recognize_face(frame)
                    if callback:
                        callback(result)
                        
                elif command == "ADD_FACE":
                    _, person_name, img_path, done_event = request
                    self._internal_add_face(person_name, img_path)
                    if done_event:
                        done_event.set()
                        
                elif command == "RESET_CACHE":
                    _, done_event = request
                    self._internal_reset_cache()
                    if done_event:
                        done_event.set()
                        
                self.request_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"[ERROR] Face Recognition Worker crashed: {e}")

    # --- PUBLIC API (Drops commands into the queue) ---

    def request_recognition(self, sink_name, frame, callback):
        self.request_queue.put(("RECOGNIZE", sink_name, frame, callback))

    def add_face_to_cache(self, person_name, img_path):
        """Drops an add command into the queue and waits for completion."""
        done_event = threading.Event()
        self.request_queue.put(("ADD_FACE", person_name, img_path, done_event))
        done_event.wait()

    def reset_face_cache(self):
        """Drops a reset command into the queue and waits for completion."""
        done_event = threading.Event()
        self.request_queue.put(("RESET_CACHE", done_event))
        done_event.wait()

    # --- INTERNAL EXECUTION METHODS (Only run by self.run loop) ---

    def _internal_recognize_face(self, frame):
        if not self.db_embeddings:
            return "UNKNOWN"
        
        faces = self.insight_app.get(frame)
        if not faces:
            return "NO_FACE"

        detected_face = faces[0]
        best_match = "UNKNOWN"
        min_dist = 1.0  

        for name, embeddings_list in self.db_embeddings.items():
            for saved_embedding in embeddings_list:
                dist = np.sum(np.square(detected_face.normed_embedding - saved_embedding))
                if dist < 0.48 and dist < min_dist:
                    min_dist = dist
                    best_match = name
        return best_match

    def _internal_add_face(self, person_name, img_path):
        if person_name not in self.db_embeddings:
            self.db_embeddings[person_name] = []
        
        db_img = cv2.imread(img_path)
        if db_img is not None:
            faces = self.insight_app.get(db_img)
            if faces:
                self.db_embeddings[person_name].append(faces[0].normed_embedding)
                cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
                with open(cache_path, 'wb') as f:
                    pickle.dump(self.db_embeddings, f)

    def _internal_reset_cache(self):
        self.db_embeddings = {}
        cache_path = os.path.join(config.DB_PATH, "face_cache.pkl")
        if os.path.exists(cache_path):
            os.remove(cache_path)
        self._load_face_engine()

    def stop(self):
        """Safely shuts down the face recognition thread."""
        self.running = False
        # Drop a dummy command to wake up the thread
        self.request_queue.put(("SHUTDOWN", None, None, None))
        self.join(timeout=3.0)

class AIModelManager:
    """Singleton: Loads heavy AI models ONCE into RAM and shares them across all sinks."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AIModelManager, cls).__new__(cls)
                cls._instance._initialize_models()
            return cls._instance

    def _initialize_models(self):
        logging.info("\n[INFO] ==================================================")
        logging.info("[INFO] Initializing Global AI Model Manager...")
        
        # 1. YOLOv8 (Shared)
        logging.info("  -> Loading Shared YOLOv8 Model...")
        self.yolo_lock = threading.Lock()
        self.yolo_model = YOLO(config.YOLO_MODEL_PATH)
        
        # 2. WHO LSTM ONNX (Shared)
        logging.info("  -> Loading Shared WHO LSTM Model...")
        self.who_session = None
        self.who_lock = threading.Lock()
        model_path = "who_cnn_lstm_model.onnx"
        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                
                # --- EXPLICIT ONNX THREAD & PERFORMANCE TUNING ---
                so = ort.SessionOptions()
                
                # Fetch baseline targets from config (allows easy benchmarking)
                so.intra_op_num_threads = getattr(config, 'ORT_INTRA_THREADS', 2)
                so.inter_op_num_threads = getattr(config, 'ORT_INTER_THREADS', 1)
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                
                # CRITICAL FIX FOR 5-CAMERA CONTENTION:
                # Disable thread spinning. Forces threads to sleep when idle 
                # rather than burning 100% of a core polling for new frames.
                so.add_session_config_entry('session.intra_op.allow_spinning', '0')
                so.add_session_config_entry('session.inter_op.allow_spinning', '0')
                
                self.who_session = ort.InferenceSession(
                    model_path, 
                    sess_options=so, 
                    providers=['CPUExecutionProvider']
                )
                self.who_input_name = self.who_session.get_inputs()[0].name
                logging.info(f"     [SUCCESS] ONNX WHO Engine Loaded (Intra: {so.intra_op_num_threads}, Inter: {so.inter_op_num_threads})")
            except Exception as e:
                logging.error(f"     [ERROR] Failed to load ONNX: {e}")
        else:
            logging.warning("     [WARNING] who_cnn_lstm_model.onnx not found.")

        # 3. InsightFace (Shared)
        self.face_lock = threading.Lock()
        self.insight_app = None
        self.db_embeddings = {}
        logging.info("[INFO] ==================================================\n")

    def run_yolo(self, frame):
        with self.yolo_lock:
            results = self.yolo_model(frame, stream=True, conf=config.YOLO_CONF_THRESHOLD, verbose=False)
            parsed_boxes = []
            for r in results:
                for box in r.boxes:
                    parsed_boxes.append({
                        'xyxy': box.xyxy[0].cpu().numpy(),
                        'cls': int(box.cls[0]),
                        'name': self.yolo_model.names[int(box.cls[0])]
                    })
            return parsed_boxes

    def run_who_lstm(self, seq_array):
        if not self.who_session:
            return 0
        with self.who_lock:
            # RESTORED FROM MAIN: Raw argmax, no strict softmax thresholds!
            logits = self.who_session.run(None, {self.who_input_name: seq_array})[0]
            return int(np.argmax(logits, axis=1)[0])

# --- 3. BACKWARD COMPATIBILITY ALIASES ---

def reset_face_cache():
    FaceRecognitionService().reset_face_cache()

def add_single_face_to_cache(person_name, img_path):
    FaceRecognitionService().add_face_to_cache(person_name, img_path)

def recognize_face_sync(frame_to_check, db_path=None):
    """Retained only for legacy synchronous calls, if any exist."""
    return FaceRecognitionService()._recognize_face(frame_to_check)


# --- 4. PER-SINK AI STATE ---

class PerSinkAIState:
    """Lightweight class created per-camera holding temporal state and MediaPipe instances."""
    def __init__(self):
        self.manager = AIModelManager()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            min_detection_confidence=config.FACE_DETECTION_CONFIDENCE
        )

        # Replaced frame_counter with time-based tracking
        self.last_yolo_time = 0.0
        self.last_yolo_boxes = []
        self.prediction_buffer = deque(maxlen=45)
        self.lstm_sequence_buffer = deque(maxlen=30)
        self.last_stable_step = 0
        self.last_who_time = 0.0
        

    def detect_ppe(self, frame):
        has_mask, has_hat = False, False
        now = time.time()

        # Execute heavy YOLO inference strictly on the time interval
        if (now - self.last_yolo_time) >= (1.0 / getattr(config, 'PPE_FPS', 5)) or not self.last_yolo_boxes:
            self.last_yolo_boxes = self.manager.run_yolo(frame)
            self.last_yolo_time = now

        # Always draw the most recently cached boxes so the UI doesn't flicker
        for box_data in self.last_yolo_boxes:
            class_name = box_data['name']
            x1, y1, x2, y2 = map(int, box_data['xyxy'])
            color = (0, 255, 0) if class_name == 'mask' else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if class_name == 'mask': has_mask = True
            elif class_name == 'hat': has_hat = True

        return frame, has_mask, has_hat

    def detect_face(self, frame_rgb):
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

    def predict_who_step(self, hand_landmarks_data):
        if not hand_landmarks_data or not hand_landmarks_data.multi_hand_landmarks:
            # FIX 1: Do NOT clear the buffer here! Just return the last known step.
            return self.last_stable_step
           
        sorted_hands = sorted(hand_landmarks_data.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
        current_features = []
        
        for hand in sorted_hands[:2]:
            wrist = hand.landmark[0]
            middle_base = hand.landmark[9]
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

        if len(sorted_hands) == 2:
            h1_w, h2_w = sorted_hands[0].landmark[0], sorted_hands[1].landmark[0]
            current_features.append(math.hypot(h1_w.x - h2_w.x, h1_w.y - h2_w.y))
            h1_i, h2_i = sorted_hands[0].landmark[8], sorted_hands[1].landmark[8]
            current_features.append(math.hypot(h1_i.x - h2_i.x, h1_i.y - h2_i.y))
        else:
            current_features.extend([1.0, 1.0])
           
        # --- CRITICAL FIX 2: Rate-limit appending to match the 10 FPS training data! ---
        now = time.time()
        if (now - getattr(self, 'last_feature_time', 0.0)) >= 0.1:
            self.lstm_sequence_buffer.append(current_features)
            self.last_feature_time = now
            
            # Only run inference if we actually added a new frame and the buffer is full
            if len(self.lstm_sequence_buffer) == 30:
                seq_array = np.array([list(self.lstm_sequence_buffer)], dtype=np.float32)
                pred = self.manager.run_who_lstm(seq_array)
                
                self.prediction_buffer.append(pred)
                self.last_stable_step = Counter(self.prediction_buffer).most_common(1)[0][0]
        
        return self.last_stable_step

    def get_smoothed_step(self):
        if not self.prediction_buffer:
            return 0
        most_common_step, _ = Counter(self.prediction_buffer).most_common(1)[0]
        return most_common_step

    def clear_buffer(self):
        self.prediction_buffer.clear()
        self.lstm_sequence_buffer.clear()
        self.last_stable_step = 0

    def cleanup(self):
        self.hands.close()
        self.face_detector.close()

AIModels = PerSinkAIState