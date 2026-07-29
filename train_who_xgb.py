import os
import cv2
import glob
import math
import joblib
import xgboost as xgb
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

# Configuration
FRAMES_DIR = "./AI handwash/dataset-pskus/PSKUS_dataset/frames/trainval"
MODEL_SAVE_PATH = "who_xgb_model.pkl"
MAX_SAMPLES_PER_CLASS = 15000 
VELOCITY_WINDOW = 5 # Calculate motion based on the frame from 5 steps ago

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

def extract_features_from_image(image_path):
    """Extracts scale-normalized hand landmarks."""
    img = cv2.imread(image_path)
    if img is None: return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks: return None
        
    sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
    features = []
    
    for hand in sorted_hands[:2]:
        wrist = hand.landmark[0]
        middle_base = hand.landmark[9]
        hand_size = math.hypot(wrist.x - middle_base.x, wrist.y - middle_base.y)
        if hand_size == 0: hand_size = 1.0
        
        for lm in hand.landmark:
            features.extend([
                (lm.x - wrist.x) / hand_size,
                (lm.y - wrist.y) / hand_size,
                (lm.z - wrist.z) / hand_size
            ])
            
    while len(features) < 126:
        features.append(0.0)
        
    return features

def main():
    print("[INFO] Starting Temporal Feature Extraction...")
    
    # We will group frames by their source video to calculate motion correctly
    # Structure: class_data[class_id][snippet_id] = [(frame_num, features), ...]
    class_data = defaultdict(lambda: defaultdict(list))
    
    for class_id in range(7):
        folder_path = os.path.join(FRAMES_DIR, str(class_id))
        if not os.path.exists(folder_path): continue
            
        images = glob.glob(os.path.join(folder_path, "*.jpg"))[:MAX_SAMPLES_PER_CLASS]
        print(f" -> Processing Class {class_id}: {len(images)} frames...")
        
        for img_path in images:
            # Parse filename: frame_5_snippet_1_2020-06-26...jpg
            basename = os.path.basename(img_path).replace(".jpg", "")
            parts = basename.split("_")
            try:
                frame_num = int(parts[1])
                snippet_id = "_".join(parts[2:]) # e.g., snippet_1_2020...
            except:
                continue
            
            features = extract_features_from_image(img_path)
            if features:
                class_data[class_id][snippet_id].append((frame_num, features))

    hands.close()

    # --- CALCULATE VELOCITY (MOTION DELTA) ---
    print("\n[INFO] Calculating Hand Velocity Vectors...")
    dataset_features = []
    dataset_labels = []

    for class_id, snippets in class_data.items():
        for snippet_id, frames in snippets.items():
            # Sort frames sequentially by time
            frames.sort(key=lambda x: x[0]) 
            
            for i in range(len(frames)):
                current_features = frames[i][1]
                
                # Get the features from 5 frames ago (or the oldest available)
                prev_index = max(0, i - VELOCITY_WINDOW)
                prev_features = frames[prev_index][1]
                
                # Velocity = Current Position - Previous Position
                velocity = [curr - prev for curr, prev in zip(current_features, prev_features)]
                
                # Final Array: 126 Positional Features + 126 Velocity Features = 252 Dimensions
                final_features = current_features + velocity
                
                dataset_features.append(final_features)
                dataset_labels.append(class_id)

    X = np.array(dataset_features)
    y = np.array(dataset_labels)
    
    print(f"[INFO] Final Dataset Shape: {X.shape} (Frames x Features)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n[INFO] Training XGBoost Model on GPU...")
    # Utilize your GPU for lightning-fast training
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=7,
            tree_method='hist',
            device='cuda',  # <-- FORCES GPU USAGE
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
    except Exception as e:
        print(f"[WARN] GPU Training failed, falling back to CPU: {e}")
        xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=7, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[INFO] Model Training Complete! Accuracy on unseen motion: {accuracy * 100:.2f}%")
    
    joblib.dump(xgb_model, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Saved model to {MODEL_SAVE_PATH}.")

if __name__ == "__main__":
    main()