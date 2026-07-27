import os
import cv2
import glob
import joblib
import pandas as pd
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
FRAMES_DIR = "./AI handwash/dataset-pskus/PSKUS_dataset/frames/trainval"
MODEL_SAVE_PATH = "who_rf_model.pkl"
MAX_SAMPLES_PER_CLASS = 15000  # Cap samples to prevent RAM overload and balance data

# Initialize MediaPipe in Static Image Mode
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

def extract_features_from_image(image_path):
    """Matches the logic in ai_models.py predict_who_step exactly."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
        
    # Sort hands from left to right just like the main app
    sorted_hands = sorted(results.multi_hand_landmarks, key=lambda h: h.landmark[0].x)
    
    features = []
    for hand in sorted_hands[:2]:
        wrist_x = hand.landmark[0].x
        wrist_y = hand.landmark[0].y
        wrist_z = hand.landmark[0].z
        
        for lm in hand.landmark:
            features.extend([
                lm.x - wrist_x,
                lm.y - wrist_y,
                lm.z - wrist_z
            ])
            
    # Pad to 126 features if only one hand is visible
    while len(features) < 126:
        features.append(0.0)
        
    return features

def main():
    dataset_features = []
    dataset_labels = []

    print("[INFO] Starting Feature Extraction with MediaPipe...")
    
    # Loop through WHO classes 0 through 6
    for class_id in range(7):
        folder_path = os.path.join(FRAMES_DIR, str(class_id))
        if not os.path.exists(folder_path):
            print(f"[WARN] Missing folder for class {class_id}")
            continue
            
        images = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        # Limit the number of frames processed per class to balance the dataset
        if len(images) > MAX_SAMPLES_PER_CLASS:
            images = images[:MAX_SAMPLES_PER_CLASS]
            
        print(f" -> Processing Class {class_id}: Extracting {len(images)} frames...")
        
        valid_frames = 0
        for img_path in images:
            features = extract_features_from_image(img_path)
            if features is not None:
                dataset_features.append(features)
                dataset_labels.append(class_id)
                valid_frames += 1
                
        print(f"    Found visible hands in {valid_frames} frames.")

    hands.close()

    if not dataset_features:
        print("[ERROR] No hands detected in any images. Check your folder paths.")
        return

    print("\n[INFO] Feature extraction complete. Training Random Forest Model...")
    
    # Split data to see how accurate the model is
    X_train, X_test, y_train, y_test = train_test_split(
        dataset_features, dataset_labels, test_size=0.2, random_state=42
    )
    
    # Train the Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Validate Accuracy
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[INFO] Model Training Complete! Accuracy on unseen frames: {accuracy * 100:.2f}%")
    
    # Save the model
    joblib.dump(rf_model, MODEL_SAVE_PATH)
    print(f"[SUCCESS] Saved model to {MODEL_SAVE_PATH}. You are ready to run main_app.py!")

if __name__ == "__main__":
    main()