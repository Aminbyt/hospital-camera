"""Updated WHO LSTM Training Script - RTMPose Version"""
import os
import glob
import math
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict

# --- NEW: IMPORT RTMPOSE INSTEAD OF MEDIAPIPE ---
from rtmlib import Hand # <-- FIXED IMPORT

# --- CONFIGURATION ---
FRAMES_DIR = "./AI handwash/dataset-pskus/PSKUS_dataset/frames/trainval"
ONNX_SAVE_PATH = "who_rtm_lstm_model.onnx"
MAX_SAMPLES_PER_CLASS = 15000
SEQ_LEN = 30       # 30 continuous frames (1 full second of motion)
INPUT_DIM = 128    # 126 (coords) + 2 (hand-to-hand distances)
HIDDEN_DIM = 64    # Lightweight hidden layer for fast CPU inference
NUM_CLASSES = 7
BATCH_SIZE = 64
EPOCHS = 25

# --- RTMPOSE INITIALIZATION ---
print("[INFO] Loading RTMPose Hand Tracker...")
hand_tracker = Hand(  # <-- FIXED INITIALIZATION
    mode='lightweight',
    backend='onnxruntime',
    device='cuda'
)

def extract_features_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # RTMPose uses BGR natively, no need to convert to RGB!
    keypoints, scores = hand_tracker(img) # <-- FIXED API CALL
    
    if keypoints is None or len(keypoints) == 0:
        return None
        
    frame_h, frame_w = img.shape[:2]
    
    # Create mock objects to match the 128-feature extraction logic
    class MockLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.z = 0.0

    class MockHand:
        def __init__(self, kp):
            self.landmark = [MockLandmark(pt[0]/frame_w, pt[1]/frame_h) for pt in kp]
            
    hands_list = []
    for kp in keypoints:
        if len(kp) == 21:
            hands_list.append(MockHand(kp))
            
    if not hands_list:
        return None

    sorted_hands = sorted(hands_list, key=lambda h: h.landmark[0].x)
    features = []
    
    # 1. Extract standard normalized coordinates for up to 2 hands
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
            
    # Pad coordinate features if only 1 hand is visible to reach 126
    while len(features) < 126:
        features.append(0.0)
        
    # 2. Extract Relative Distance Features (Crucial for WHO steps)
    if len(sorted_hands) == 2:
        h1_wrist = sorted_hands[0].landmark[0]
        h2_wrist = sorted_hands[1].landmark[0]
        wrist_dist = math.hypot(h1_wrist.x - h2_wrist.x, h1_wrist.y - h2_wrist.y)
        
        h1_index = sorted_hands[0].landmark[8]
        h2_index = sorted_hands[1].landmark[8]
        index_dist = math.hypot(h1_index.x - h2_index.x, h1_index.y - h2_index.y)
        
        features.extend([wrist_dist, index_dist])
    else:
        # If only one hand, apply maximum distance penalty (1.0 is max normalized space)
        features.extend([1.0, 1.0])
        
    return features


# --- PYTORCH CNN + LSTM ARCHITECTURE ---
class HandwashCNN_LSTM(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super(HandwashCNN_LSTM, self).__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        last_out = out[:, -1, :]
        return self.fc(last_out)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    print("[INFO] Starting Sequential Feature Extraction with RTMPose...")
    class_data = defaultdict(lambda: defaultdict(list))
   
    # 1. ADD THIS IMPORT HERE
    from tqdm import tqdm

    for class_id in range(7):
        folder_path = os.path.join(FRAMES_DIR, str(class_id))
        if not os.path.exists(folder_path): continue
        images = glob.glob(os.path.join(folder_path, "*.jpg"))[:MAX_SAMPLES_PER_CLASS]
        print(f"\n -> Processing Class {class_id}: {len(images)} frames...")

        # 2. WRAP 'images' WITH tqdm() LIKE THIS:
        for img_path in tqdm(images, desc=f"Class {class_id}", unit="frame"):
            basename = os.path.basename(img_path).replace(".jpg", "")
            parts = basename.split("_")
            try:
                frame_num = int(parts[1])
                snippet_id = "_".join(parts[2:])
            except:
                continue
            features = extract_features_from_image(img_path)
            if features:
                class_data[class_id][snippet_id].append((frame_num, features))

    # --- BUILD TEMPORAL SLIDING WINDOWS ---
    print(f"\n[INFO] Building {SEQ_LEN}-Frame Time Sequences...")
    X_seq, y_seq = [], []

    for class_id, snippets in class_data.items():
        for snippet_id, frames in snippets.items():
            frames.sort(key=lambda x: x[0])
            for i in range(len(frames) - SEQ_LEN + 1):
                window = [f[1] for f in frames[i : i + SEQ_LEN]]
                X_seq.append(window)
                y_seq.append(class_id)

    X = np.array(X_seq)
    y = np.array(y_seq)
    print(f"[INFO] Final Sequence Dataset Shape: {X.shape} (Samples x {SEQ_LEN} frames x {INPUT_DIM} features)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training CNN-LSTM on device: {device}...")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = HandwashCNN_LSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- TRAINING LOOP ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X = batch_X.to(device)
                    preds = torch.argmax(model(batch_X), dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(batch_y.numpy())
            acc = accuracy_score(all_targets, all_preds)
            print(f"  Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}%")

    # --- EXPORT TO ONNX ---
    print(f"\n[INFO] Exporting CNN-LSTM to ONNX format ({ONNX_SAVE_PATH})...")
    model.eval().to("cpu")
    
    dummy_input = torch.randn(1, SEQ_LEN, INPUT_DIM)
    
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_SAVE_PATH,
        input_names=["input_sequence"],
        output_names=["class_logits"],
        dynamic_axes={"input_sequence": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        opset_version=12
    )
    print(f"[SUCCESS] Saved ONNX model to {ONNX_SAVE_PATH}. Ready for main_app.py!")

if __name__ == "__main__":
    main()